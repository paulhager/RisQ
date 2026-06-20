# =====================================================================
# FILE: models/integrated_gradients.py
# PURPOSE:
#   - Compute Integrated Gradients (IG) for RepQuery at fixed horizons
#   - Aggregate attributions and save to disk
#
# SUPPORTED ATTRIBUTIONS:
#   - Tabular x
#   - ICD/MEDS *token embeddings after lookup/projection* (i.e. the continuous
#     token vectors that enter the transformer encoder), which yields per-code
#     token attribution conditional on the code being present.
#
# IMPORTANT NUANCE:
#   - This does NOT attribute to the discrete presence/absence indicator of a code
#     (the boolean selection step is non-differentiable).
#   - It DOES attribute to the model's continuous representation of that code token
#     as used by the encoder (projected embedding +/- temporal PE).
# =====================================================================

from __future__ import annotations

import os, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------
# Helper: collect full eval split once (CPU tensors)
# ---------------------------------------------------------------------
def _collect_eval_tensors(
    eval_loader: DataLoader,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Tensor]:
    """
    Collect full eval split (CPU tensors) so we can do target-specific sampling.

    Returns:
      xs_all:   (N, D) float with NaNs
      icds_all: (N, N_icd) or None
      meds_all: (N, N_meds) or None
      prog_all: (N, n_targets) prognosis_targets (same as batch[3])
    """
    xs_list: List[Tensor] = []
    icd_list: List[Optional[Tensor]] = []
    meds_list: List[Optional[Tensor]] = []
    prog_list: List[Tensor] = []

    for batch in eval_loader:
        x, icd, meds, prognosis_targets, *_ = batch
        xs_list.append(x.detach().cpu())
        icd_list.append(None if icd is None else icd.detach().cpu())
        meds_list.append(None if meds is None else meds.detach().cpu())
        prog_list.append(prognosis_targets.detach().cpu())

    xs_all = torch.cat(xs_list, dim=0)
    prog_all = torch.cat(prog_list, dim=0)

    icds_all: Optional[Tensor] = None
    if icd_list and icd_list[0] is not None:
        if any(t is None for t in icd_list):
            raise ValueError("Mixed ICD presence across batches; cannot align safely.")
        icds_all = torch.cat([t for t in icd_list if t is not None], dim=0)

    meds_all: Optional[Tensor] = None
    if meds_list and meds_list[0] is not None:
        if any(t is None for t in meds_list):
            raise ValueError("Mixed MEDS presence across batches; cannot align safely.")
        meds_all = torch.cat([t for t in meds_list if t is not None], dim=0)

    return xs_all, icds_all, meds_all, prog_all


# ---------------------------------------------------------------------
# Helper: case-control sampling indices for a single (target, horizon)
# ---------------------------------------------------------------------
def _sample_case_control_indices(
    prog_all: Tensor,
    target_idx: int,
    horizon_years: int,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Case-control contrast sampling (for interpretability-style ranking):
      - positives: valid & (0 < time_to_event < horizon_months)
      - negatives: valid & NOT positive (includes 0 or >= horizon, depending on encoding)
    We target ~half positives, half negatives. If positives are fewer than half,
    we include ALL positives and fill remaining with negatives.

    Returns:
      idx_all: sampled indices (shuffled)
      idx_pos: sampled positives (subset of idx_all)
      idx_neg: sampled negatives (subset of idx_all)
    """
    horizon_months = float(horizon_years) * 12.0

    y = prog_all[:, target_idx]  # (N,)
    valid = y != -1
    pos = valid & (y > 0) & (y < horizon_months)
    neg = valid & (~pos)

    pos_idx = torch.nonzero(pos, as_tuple=False).view(-1).cpu().numpy().astype(np.int64)
    neg_idx = torch.nonzero(neg, as_tuple=False).view(-1).cpu().numpy().astype(np.int64)

    rng = np.random.default_rng(int(seed))

    half = max_samples // 2

    if pos_idx.size == 0 and neg_idx.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if pos_idx.size <= half:
        take_pos = pos_idx  # all positives
    else:
        take_pos = rng.choice(pos_idx, size=half, replace=False)

    remaining = max_samples - int(take_pos.size)
    if remaining <= 0:
        take_neg = np.array([], dtype=np.int64)
    else:
        if neg_idx.size <= remaining:
            take_neg = neg_idx
        else:
            take_neg = rng.choice(neg_idx, size=remaining, replace=False)

    idx_all = np.concatenate([take_pos, take_neg], axis=0).astype(np.int64)
    rng.shuffle(idx_all)

    take_pos_set = set(map(int, take_pos.tolist()))
    take_neg_set = set(map(int, take_neg.tolist()))
    idx_pos = np.array([i for i in idx_all if int(i) in take_pos_set], dtype=np.int64)
    idx_neg = np.array([i for i in idx_all if int(i) in take_neg_set], dtype=np.int64)

    return idx_all, idx_pos, idx_neg


# ---------------------------------------------------------------------
# Baseline for x
# ---------------------------------------------------------------------
def _compute_x_baseline_from_train_loader(
    train_loader: DataLoader,
    device: torch.device,
    mode: str,
) -> Tensor:
    """
    Builds the baseline tensor for x.

    mode:
      - "zero": baseline is all zeros
      - "mean_train": per-feature mean over TRAIN (ignoring NaNs)
      - "median_train": per-feature median over TRAIN (ignoring NaNs) [more expensive; collects values]

    Returns:
      baseline_x: Tensor shaped (1, n_features)
    """
    x0, *_ = next(iter(train_loader))
    n_features = x0.shape[1]

    if mode == "zero":
        return torch.zeros(1, n_features, device=device, dtype=torch.float32)

    if mode == "mean_train":
        sum_ = torch.zeros(n_features, device=device, dtype=torch.float64)
        cnt_ = torch.zeros(n_features, device=device, dtype=torch.float64)

        for batch in train_loader:
            x, *_ = batch
            x = x.to(device).float()

            mask = ~torch.isnan(x)
            x_filled = torch.where(mask, x, torch.zeros_like(x))

            sum_ += x_filled.double().sum(dim=0)
            cnt_ += mask.double().sum(dim=0)

        mean = sum_ / torch.clamp(cnt_, min=1.0)
        return mean.float().unsqueeze(0)

    if mode == "median_train":
        values: List[List[float]] = [[] for _ in range(n_features)]
        for batch in train_loader:
            x, *_ = batch
            x = x.cpu().numpy()
            for j in range(n_features):
                col = x[:, j]
                col = col[~np.isnan(col)]
                values[j].extend(col.tolist())

        med = np.array([np.median(v) if len(v) else 0.0 for v in values], dtype=np.float32)
        return torch.from_numpy(med).to(device).unsqueeze(0)

    raise ValueError(f"Unknown ig_baseline_x={mode}")


# ---------------------------------------------------------------------
# IG for x
# ---------------------------------------------------------------------
def _integrated_gradients_x(
    model,
    x: Tensor,
    icd_multi_hot: Optional[Tensor],
    meds_multi_hot: Optional[Tensor],
    target_idx: int,
    horizon_years: int,
    baseline_x: Tensor,
    steps: int,
    output_type: str,
) -> Tensor:
    device = x.device
    B, D = x.shape

    obs = ~torch.isnan(x)  # (B, D)

    x0 = baseline_x.to(device).expand(B, D).clone()

    x_end = x.clone()
    x_end = torch.where(obs, x_end, x0)

    dx = x_end - x0
    total_grad = torch.zeros_like(x_end, dtype=torch.float32, device=device)

    for k in range(1, steps + 1):
        alpha = float(k) / float(steps)
        xk = (x0 + alpha * dx).detach()
        xk.requires_grad_(True)

        logits = model.predict_logits_at_horizon_years(
            xk, icd_multi_hot, meds_multi_hot, horizon_years=horizon_years
        )

        if output_type == "prob":
            logits = torch.sigmoid(logits)

        F = logits[:, target_idx].sum()

        grad_x = torch.autograd.grad(
            outputs=F,
            inputs=xk,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        total_grad += grad_x

    avg_grad = total_grad / float(steps)
    attr = dx * avg_grad

    attr = attr * obs.float()
    return attr


# ---------------------------------------------------------------------
# IG for pre-encoder token embeddings
# ---------------------------------------------------------------------
def _integrated_gradients_tokens(
    model,
    tokens: Tensor,
    padding_mask: Tensor,
    target_idx: int,
    horizon_years: int,
    steps: int,
    output_type: str,
    baseline_mode: str,
) -> Tensor:
    """
    Computes IG attributions for pre-encoder token embeddings (continuous) for a batch.

    Inputs:
      - tokens: (B, T, D) token embeddings that feed into the encoder
      - padding_mask: (B, T) True for padding positions

    Baseline:
      - "zero": all-zero token embeddings baseline

    Returns:
      - attr_tokens: (B, T, D)
    """
    device = tokens.device
    B, T, D = tokens.shape

    if baseline_mode != "zero":
        raise ValueError(f"Unsupported ig_baseline_tokens={baseline_mode} (use 'zero')")

    tok0 = torch.zeros((B, T, D), device=device, dtype=tokens.dtype)
    dtok = tokens - tok0

    total_grad = torch.zeros_like(tokens, dtype=torch.float32, device=device)

    for k in range(1, steps + 1):
        alpha = float(k) / float(steps)
        tok_k = (tok0 + alpha * dtok).detach()
        tok_k.requires_grad_(True)

        logits = model.predict_logits_at_horizon_years_from_tokens(
            tok_k, padding_mask, horizon_years=horizon_years
        )

        if output_type == "prob":
            logits = torch.sigmoid(logits)

        F = logits[:, target_idx].sum()

        grad_tok = torch.autograd.grad(
            outputs=F,
            inputs=tok_k,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        total_grad += grad_tok

    avg_grad = total_grad / float(steps)
    attr = dtok * avg_grad
    return attr


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------
def _resolve_eval_patient_ids(
    eval_loader: DataLoader,
    n_samples: int,
    patient_id_col: str,
) -> np.ndarray:
    """
    Resolve a stable patient identifier aligned with collected eval tensors.

    If eval loader is not sequential or the configured ID column is unavailable,
    falls back to local row indices [0..N-1].
    """
    try:
        sampler_name = type(getattr(eval_loader, "sampler", None)).__name__.lower()
        is_sequential = "sequential" in sampler_name

        dataset = getattr(eval_loader, "dataset", None)
        adata = getattr(dataset, "data", None)
        obs = getattr(adata, "obs", None)

        if (not is_sequential) or (obs is None) or (patient_id_col not in obs.columns):
            return np.arange(n_samples, dtype=np.int64)

        ids = obs[patient_id_col].to_numpy()
        if ids.shape[0] != n_samples:
            return np.arange(n_samples, dtype=np.int64)

        return ids
    except Exception:
        return np.arange(n_samples, dtype=np.int64)


def _py_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def _ig_args_snapshot(args) -> Dict[str, object]:
    """Collect effective IG-related args for logging/debugging."""
    snapshot: Dict[str, object] = {}
    for key in sorted(vars(args).keys()):
        if not str(key).startswith("ig_"):
            continue
        val = getattr(args, key)
        if hasattr(val, "value"):  # enums
            snapshot[key] = val.value
        elif isinstance(val, np.ndarray):
            snapshot[key] = val.tolist()
        elif isinstance(val, (list, tuple)):
            snapshot[key] = list(val)
        else:
            snapshot[key] = val
    return snapshot


def _validate_missing_aggregation_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in {"zero", "nan"}:
        raise ValueError(
            f"Unsupported ig_missing_aggregation={mode}. Use 'zero' or 'nan'."
        )
    return mode


def _select_aggregation_value(
    zero_value: float,
    nan_value: float,
    missing_aggregation: str,
) -> float:
    if missing_aggregation == "nan":
        return nan_value
    return zero_value


def _gaussian_quantile_inverse(
    z: np.ndarray,
    x_sorted: np.ndarray,
    z_sorted: np.ndarray,
) -> np.ndarray:
    out = np.full_like(z, np.nan, dtype=float)
    mask = ~np.isnan(z)
    if x_sorted.size == 0 or z_sorted.size == 0:
        return out
    z_clip = np.clip(z[mask], z_sorted[0], z_sorted[-1])
    out[mask] = np.interp(z_clip, z_sorted, x_sorted)
    return out


def _resolve_feature_column(
    adata_view,
    field_name: str,
) -> np.ndarray:
    idx = np.where(np.asarray(adata_view.var_names) == field_name)[0]
    if len(idx) != 1:
        raise ValueError(f"Feature field not found uniquely in adata.var_names: {field_name}")
    values = np.asarray(adata_view.X[:, int(idx[0])]).reshape(-1)
    return values.astype(float)


def _normalize_stratify_by(raw_values) -> List[str]:
    normalized: List[str] = []
    for value in raw_values or []:
        key = str(value).strip().lower()
        if not key:
            continue
        if key not in {"sex", "age_bin", "sex_age_bin"}:
            raise ValueError(
                f"Unsupported ig_stratify_by entry '{value}'. Supported values: 'sex', 'age_bin', 'sex_age_bin'."
            )
        if key not in normalized:
            normalized.append(key)
    return normalized


def _format_age_bin_labels(edges: List[float]) -> List[str]:
    labels: List[str] = []
    for start, end in zip(edges[:-2], edges[1:-1]):
        labels.append(f"{int(start)}-{int(end)}")
    last_start = edges[-2]
    labels.append(f"{int(last_start)}+")
    return labels


def _prepare_eval_stratification(
    adata_view,
    args,
    n_samples: int,
) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    stratify_by = _normalize_stratify_by(getattr(args, "ig_stratify_by", []))
    if not stratify_by:
        return {}

    prepared: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    sex_labels: Optional[np.ndarray] = None
    age_bin_labels: Optional[np.ndarray] = None
    age_levels: Optional[List[str]] = None

    need_sex = "sex" in stratify_by or "sex_age_bin" in stratify_by
    need_age_bin = "age_bin" in stratify_by or "sex_age_bin" in stratify_by

    if need_sex:
        sex_field = str(getattr(args, "ig_sex_field", "31-0.0"))
        sex_values = _resolve_feature_column(adata_view, sex_field)
        sex_labels = np.full(n_samples, "", dtype=object)
        sex_labels[np.isclose(sex_values, 0.0, equal_nan=False)] = "Female"
        sex_labels[np.isclose(sex_values, 1.0, equal_nan=False)] = "Male"
        if "sex" in stratify_by:
            prepared["sex"] = (sex_labels, ["Female", "Male"])

    if need_age_bin:
        age_field = str(getattr(args, "ig_age_field", "21003-0.0"))
        gq = getattr(adata_view, "uns", {}).get("gaussian_quantile_norm", {})
        if age_field not in gq:
            raise ValueError(
                f"ig_age_field '{age_field}' is missing from adata.uns['gaussian_quantile_norm']."
            )
        age_values = _resolve_feature_column(adata_view, age_field)
        age_map = gq[age_field]
        raw_age = _gaussian_quantile_inverse(
            age_values,
            np.asarray(age_map["x_sorted"], dtype=float),
            np.asarray(age_map["z_sorted"], dtype=float),
        )
        age_edges = [float(v) for v in list(getattr(args, "ig_age_bin_edges", []))]
        if len(age_edges) < 2:
            raise ValueError("ig_age_bin_edges must contain at least two values.")
        edges_with_inf = list(age_edges)
        if not np.isinf(edges_with_inf[-1]):
            edges_with_inf.append(float("inf"))
        age_levels = _format_age_bin_labels(edges_with_inf)
        age_bins = pd.cut(
            raw_age,
            bins=edges_with_inf,
            labels=age_levels,
            right=False,
            include_lowest=True,
        )
        age_bin_labels = np.asarray(age_bins.astype(object))
        if "age_bin" in stratify_by:
            prepared["age_bin"] = (age_bin_labels, age_levels)

    if "sex_age_bin" in stratify_by:
        if sex_labels is None or age_bin_labels is None or age_levels is None:
            raise ValueError("sex_age_bin stratification requires both sex and age_bin labels.")
        joint_labels = np.full(n_samples, "", dtype=object)
        valid = pd.notna(sex_labels) & (sex_labels != "") & pd.notna(age_bin_labels) & (age_bin_labels != "")
        joint_labels[valid] = np.asarray(
            [f"{sex}__{age}" for sex, age in zip(sex_labels[valid], age_bin_labels[valid])],
            dtype=object,
        )
        ordered_levels = [f"{sex}__{age}" for sex in ["Female", "Male"] for age in age_levels]
        prepared["sex_age_bin"] = (joint_labels, ordered_levels)

    return prepared


def _iter_group_views(
    view_name: str,
    view_mask: np.ndarray,
    sampled_strata: Dict[str, Tuple[np.ndarray, List[str]]],
):
    yield ("none", "all", view_name, view_mask)
    for stratify_by, (labels, ordered_levels) in sampled_strata.items():
        valid = pd.notna(labels) & (labels != "")
        for level in ordered_levels:
            group_mask = view_mask & valid & (labels == level)
            if np.any(group_mask):
                yield (stratify_by, str(level), view_name, group_mask)


def run_repquery_integrated_gradients(
    model,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    args,
) -> str:
    """
    Main entry point called from train.py (post-training evaluation section).

    Implements:
      - Case-control contrast ranking sampling per (target, horizon)
      - 3 views saved from same run: overall / pos / neg
      - Supports:
          * tabular x IG in raw-x space and/or token space
          * ICD/MEDS token-embedding IG (post-lookup/projection tokens entering encoder)
      - Optional patient-level output (single sidecar file)

    Returns:
      out_path: path to aggregated artifact
    """
    device = next(model.parameters()).device
    model.eval()

    ig_attr_x = bool(getattr(args, "ig_attr_x", True))
    ig_attr_icd = bool(getattr(args, "ig_attr_icd", False))
    ig_attr_meds = bool(getattr(args, "ig_attr_meds", False))
    ig_x_attr_mode = str(getattr(args, "ig_x_attr_mode", "token")).lower()

    if ig_x_attr_mode not in {"raw", "token", "both"}:
        raise ValueError(f"Unsupported ig_x_attr_mode={ig_x_attr_mode}. Use 'raw', 'token', or 'both'.")

    ig_attr_x_raw = ig_attr_x and ig_x_attr_mode in {"raw", "both"}
    ig_attr_x_token = ig_attr_x and ig_x_attr_mode in {"token", "both"}
    ig_attr_tokens_any = bool(ig_attr_x_token or ig_attr_icd or ig_attr_meds)

    if not (ig_attr_x_raw or ig_attr_tokens_any):
        raise ValueError("Nothing to attribute: set ig_attr_x and/or ig_attr_icd and/or ig_attr_meds.")

    # ---- Output directory ----
    out_dir = getattr(args, "ig_out_dir", None)
    if out_dir is None:
        out_dir = os.path.join(getattr(args, "checkpoint_dir_path", "."), "integrated_gradients")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Effective IG config snapshot (post-merge, post-overrides) ----
    ig_cfg = _ig_args_snapshot(args)
    print("[IG CONFIG] Effective IG args:\n" + json.dumps(ig_cfg, indent=2, sort_keys=True, default=str))
    ig_cfg_path = os.path.join(out_dir, f"ig_run_config_{getattr(args, 'ig_split', 'val')}.json")
    with open(ig_cfg_path, "w") as f:
        json.dump(ig_cfg, f, indent=2, sort_keys=True, default=str)

    # ---- Baselines ----
    baseline_x: Optional[Tensor] = None
    if ig_attr_x_raw:
        baseline_x = _compute_x_baseline_from_train_loader(
            train_loader=train_loader,
            device=device,
            mode=str(getattr(args, "ig_baseline_x", "mean_train")),
        )

    baseline_tokens_mode = str(getattr(args, "ig_baseline_tokens", "zero"))

    # ---- Always focus on the last 588 targets (same convention as evaluate_logits) ----
    n_targets_total = int(model.icd_prognosis_embeddings.shape[0])
    print("n_targets_total(model) =", n_targets_total)

    if n_targets_total < 588:
        raise ValueError(f"Model has only {n_targets_total} targets, cannot take last 588.")
    target_offset = n_targets_total - 588  # local idx 0..587 -> full idx target_offset..end

    # ---- Targets selection (ALWAYS in last-588 index space: 0..587) ----
    targets_mode = str(getattr(args, "ig_targets_mode", "all"))
    if targets_mode == "all":
        target_indices = list(range(588))
    elif targets_mode == "list":
        target_indices = list(getattr(args, "ig_targets_list", []) or [])
    else:
        raise ValueError(f"Unsupported ig_targets_mode: {targets_mode}")

    if len(target_indices) == 0:
        raise ValueError("No targets selected for IG (target list is empty).")

    bad = [tt for tt in target_indices if int(tt) < 0 or int(tt) >= 588]
    if bad:
        raise ValueError(f"ig_targets_list must use 0..587 (last-588 space). Bad: {bad[:10]}")

    horizons = list(getattr(args, "ig_horizons_years", [2, 5, 10, 15]))
    steps = int(getattr(args, "ig_steps", 20))
    output_type = str(getattr(args, "ig_output_type", "logit"))
    missing_aggregation = _validate_missing_aggregation_mode(
        getattr(args, "ig_missing_aggregation", "zero")
    )

    # ---- Sampling ----
    max_samples = int(getattr(args, "ig_num_samples", 2000))
    ig_seed = int(getattr(args, "ig_seed", 2024))

    # ---- Batching ----
    ig_batch_size = int(getattr(args, "ig_batch_size", 32))
    # Optional: separate batch size for token IG; defaults to ig_batch_size
    ig_token_batch_size = int(getattr(args, "ig_token_batch_size", ig_batch_size))
    ig_save_patient_level = bool(getattr(args, "ig_save_patient_level", False))
    ig_patient_topk = int(getattr(args, "ig_patient_topk", 100))
    patient_id_col = str(getattr(args, "ig_patient_id_col", "global_row"))

    # ---- Collect full eval once (CPU) ----
    xs_all, icds_all, meds_all, prog_all = _collect_eval_tensors(eval_loader)
    print("prog_all.shape =", tuple(prog_all.shape))
    eval_patient_ids_all = _resolve_eval_patient_ids(
        eval_loader=eval_loader,
        n_samples=int(xs_all.shape[0]),
        patient_id_col=patient_id_col,
    )

    # ---- Feature names ----
    adata_view = eval_loader.dataset.data
    feature_names = list(map(str, adata_view.var.index))
    n_features = len(feature_names)
    eval_strata = _prepare_eval_stratification(
        adata_view=adata_view,
        args=args,
        n_samples=int(xs_all.shape[0]),
    )

    rows_agg: List[Dict[str, object]] = []
    rows_patient: List[Dict[str, object]] = []

    # ---- Load id->name mappings (ADD HERE) ----
    def invert_mapping(path):
        with open(path, "r") as f:
            m = json.load(f)  # code/name -> row
        inv = {}
        for k, v in m.items():
            inv[int(v)] = str(k)
        return inv

    icd_id_to_name = None
    med_id_to_name = None

    if ig_attr_icd:
        path = os.path.join(args.data_root_path, args.icd_codes_to_row_name)
        icd_id_to_name = invert_mapping(path)

    if ig_attr_meds:
        path = os.path.join(args.data_root_path, args.meds_embeddings_codes_to_row_name)
        med_id_to_name = invert_mapping(path)

    # ---- Loop horizons/targets ----
    for h in horizons:
        h = int(h)
        for t in target_indices:
            t = int(t)
            t_full = t + target_offset

            # Unique deterministic seed per (h, t)
            seed_th = int(ig_seed + 100000 * h + t)

            # decide which index space prog_all uses
            M_prog = int(prog_all.shape[1])
            T_model = int(model.icd_prognosis_embeddings.shape[0])

            if M_prog == 588:
                t_prog = t               # prog_all is last-588
            elif M_prog == T_model:
                t_prog = t_full          # prog_all is full target space
            else:
                raise ValueError(
                    f"Target dim mismatch: prog_all has {M_prog} targets, model has {T_model}."
                )

            idx_all, idx_pos, idx_neg = _sample_case_control_indices(
                prog_all=prog_all,
                target_idx=t_prog,
                horizon_years=h,
                max_samples=max_samples,
                seed=seed_th,
            )

            if idx_all.size == 0:
                continue

            # Subset sampled tensors
            xs = xs_all[idx_all].to(device).float()
            icds = icds_all[idx_all].to(device) if icds_all is not None else None
            meds = meds_all[idx_all].to(device) if meds_all is not None else None
            patient_ids = eval_patient_ids_all[idx_all]
            sampled_strata = {
                name: (labels[idx_all], ordered_levels)
                for name, (labels, ordered_levels) in eval_strata.items()
            }

            if icds is None:
                print("[IG SANITY] icds is None")
            else:
                icds_b = icds.detach().float()
                n_pat, n_codes = icds_b.shape
                n_with_any = int((icds_b.sum(dim=1) > 0).sum().item())
                total_ones = int(icds_b.sum().item())
                topk_idx = torch.topk(icds_b.sum(dim=0), k=min(10, n_codes)).indices.detach().cpu().tolist()
                print(
                    f"[IG SANITY] ICD multi-hot: shape={tuple(icds.shape)} "
                    f"total_ones={total_ones} patients_with_any={n_with_any}/{n_pat}"
                )
                print(f"[IG SANITY] ICD top10 code-idx by freq (sampled subset): {topk_idx}")

            if meds is None:
                print("[IG SANITY] meds is None")
            else:
                meds_b = meds.detach().float()
                n_pat, n_codes = meds_b.shape
                n_with_any = int((meds_b.sum(dim=1) > 0).sum().item())
                total_ones = int(meds_b.sum().item())
                topk_idx = torch.topk(meds_b.sum(dim=0), k=min(10, n_codes)).indices.detach().cpu().tolist()
                print(
                    f"[IG SANITY] MEDS multi-hot: shape={tuple(meds.shape)} "
                    f"total_ones={total_ones} patients_with_any={n_with_any}/{n_pat}"
                )
                print(f"[IG SANITY] MEDS top10 code-idx by freq (sampled subset): {topk_idx}")

            # 3 views (masks in idx_all order)
            pos_set = set(map(int, idx_pos.tolist()))
            neg_set = set(map(int, idx_neg.tolist()))
            view_mask_overall = np.ones(len(idx_all), dtype=bool)
            view_mask_pos = np.array([int(i) in pos_set for i in idx_all], dtype=bool)
            view_mask_neg = np.array([int(i) in neg_set for i in idx_all], dtype=bool)
            views = [
                ("overall", view_mask_overall),
                ("pos", view_mask_pos),
                ("neg", view_mask_neg),
            ]

            # Build tokens per sampled set (required to also subset meta)
            tokens: Optional[Tensor] = None
            padding_mask: Optional[Tensor] = None
            icd_code_ids: Optional[Tensor] = None
            meds_code_ids: Optional[Tensor] = None
            x_feature_ids: Optional[Tensor] = None

            if ig_attr_tokens_any:
                if not hasattr(model, "build_preencoder_tokens"):
                    raise ValueError("Token IG requested, but model.build_preencoder_tokens(...) is missing.")
                if not hasattr(model, "predict_logits_at_horizon_years_from_tokens"):
                    raise ValueError(
                        "Token IG requested, but model.predict_logits_at_horizon_years_from_tokens(...) is missing."
                    )

                tokens, padding_mask, meta = model.build_preencoder_tokens(
                    xs, icds, meds, include_cls=True, deterministic=True
                )

                tokens = tokens.to(device)
                padding_mask = padding_mask.to(device)

                icd_code_ids = meta.get("icd_code_ids", None)
                meds_code_ids = meta.get("meds_code_ids", None)
                x_feature_ids = meta.get("x_feature_ids", None)

                if ig_attr_icd and icd_code_ids is None:
                    raise ValueError("ig_attr_icd=True but meta['icd_code_ids'] missing.")
                if ig_attr_meds and meds_code_ids is None:
                    raise ValueError("ig_attr_meds=True but meta['meds_code_ids'] missing.")
                if ig_attr_x_token and x_feature_ids is None:
                    raise ValueError("ig_attr_x in token mode requires meta['x_feature_ids'].")

                if isinstance(icd_code_ids, torch.Tensor):
                    icd_code_ids = icd_code_ids.to(device)
                if isinstance(meds_code_ids, torch.Tensor):
                    meds_code_ids = meds_code_ids.to(device)
                if isinstance(x_feature_ids, torch.Tensor):
                    x_feature_ids = x_feature_ids.to(device)

            # -------------------------
            # IG for x in raw-x space (legacy mode)
            # -------------------------
            if ig_attr_x_raw:
                if baseline_x is None:
                    raise ValueError("ig_attr_x=True but baseline_x is None (should not happen).")

                attr_x_chunks: List[Tensor] = []
                N = xs.shape[0]
                obs_x = ~torch.isnan(xs)

                for s in range(0, N, ig_batch_size):
                    e = min(s + ig_batch_size, N)

                    x_b = xs[s:e]
                    icd_b = icds[s:e] if icds is not None else None
                    meds_b = meds[s:e] if meds is not None else None

                    attr_b = _integrated_gradients_x(
                        model=model,
                        x=x_b,
                        icd_multi_hot=icd_b,
                        meds_multi_hot=meds_b,
                        target_idx=t_full,
                        horizon_years=h,
                        baseline_x=baseline_x,
                        steps=steps,
                        output_type=output_type,
                    )  # (B, D)

                    attr_x_chunks.append(attr_b)

                    del attr_b, x_b, icd_b, meds_b
                    torch.cuda.empty_cache()

                attr_x = torch.cat(attr_x_chunks, dim=0)  # (N, D)
                del attr_x_chunks
                torch.cuda.empty_cache()

                topk_feat = int(getattr(args, "ig_topk_features", 50))
                baseline_x_mode = str(getattr(args, "ig_baseline_x", "mean_train"))

                def emit_x_view(
                    view_name: str,
                    mask: np.ndarray,
                    stratify_by: str = "none",
                    stratum: str = "all",
                ) -> None:
                    if mask.sum() == 0:
                        return
                    m = torch.from_numpy(mask.astype(np.bool_)).to(attr_x.device)
                    mean_abs_zero = attr_x[m].abs().mean(dim=0).detach().cpu().numpy()
                    mean_ig_zero = attr_x[m].mean(dim=0).detach().cpu().numpy()
                    obs_view = obs_x[m]
                    den = obs_view.sum(dim=0).clamp_min(1).to(attr_x.dtype)
                    mean_abs_nan = (
                        attr_x[m].abs().sum(dim=0) / den
                    ).detach().cpu().numpy()
                    mean_ig_nan = (
                        attr_x[m].sum(dim=0) / den
                    ).detach().cpu().numpy()
                    if missing_aggregation == "nan":
                        order_metric = mean_abs_nan
                    else:
                        order_metric = mean_abs_zero
                    order = np.argsort(-order_metric)[:topk_feat]
                    for j in order:
                        mean_abs_zero_j = float(mean_abs_zero[int(j)])
                        mean_ig_zero_j = float(mean_ig_zero[int(j)])
                        mean_abs_nan_j = float(mean_abs_nan[int(j)])
                        mean_ig_nan_j = float(mean_ig_nan[int(j)])
                        rows_agg.append(
                            {
                                "view": view_name,
                                "attribution_type": "x_feature",
                                "attribution_space": "raw_x",
                                "target_idx": t,
                                "target_idx_full": t_full,
                                "horizon_years": h,
                                "feature": feature_names[int(j)],
                                "feature_idx": int(j),
                                "code_id": None,
                                "code_name": None,
                                "mean_abs_ig": _select_aggregation_value(
                                    mean_abs_zero_j, mean_abs_nan_j, missing_aggregation
                                ),
                                "mean_ig": _select_aggregation_value(
                                    mean_ig_zero_j, mean_ig_nan_j, missing_aggregation
                                ),
                                "mean_abs_ig_zero": mean_abs_zero_j,
                                "mean_ig_zero": mean_ig_zero_j,
                                "mean_abs_ig_nan": mean_abs_nan_j,
                                "mean_ig_nan": mean_ig_nan_j,
                                "n_samples": int(mask.sum()),
                                "n_present_samples": int(obs_view[:, int(j)].sum().item()),
                                "stratify_by": stratify_by,
                                "stratum": stratum,
                                "steps": steps,
                                "baseline": baseline_x_mode,
                                "output_type": output_type,
                                "sampling": "case_control",
                                "seed": seed_th,
                            }
                        )

                for view_name, view_mask in views:
                    for stratify_by, stratum, grouped_view_name, grouped_mask in _iter_group_views(
                        view_name=view_name,
                        view_mask=view_mask,
                        sampled_strata=sampled_strata,
                    ):
                        emit_x_view(
                            view_name=grouped_view_name,
                            mask=grouped_mask,
                            stratify_by=stratify_by,
                            stratum=stratum,
                        )

                if ig_save_patient_level:
                    attr_x_cpu = attr_x.detach().cpu().numpy()
                    for view_name, view_mask in views:
                        idxs = np.nonzero(view_mask.astype(np.bool_))[0]
                        for ridx in idxs:
                            signed = attr_x_cpu[ridx]
                            abs_v = np.abs(signed)
                            if ig_patient_topk > 0:
                                keep = np.argsort(-abs_v)[:ig_patient_topk]
                            else:
                                keep = np.argsort(-abs_v)
                            pid = _py_scalar(patient_ids[ridx])
                            for rank, j in enumerate(keep.tolist(), start=1):
                                if abs_v[j] == 0.0:
                                    continue
                                rows_patient.append(
                                    {
                                        "patient_id": pid,
                                        "patient_id_col": patient_id_col,
                                        "view": view_name,
                                        "attribution_type": "x_feature",
                                        "attribution_space": "raw_x",
                                        "target_idx": t,
                                        "target_idx_full": t_full,
                                        "horizon_years": h,
                                        "feature": feature_names[int(j)],
                                        "feature_idx": int(j),
                                        "code_id": None,
                                        "code_name": None,
                                        "ig": float(signed[j]),
                                        "abs_ig": float(abs_v[j]),
                                        "rank_in_patient": int(rank),
                                        "steps": steps,
                                        "baseline": baseline_x_mode,
                                        "output_type": output_type,
                                        "sampling": "case_control",
                                        "seed": seed_th,
                                    }
                                )

                # free per (h,t) to keep memory stable
                del attr_x, obs_x
                torch.cuda.empty_cache()

            # -------------------------
            # IG in token space (ICD/MEDS and optional tabular token view)
            # -------------------------
            if ig_attr_tokens_any:
                if tokens is None or padding_mask is None:
                    raise ValueError("Token IG requested but tokens/padding_mask are None (should not happen).")

                topk_codes = int(getattr(args, "ig_topk_codes", 100))
                topk_feat = int(getattr(args, "ig_topk_features", 50))
                baseline_token = baseline_tokens_mode

                def emit_view_streaming(
                    view_name: str,
                    mask: np.ndarray,
                    ids_tensor: Tensor,
                    attr_type: str,
                    topk_limit: int,
                    stratify_by: str = "none",
                    stratum: str = "all",
                ) -> None:
                    """
                    Aggregation with absent entity contribution = 0.

                    For each sample n and id c:
                      S_{n,c} = sum_{t: id_{n,t}=c} score(n,t)

                    We compute:
                      - score_signed(n,t) = sum_d IG(n,t,d)
                      - score_abs(n,t)    = sum_d |IG(n,t,d)|
                    """
                    if mask.sum() == 0:
                        return

                    device = tokens.device
                    N_view = int(mask.sum())

                    sum_signed_by_id: Dict[int, float] = {}
                    sum_abs_by_id: Dict[int, float] = {}
                    count_by_id: Dict[int, int] = {}

                    idxs = np.nonzero(mask.astype(np.bool_))[0]

                    for start in range(0, len(idxs), ig_token_batch_size):
                        chunk_pos = idxs[start : start + ig_token_batch_size]
                        chunk_pos_t = torch.from_numpy(chunk_pos.astype(np.int64)).to(device)

                        tok_b = tokens.index_select(0, chunk_pos_t)
                        pad_b = padding_mask.index_select(0, chunk_pos_t)
                        ids_b = ids_tensor.index_select(0, chunk_pos_t)

                        attr_b = _integrated_gradients_tokens(
                            model=model,
                            tokens=tok_b,
                            padding_mask=pad_b,
                            target_idx=t_full,
                            horizon_years=h,
                            steps=steps,
                            output_type=output_type,
                            baseline_mode=baseline_tokens_mode,
                        )  # (B, T, D)

                        # Per-token scalar scores
                        score_signed_b = attr_b.sum(dim=-1)    # (B, T), signed
                        score_abs_b = attr_b.abs().sum(dim=-1) # (B, T), magnitude

                        mask_b = (~pad_b).float()
                        score_signed_b *= mask_b
                        score_abs_b *= mask_b

                        score_signed_cpu = score_signed_b.detach().cpu().numpy()
                        score_abs_cpu = score_abs_b.detach().cpu().numpy()
                        ids_cpu = ids_b.detach().cpu().numpy()
                        patient_chunk = patient_ids[chunk_pos]

                        Bv, TT = ids_cpu.shape

                        for i in range(Bv):
                            per_signed: Dict[int, float] = {}
                            per_abs: Dict[int, float] = {}
                            present_ids = set()

                            for k in range(TT):
                                cid = int(ids_cpu[i, k])
                                if cid < 0:
                                    continue
                                present_ids.add(cid)

                                s_sgn = float(score_signed_cpu[i, k])
                                s_abs = float(score_abs_cpu[i, k])

                                if s_sgn != 0.0:
                                    per_signed[cid] = per_signed.get(cid, 0.0) + s_sgn
                                if s_abs != 0.0:
                                    per_abs[cid] = per_abs.get(cid, 0.0) + s_abs

                            for cid, v in per_signed.items():
                                sum_signed_by_id[cid] = sum_signed_by_id.get(cid, 0.0) + v
                            for cid, v in per_abs.items():
                                sum_abs_by_id[cid] = sum_abs_by_id.get(cid, 0.0) + v
                            if missing_aggregation == "nan":
                                for cid in present_ids:
                                    count_by_id[cid] = count_by_id.get(cid, 0) + 1

                            if ig_save_patient_level and len(per_abs) > 0:
                                ranked = sorted(per_abs.items(), key=lambda x: -x[1])
                                if ig_patient_topk > 0:
                                    ranked = ranked[:ig_patient_topk]
                                pid = _py_scalar(patient_chunk[i])
                                for rank, (cid, abs_val) in enumerate(ranked, start=1):
                                    signed_val = per_signed.get(cid, 0.0)
                                    if attr_type == "x_feature":
                                        fname = feature_names[int(cid)] if 0 <= int(cid) < n_features else None
                                        cname = None
                                        code_id = None
                                        feature_idx = int(cid)
                                    elif attr_type == "icd_token":
                                        fname = None
                                        cname = icd_id_to_name.get(int(cid)) if icd_id_to_name is not None else None
                                        code_id = int(cid)
                                        feature_idx = None
                                    else:
                                        fname = None
                                        cname = med_id_to_name.get(int(cid)) if med_id_to_name is not None else None
                                        code_id = int(cid)
                                        feature_idx = None

                                    rows_patient.append(
                                        {
                                            "patient_id": pid,
                                            "patient_id_col": patient_id_col,
                                            "view": view_name,
                                            "attribution_type": attr_type,
                                            "attribution_space": "token",
                                            "target_idx": t,
                                            "target_idx_full": t_full,
                                            "horizon_years": h,
                                            "feature": fname,
                                            "feature_idx": feature_idx,
                                            "code_id": code_id,
                                            "code_name": cname,
                                            "ig": float(signed_val),
                                            "abs_ig": float(abs_val),
                                            "rank_in_patient": int(rank),
                                            "steps": steps,
                                            "baseline": baseline_token,
                                            "output_type": output_type,
                                            "sampling": "case_control",
                                            "seed": seed_th,
                                        }
                                    )

                        del attr_b, tok_b, pad_b, ids_b, chunk_pos_t
                        torch.cuda.empty_cache()

                    den = max(N_view, 1)

                    items = []
                    for cid in sum_abs_by_id.keys():
                        den_nan = max(count_by_id.get(cid, 0), 1)
                        mean_abs_zero = sum_abs_by_id.get(cid, 0.0) / den
                        mean_sgn_zero = sum_signed_by_id.get(cid, 0.0) / den
                        mean_abs_nan = sum_abs_by_id.get(cid, 0.0) / den_nan
                        mean_sgn_nan = sum_signed_by_id.get(cid, 0.0) / den_nan
                        if missing_aggregation == "nan":
                            order_metric = mean_abs_nan
                        else:
                            order_metric = mean_abs_zero
                        items.append(
                            (
                                cid,
                                mean_abs_zero,
                                mean_sgn_zero,
                                mean_abs_nan,
                                mean_sgn_nan,
                                int(count_by_id.get(cid, 0)),
                                order_metric,
                            )
                        )

                    items.sort(key=lambda x: -x[6])  # rank by selected magnitude

                    for cid, mean_abs_zero, mean_sgn_zero, mean_abs_nan, mean_sgn_nan, n_present_samples, _ in items[:topk_limit]:
                        if attr_type == "x_feature":
                            rows_agg.append(
                                {
                                    "view": view_name,
                                    "attribution_type": "x_feature",
                                    "attribution_space": "token",
                                    "target_idx": t,
                                    "target_idx_full": t_full,
                                    "horizon_years": h,
                                    "feature": feature_names[int(cid)] if 0 <= int(cid) < n_features else None,
                                    "feature_idx": int(cid),
                                    "code_id": None,
                                    "code_name": None,
                                    "mean_abs_ig": _select_aggregation_value(
                                        float(mean_abs_zero), float(mean_abs_nan), missing_aggregation
                                    ),
                                    "mean_ig": _select_aggregation_value(
                                        float(mean_sgn_zero), float(mean_sgn_nan), missing_aggregation
                                    ),
                                    "mean_abs_ig_zero": float(mean_abs_zero),
                                    "mean_ig_zero": float(mean_sgn_zero),
                                    "mean_abs_ig_nan": float(mean_abs_nan),
                                    "mean_ig_nan": float(mean_sgn_nan),
                                    "n_samples": int(N_view),
                                    "n_present_samples": int(n_present_samples),
                                    "stratify_by": stratify_by,
                                    "stratum": stratum,
                                    "steps": steps,
                                    "baseline": baseline_token,
                                    "output_type": output_type,
                                    "sampling": "case_control",
                                    "seed": seed_th,
                                }
                            )
                        elif attr_type == "icd_token":
                            rows_agg.append(
                                {
                                    "view": view_name,
                                    "attribution_type": "icd_token",
                                    "attribution_space": "token",
                                    "target_idx": t,
                                    "target_idx_full": t_full,
                                    "horizon_years": h,
                                    "feature": None,
                                    "feature_idx": None,
                                    "code_id": int(cid),
                                    "code_name": icd_id_to_name.get(int(cid)) if icd_id_to_name is not None else None,
                                    "mean_abs_ig": _select_aggregation_value(
                                        float(mean_abs_zero), float(mean_abs_nan), missing_aggregation
                                    ),
                                    "mean_ig": _select_aggregation_value(
                                        float(mean_sgn_zero), float(mean_sgn_nan), missing_aggregation
                                    ),
                                    "mean_abs_ig_zero": float(mean_abs_zero),
                                    "mean_ig_zero": float(mean_sgn_zero),
                                    "mean_abs_ig_nan": float(mean_abs_nan),
                                    "mean_ig_nan": float(mean_sgn_nan),
                                    "n_samples": int(N_view),
                                    "n_present_samples": int(n_present_samples),
                                    "stratify_by": stratify_by,
                                    "stratum": stratum,
                                    "steps": steps,
                                    "baseline": baseline_token,
                                    "output_type": output_type,
                                    "sampling": "case_control",
                                    "seed": seed_th,
                                }
                            )
                        else:
                            rows_agg.append(
                                {
                                    "view": view_name,
                                    "attribution_type": "meds_token",
                                    "attribution_space": "token",
                                    "target_idx": t,
                                    "target_idx_full": t_full,
                                    "horizon_years": h,
                                    "feature": None,
                                    "feature_idx": None,
                                    "code_id": int(cid),
                                    "code_name": med_id_to_name.get(int(cid)) if med_id_to_name is not None else None,
                                    "mean_abs_ig": _select_aggregation_value(
                                        float(mean_abs_zero), float(mean_abs_nan), missing_aggregation
                                    ),
                                    "mean_ig": _select_aggregation_value(
                                        float(mean_sgn_zero), float(mean_sgn_nan), missing_aggregation
                                    ),
                                    "mean_abs_ig_zero": float(mean_abs_zero),
                                    "mean_ig_zero": float(mean_sgn_zero),
                                    "mean_abs_ig_nan": float(mean_abs_nan),
                                    "mean_ig_nan": float(mean_sgn_nan),
                                    "n_samples": int(N_view),
                                    "n_present_samples": int(n_present_samples),
                                    "stratify_by": stratify_by,
                                    "stratum": stratum,
                                    "steps": steps,
                                    "baseline": baseline_token,
                                    "output_type": output_type,
                                    "sampling": "case_control",
                                    "seed": seed_th,
                                }
                            )

                if ig_attr_x_token:
                    if x_feature_ids is None:
                        raise ValueError("ig_attr_x token mode requested but x_feature_ids is None.")
                    for view_name, view_mask in views:
                        for stratify_by, stratum, grouped_view_name, grouped_mask in _iter_group_views(
                            view_name=view_name,
                            view_mask=view_mask,
                            sampled_strata=sampled_strata,
                        ):
                            emit_view_streaming(
                                view_name=grouped_view_name,
                                mask=grouped_mask,
                                ids_tensor=x_feature_ids,
                                attr_type="x_feature",
                                topk_limit=topk_feat,
                                stratify_by=stratify_by,
                                stratum=stratum,
                            )

                if ig_attr_icd:
                    if icd_code_ids is None:
                        raise ValueError("ig_attr_icd=True but icd_code_ids is None (should not happen).")
                    for view_name, view_mask in views:
                        for stratify_by, stratum, grouped_view_name, grouped_mask in _iter_group_views(
                            view_name=view_name,
                            view_mask=view_mask,
                            sampled_strata=sampled_strata,
                        ):
                            emit_view_streaming(
                                view_name=grouped_view_name,
                                mask=grouped_mask,
                                ids_tensor=icd_code_ids,
                                attr_type="icd_token",
                                topk_limit=topk_codes,
                                stratify_by=stratify_by,
                                stratum=stratum,
                            )

                if ig_attr_meds:
                    if meds_code_ids is None:
                        raise ValueError("ig_attr_meds=True but meds_code_ids is None (should not happen).")
                    for view_name, view_mask in views:
                        for stratify_by, stratum, grouped_view_name, grouped_mask in _iter_group_views(
                            view_name=view_name,
                            view_mask=view_mask,
                            sampled_strata=sampled_strata,
                        ):
                            emit_view_streaming(
                                view_name=grouped_view_name,
                                mask=grouped_mask,
                                ids_tensor=meds_code_ids,
                                attr_type="meds_token",
                                topk_limit=topk_codes,
                                stratify_by=stratify_by,
                                stratum=stratum,
                            )

                # free big tensors per (h,t)
                del tokens, padding_mask, icd_code_ids, meds_code_ids, x_feature_ids
                torch.cuda.empty_cache()

            # free sampled inputs per (h,t)
            del xs, icds, meds
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows_agg)

    fmt = str(getattr(args, "ig_save_format", "parquet")).lower()
    out_path = os.path.join(out_dir, f"ig_repquery_{getattr(args, 'ig_split', 'val')}.{fmt}")
    if fmt == "parquet":
        df.to_parquet(out_path, index=False)
    elif fmt == "csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError(f"Unsupported ig_save_format={fmt}")

    return out_path
