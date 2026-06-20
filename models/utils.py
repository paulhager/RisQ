import os
import json
import anndata as ad
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from importlib import import_module
from torchmetrics.functional import (
    mean_absolute_error,
    auroc,
)
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
)
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
from torch import nn
from typing import List, Dict, Optional, Sequence, Tuple, Union, Iterable, Callable

from config_types import Config, Task
import matplotlib.pyplot as plt
import seaborn as sns

from dataset.utils import get_icd_embedding_names, parse_date_safe_vectorized

sns.set_theme(style="whitegrid")
ASSESSMENT_COL = "53-0.0"

from sklearn.metrics import roc_auc_score, roc_curve


def model_selector(args: Config):
    try:
        model_name = args.model_name
        # Dynamically import the module based on model_name
        module = import_module(f"models.{model_name}")
        # Get the class from the module
        model_class = getattr(module, model_name)
        return model_class
    except (ModuleNotFoundError, AttributeError):
        raise


class MetricTracker:
    def __init__(self, task: Task):
        self.best_scores = {}
        if task == Task.REGRESSION:
            self.comparator_dict = {"mae": min}
        elif task == Task.CLASSIFICATION:
            self.comparator_dict = {"auc": max}
        elif task == Task.SURVIVAL:
            self.comparator_dict = {
                "c_index_harrell": max,
                "c_index_ipcw": max,
                "c_index_antolini": max,
            }
        elif task == Task.PROGNOSIS:
            self.comparator_dict = {"loss": min}

    def update(self, metric: str, score: float):
        if metric not in self.best_scores:
            self.best_scores[metric] = score
        else:
            assert (
                metric in self.comparator_dict
            ), f"metric {metric} not in comparator_dict"
            self.best_scores[metric] = self.comparator_dict[metric](
                self.best_scores[metric], score
            )

    def get_best_score(self, metric: str) -> float:
        return self.best_scores.get(metric, None)


def evaluate_metrics(
    args: Config,
    preds: torch.Tensor,
    gt: torch.Tensor,
    split: str,
    metric_tracker: Optional[MetricTracker],
    time_grid: Optional[np.ndarray] = None,
    survival_train: Optional[np.ndarray] = None,
    survival_val: Optional[np.ndarray] = None,
) -> dict:
    metrics = {}
    task = args.task
    checkpoint_dir_path = args.checkpoint_dir_path
    target_outcomes = args.target_outcomes
    times = None
    events = None


    if task == Task.REGRESSION:
        metric_names = ["mae"]
        ms = ["min"]
        metric_function = mean_absolute_error

        score = metric_function(preds, gt)
        metrics[f"{split}.mae"] = score
    elif task == Task.CLASSIFICATION:
        metric_names = ["auc"]
        ms = ["max"]
        metric_function = auroc

        if len(preds.shape) == 1 or preds.shape[1] == 1:
            score = metric_function(preds, gt.to(torch.int), task="binary")
        else:
            # Calculate overall multilabel score
            score = metric_function(
                preds,
                gt.long(),
                task="multilabel",
                num_labels=preds.shape[1],
                ignore_index=-1,
            )

            # Calculate individual target metrics if target_outcomes is provided
            if target_outcomes is not None:
                individual_scores = metric_function(
                    preds,
                    gt.long(),
                    task="multilabel",
                    num_labels=preds.shape[1],
                    ignore_index=-1,
                    average=None,  # This returns per-label scores
                )

                # Log individual target metrics
                assert individual_scores is not None, "individual_scores is None"
                for i, target_name in enumerate(target_outcomes):
                    if i < len(individual_scores):
                        metrics[f"{split}.auc.{target_name}"] = individual_scores[i]

        metrics[f"{split}.auc"] = score
    elif task == Task.SURVIVAL:
        # preds: Dictionary with 'risk_scores' and 'survival_probs' keys (structured predictions)
        # gt: tensor with columns [event, time], for XGBoost: positive time means event, negative time means censoring

        metric_names = ["c_index_harrell", "c_index_ipcw", "c_index_antolini"]
        ms = ["max", "max", "max"]

        events = gt[:, 0].cpu().numpy().astype(int).astype(bool)
        times = gt[:, 1].cpu().numpy().astype(float)

        # ---- Harrell & IPCW (unchanged) ----
        risk_scores = (
            preds["risk_scores"].detach().cpu().numpy().astype(float).squeeze()
        )
        c_index_harrell = concordance_index_censored(events, times, risk_scores)[0]
        c_index_ipcw = concordance_index_ipcw(
            survival_train, survival_val, risk_scores
        )[0]
        score_harrell = torch.tensor(c_index_harrell, dtype=torch.float32)
        score_ipcw = torch.tensor(c_index_ipcw, dtype=torch.float32)
        metrics[f"{split}.c_index_harrell"] = score_harrell
        metrics[f"{split}.c_index_ipcw"] = score_ipcw

        surv_probs_t = preds["survival_probs"]                  # torch [N, H]

        # ---- Antolini C-index (discrete-time fixed implementation) ----
        # train skipped for cost.
        if split == "train":
            metrics[f"{split}.c_index_antolini"] = float("nan")
        elif time_grid is None or len(time_grid) == 0:
            metrics[f"{split}.c_index_antolini"] = float("nan")
        else:
            eps = 1e-7
            p_cum_fixed = 1.0 - surv_probs_t
            p_cum_fixed = p_cum_fixed.clamp(eps, 1.0 - eps)
            logits_fixed = torch.log(p_cum_fixed / (1.0 - p_cum_fixed)).unsqueeze(1)
            obs_t = gt[:, 1].to(dtype=torch.float32, device=logits_fixed.device).unsqueeze(1)
            ev_t = gt[:, 0].to(device=logits_fixed.device).bool().unsqueeze(1)
            horizons_fixed = torch.as_tensor(
                time_grid, dtype=logits_fixed.dtype, device=logits_fixed.device
            )
            _, macro_ant_fixed, _ = antolini_cindex_fixed(
                logits=logits_fixed,
                observed_times=obs_t,
                events=ev_t,
                horizons_years=horizons_fixed,
                horizon_max=float(time_grid[-1]),
            )
            metrics[f"{split}.c_index_antolini"] = float(macro_ant_fixed)

    else:
        raise ValueError(f"Invalid task: {task}")
    if split == "val" and metric_tracker is not None:
        for metric, m in zip(metric_names, ms):
            current_best_score = metric_tracker.get_best_score(metric)
            metric_tracker.update(metric, metrics[f"{split}.{metric}"])
            # Log gt and preds for offline evaluation if new best
            if current_best_score != metric_tracker.get_best_score(metric):
                target_file = os.path.join(checkpoint_dir_path, f"preds_{split}.csv")
                if not os.path.exists(checkpoint_dir_path):
                    os.makedirs(checkpoint_dir_path, exist_ok=True)

                # Handle survival vs others
                if task == Task.SURVIVAL:
                    if split == "val":
                        continue
                    # For structured predictions, save risk scores (most relevant for ranking)
                    pred_values = preds["risk_scores"].cpu().numpy().flatten()
                    pd.DataFrame(
                        {
                            "times": times,
                            "events": events,
                            "risk_scores": pred_values,
                        }
                    ).to_csv(target_file, index=False)
                else:
                    # For simple tensor predictions
                    pd.DataFrame(
                        {
                            "y_gt": gt.cpu().numpy().flatten(),
                            "y_pred": preds.cpu().numpy().flatten(),
                        }
                    ).to_csv(target_file, index=False)
            metrics[f"supervised.{split}.{metric}.{m}"] = metric_tracker.get_best_score(
                metric
            )

    return metrics


def compute_global_decision_boundary(
    logits: np.ndarray,  # shape [B, D, H] cumulative *logits* per horizon
    diagnosis_years: np.ndarray,  # shape [B, D], years; NaN=no event by cutoff; <0 = pre-baseline (exclude)
) -> Tuple[float, float]:
    """
    Learn a global threshold on cumulative logits to classify:
      y = 1  if event occurred at time <= horizon_max
      y = 0  if no event by horizon_max (either censored or event after horizon_max)
    Excludes subjects with pre-baseline diagnoses (time < 0) for that disease.

    Returns:
      threshold (float) in logit space (and auc if return_auc=True).
    """
    B, D, H = logits.shape
    # 1) Flatten subject–disease pairs
    X = logits.reshape(-1, H)  # [B*D, H]
    t = diagnosis_years.reshape(-1)  # [B*D]
    horizon_max = H

    # 2) Exclude pre-baseline diagnoses
    eligible = (t >= 0) | np.isnan(t)

    # 3) Labels: positives = event within horizon; negatives = censored or event after horizon
    pos = (~np.isnan(t)) & (t <= horizon_max)
    neg = np.isnan(t) | (t > horizon_max)
    valid = eligible & (pos | neg)

    if valid.sum() == 0:
        raise ValueError("No valid pairs after filtering (check inputs).")

    # 4) Use last cumulative horizon score; enforce monotonicity to be safe
    X_mono_last = np.maximum.accumulate(X, axis=1)[:, -1]  # [B*D]
    scores = X_mono_last[valid]
    labels = pos[valid].astype(int)

    # 5) ROC + Youden J to pick a single threshold
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_threshold = float(thresholds[best_idx])

    auc = float(roc_auc_score(labels, scores))
    return best_threshold, auc


def prepare_diagnosis_frames(
    adata_val: ad.AnnData,
    disease_columns: Sequence[str],
    assessment_column: str = ASSESSMENT_COL,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return assessment dates, raw diagnosis dates, and diagnosis years post-assessment."""
    obs_df = adata_val.obs.copy()

    assessment_raw = obs_df[assessment_column].astype("object")
    assessment_dates = pd.to_datetime(assessment_raw, errors="coerce")

    def _to_datetime(col: pd.Series) -> pd.Series:
        return pd.to_datetime(col.astype("object"), errors="coerce")

    disease_dates = obs_df[disease_columns].apply(_to_datetime)

    diagnosis_years = (
        disease_dates.subtract(assessment_dates, axis=0)
        .div(np.timedelta64(1, "D"))
        .div(365.25)
    )
    return assessment_dates, disease_dates, diagnosis_years


def evaluate_monotonicity(logits: np.ndarray, atol: float = 1e-6) -> Dict[str, float]:
    """Compute monotonicity statistics over subject–disease trajectories."""
    # logits has shape (N, n_diseases, H)
    diffs = np.diff(logits, axis=-1) # shape (N, D, H-1)
    non_decreasing = np.all(diffs >= -atol, axis=-1)
    non_increasing = np.all(diffs <= atol, axis=-1)
    strictly_monotonic = np.all(diffs > atol, axis=-1) | np.all(diffs < -atol, axis=-1) # shape (N, D) bool

    sign_flips = np.count_nonzero(
        (logits[..., 1:] > 0) != (logits[..., :-1] > 0), axis=-1
    )

    flat_total = logits.shape[0] * logits.shape[1]
    return {
        "non_decreasing_rate": non_decreasing.sum() / flat_total,
        "non_increasing_rate": non_increasing.sum() / flat_total,
        "strictly_monotonic_rate": strictly_monotonic.sum() / flat_total,
        "mean_sign_changes": sign_flips.mean(),
        "median_sign_changes": float(np.median(sign_flips)),
    }

@torch.no_grad()
def antolini_cindex_fixed(
    logits: torch.Tensor,         # [B, D, H]
    observed_times: torch.Tensor, # [B, D]
    events: torch.Tensor,         # [B, D] bool/int
    horizons_years: torch.Tensor, # [H] strictly increasing
    horizon_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Discrete-time Antolini concordance index.

    Uses horizon-bucket evaluation:
        - event time t_i is mapped to the smallest horizon H_k >= t_i
        - comparable set uses:
              t_j > H_k

    Parameters
    ----------
    logits:
        [B, D, H]
        Cumulative-risk logits over horizons.

    observed_times:
        [B, D]
        True observed follow-up times:
            - event time if event observed
            - censoring time otherwise

    events:
        [B, D]
        Event indicators:
            1 = event observed
            0 = right censored

    horizons_years:
        [H]
        Strictly increasing horizon grid.

    horizon_max:
        Administrative truncation horizon.

    Returns
    -------
    per_disease:
        [D] c-index per disease

    macro:
        Mean over diseases

    micro:
        Pooled concordance
    """

    assert logits.ndim == 3
    assert observed_times.shape == logits.shape[:2]
    assert events.shape == logits.shape[:2]
    assert horizons_years.numel() == logits.shape[2]

    device = logits.device
    dtype = logits.dtype

    B, D, H = logits.shape

    Hvec = horizons_years.to(device=device, dtype=dtype)

    # --------------------------------------------------------------
    # Enforce monotone cumulative risk
    # --------------------------------------------------------------

    logits_mono = torch.cummax(logits, dim=2).values

    # --------------------------------------------------------------
    # Inputs
    # --------------------------------------------------------------

    t = observed_times.to(device=device, dtype=dtype)
    e = events.to(device=device).bool()

    # remove invalid rows
    valid_global = t > 0

    # administrative truncation
    t = torch.clamp(t, max=horizon_max)

    # events after horizon_max become censored
    e = e & (observed_times.to(device=device) <= horizon_max)

    # --------------------------------------------------------------
    # Map times to discrete horizons
    #
    # k_i = smallest k such that H_k >= t_i
    # --------------------------------------------------------------

    k_idx = torch.bucketize(t, Hvec, right=False)

    # clamp right edge
    k_idx = torch.clamp(k_idx, max=H - 1)

    # --------------------------------------------------------------
    # Outputs
    # --------------------------------------------------------------

    per_disease = torch.full(
        (D,),
        float("nan"),
        device=device,
        dtype=torch.float64,
    )

    total_conc = torch.tensor(
        0.0,
        device=device,
        dtype=torch.float64,
    )

    total_pairs = torch.tensor(
        0.0,
        device=device,
        dtype=torch.float64,
    )

    # --------------------------------------------------------------
    # Disease loop
    # --------------------------------------------------------------

    for d in range(D):

        valid = valid_global[:, d]

        if not torch.any(valid):
            continue

        td = t[valid, d]               # [N]
        ed = e[valid, d]               # [N]
        kd = k_idx[valid, d]           # [N]
        ld = logits_mono[valid, d, :]  # [N, H]

        N = td.shape[0]

        if N <= 1 or not torch.any(ed):
            continue

        conc_sum_d = torch.tensor(
            0.0,
            device=device,
            dtype=torch.float64,
        )

        pairs_sum_d = torch.tensor(
            0.0,
            device=device,
            dtype=torch.float64,
        )

        # ----------------------------------------------------------
        # Process events grouped by horizon bucket
        # ----------------------------------------------------------

        ks = torch.unique(kd[ed], sorted=True)

        for k in ks.tolist():

            Hk = Hvec[k]

            # ------------------------------------------------------
            # Event subjects whose event maps to bucket k
            # ------------------------------------------------------

            I_mask = ed & (kd == k)

            if not torch.any(I_mask):
                continue

            Fi = ld[I_mask, k]  # [Ni]

            # ------------------------------------------------------
            # Comparable set:
            #
            # subjects observed strictly after H_k
            # ------------------------------------------------------

            S_mask = td > Hk

            if not torch.any(S_mask):
                continue

            Fj = ld[S_mask, k]  # [Nj]

            # ------------------------------------------------------
            # Fast concordance counting via sorting
            # ------------------------------------------------------

            Fj_sorted, _ = torch.sort(Fj)

            left = torch.searchsorted(
                Fj_sorted,
                Fi,
                right=False,
            )

            right = torch.searchsorted(
                Fj_sorted,
                Fi,
                right=True,
            )

            less = left.to(torch.float64)

            ties = (right - left).to(torch.float64)

            conc_sum_d += (less + 0.5 * ties).sum()

            pairs_sum_d += (
                Fi.shape[0] * Fj.shape[0]
            )

        # ----------------------------------------------------------
        # Finalize disease
        # ----------------------------------------------------------

        if pairs_sum_d > 0:

            cind = conc_sum_d / pairs_sum_d

            per_disease[d] = cind

            total_conc += conc_sum_d
            total_pairs += pairs_sum_d

    # --------------------------------------------------------------
    # Macro / micro
    # --------------------------------------------------------------

    macro = (
        torch.nanmean(per_disease)
        if torch.any(~torch.isnan(per_disease))
        else torch.tensor(float("nan"), device=device)
    )

    micro = (
        total_conc / total_pairs
        if total_pairs > 0
        else torch.tensor(float("nan"), device=device)
    )

    return per_disease, macro, micro


def antolini_times_events_from_diagnosis_years(
    diagnosis_years: np.ndarray,
    subject_censoring_years: np.ndarray,
    *,
    horizon_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build observed times and event indicators for ``antolini_cindex_fixed``.

    Matches the survival definition in ``evaluate_logits`` (event if
    0 < diagnosis time <= horizon_max; otherwise censor at subject-level time).
    """
    times = np.asarray(diagnosis_years, dtype=np.float32)
    events = (~np.isnan(times)) & (times > 0) & (times <= horizon_max)
    censor = np.asarray(subject_censoring_years, dtype=np.float32).reshape(-1, 1)
    observed_times = np.where(events, times, censor)
    return observed_times, events.astype(bool, copy=False)


def uno_c_index_ipcw_per_disease(
    logits: np.ndarray,  # [B, D, H] cumulative logits
    times_years: pd.DataFrame,  # [B, D] years; NaN=censored; <0 => exclude
    horizons_years: np.ndarray,  # [H]
    horizon_max: float,  # e.g. 16.0
    disease_order: list[str],  # length D
):
    """
    Compute Uno-style IPCW C-index per disease separately.

    Returns:
        {
            disease_code: {
                "c_index": ...,
                "concordant": ...,
                "discordant": ...,
                "tied_risk": ...,
                "tied_time": ...
            },
            ...
        }
    """
    assert logits.ndim == 3
    B, D, H = logits.shape
    t = times_years.to_numpy()  # [B, D]

    # --- Enforce monotonicity once ---
    logits_mono = np.maximum.accumulate(logits, axis=2)

    out = {}

    for d in range(D):
        td = t[:, d]  # [B]
        Xd = logits_mono[:, d, :]  # [B, H]

        # --- Exclude pre-baseline completely ---
        eligible = (td >= 0) | np.isnan(td)

        if not np.any(eligible):
            out[disease_order[d]] = {
                "c_index": float("nan"),
                "concordant": 0,
                "discordant": 0,
                "tied_risk": 0,
                "tied_time": 0,
            }
            continue

        td_e = td[eligible]  # eligible times
        Xd_e = Xd[eligible, :]
        risk = Xd_e[:, -1]  # final-horizon cumulative logit

        # --- Event / censoring definition ---
        # event if 0 <= time <= horizon_max
        is_event = (~np.isnan(td_e)) & (td_e <= horizon_max)

        # observed time = min(time, horizon_max); NaN → horizon_max (censor)
        time_obs = np.where(np.isnan(td_e), horizon_max, td_e)
        time_obs = np.clip(time_obs, 0, horizon_max)

        # --- Structured array for sksurv ---
        dtype = np.dtype([("event", bool), ("time", float)])
        surv = np.empty(time_obs.shape[0], dtype=dtype)
        surv["event"] = is_event.astype(bool)
        surv["time"] = time_obs.astype(float)

        # same survival set for train and test
        survival_train = surv
        survival_test = surv

        # --- Compute IPCW Uno C-index ---
        try:
            cindex, C, Dd, TR, TT = concordance_index_ipcw(
                survival_train,
                survival_test,
                risk,
                tau=horizon_max,
            )
            out[disease_order[d]] = {
                "c_index": float(cindex),
                "concordant": int(C),
                "discordant": int(Dd),
                "tied_risk": int(TR),
                "tied_time": int(TT),
            }
        except Exception:
            out[disease_order[d]] = {
                "c_index": float("nan"),
                "concordant": 0,
                "discordant": 0,
                "tied_risk": 0,
                "tied_time": 0,
            }

    return out


def summarize_positive_negative_performance_all_horizons(
    logits: np.ndarray,  # [B, D, H] cumulative logits per horizon
    diagnosis_years: pd.DataFrame,  # [B, D] years; NaN=censored; <0 => exclude
    horizons_years: np.ndarray,  # [H] strictly increasing (e.g., [1., 2., ..., 16.])
    threshold: float,  # global decision boundary in *logit* space
    include_at_event_bin: bool = True,  # True = count detection at the first bin ≥ event time
) -> pd.DataFrame:
    assert logits.ndim == 3
    B, D, H = logits.shape
    assert diagnosis_years.values.shape == (B, D)
    assert horizons_years.shape[0] == H

    # 1) Flatten subject–disease pairs and enforce monotonicity across horizons
    X = logits.reshape(-1, H)  # [N, H], N = B*D
    X_mono = np.maximum.accumulate(X, axis=1)  # ensure cumulative
    t = diagnosis_years.to_numpy().reshape(-1)  # [N]

    # 2) Eligibility: exclude pre-baseline entirely
    eligible = (t >= 0) | np.isnan(t)

    # Precompute per-positive quantities used for detection/lead time
    # We will re-subset by horizon k each time.
    # Map each (positive) event time to its first horizon index ≥ t
    k_event_all = np.searchsorted(
        horizons_years, np.where(np.isnan(t), np.inf, t), side="left"
    )
    k_event_all = np.clip(k_event_all, 0, H - 1)

    rows = []
    for k in range(H):
        Hk = horizons_years[k]

        # 3) Labels at horizon k
        pos_k = (~np.isnan(t)) & (t <= Hk)
        neg_k = np.isnan(t) | (t > Hk)
        valid_k = eligible & (pos_k | neg_k)

        # If nothing valid at this horizon, return NaNs
        if not np.any(valid_k):
            rows.append(
                {
                    "horizon_years": float(Hk),
                    "num_negatives": 0,
                    "true_negative_rate": np.nan,
                    "false_positive_rate": np.nan,
                    "num_positives_total": 0,
                    "num_positives_within_horizon": 0,
                    "detection_rate_after_diag": np.nan,
                    "mean_lead_time_years": np.nan,
                    "median_lead_time_years": np.nan,
                    "auc": np.nan,
                }
            )
            continue

        # 4) Scores at horizon k (after enforcing monotonicity)
        s_k = X_mono[:, k]

        # --- Negatives (for TNR/FPR at threshold) ---
        neg_idx = valid_k & neg_k
        num_negatives = int(neg_idx.sum())
        if num_negatives > 0:
            neg_scores = s_k[neg_idx]
            true_negatives = int((neg_scores <= threshold).sum())
            tnr = true_negatives / num_negatives
            fpr = 1.0 - tnr
        else:
            tnr = np.nan
            fpr = np.nan

        # --- Positives within horizon k (for detection & lead time) ---
        pos_idx = valid_k & pos_k
        num_pos_total = int(pos_idx.sum())  # equals within-horizon at k by definition
        num_pos_within = num_pos_total

        if num_pos_within > 0:
            t_pos = t[pos_idx]  # [N_pos]
            # Map each event time to first horizon index ≥ t
            k_pos = k_event_all[pos_idx]
            # Detection window: at or after the event horizon
            start_bin = k_pos if include_at_event_bin else (k_pos + 1)
            start_bin = np.clip(start_bin, 0, H - 1)

            # Build mask [N_pos, H] of columns ≥ start_bin per row
            bins = np.arange(H)[None, :]
            after_diag = bins >= start_bin[:, None]  # [N_pos, H]

            # Use the *full* horizon grid for detection (not just up to k),
            # because we’re asking “did the model cross threshold at/after diagnosis?”
            X_pos = X_mono[pos_idx, :]  # [N_pos, H]
            crossed = (X_pos > threshold) & after_diag
            detected = crossed.any(axis=1)
            detection_rate = float(detected.mean()) if detected.size else np.nan

            # Lead time: first crossing horizon time minus event time (years)
            # first index where crossed==True; NaN if none
            first_idx = np.where(detected, crossed.argmax(axis=1), np.nan).astype(float)
            valid_hits = np.isfinite(first_idx)
            if np.any(valid_hits):
                hit_years = horizons_years[first_idx[valid_hits].astype(int)]
                lead = hit_years - t_pos[valid_hits]
                # Guard tiny negatives from floating point
                lead = lead[np.isfinite(lead) & (lead >= -1e-9)]
                mean_lead = float(np.mean(lead)) if lead.size else np.nan
                median_lead = float(np.median(lead)) if lead.size else np.nan
            else:
                mean_lead = np.nan
                median_lead = np.nan
        else:
            detection_rate = np.nan
            mean_lead = np.nan
            median_lead = np.nan

        # --- AUC at horizon k (simple label-based; no IPCW here) ---
        auc_k = np.nan
        y_k = pos_k[valid_k].astype(int)
        s_valid = s_k[valid_k]
        if y_k.size and np.unique(y_k).size > 1:
            try:
                auc_k = float(roc_auc_score(y_k, s_valid))
            except Exception:
                auc_k = np.nan

        rows.append(
            {
                "horizon_years": float(Hk),
                "num_negatives": num_negatives,
                "true_negative_rate": tnr,
                "false_positive_rate": fpr,
                "num_positives_total": num_pos_total,
                "num_positives_within_horizon": num_pos_within,
                "detection_rate_after_diag": detection_rate,
                "mean_lead_time_years": mean_lead,
                "median_lead_time_years": median_lead,
                "auc": auc_k,
            }
        )

    df = pd.DataFrame(rows).set_index("horizon_years")
    return df


def compute_per_disease_auc_all_horizons(
    logits: np.ndarray,  # [B, D, H] cumulative logits
    diagnosis_years: pd.DataFrame,  # [B, D] years; NaN=censored; <0 => exclude
    horizons_years: np.ndarray,  # [H] strictly increasing
    disease_order: list[str],  # length D; codes per disease index
) -> dict:
    """
    Returns a dict: {disease_code: { "<h>y": auc, ... }, ... }
    AUC at horizon h uses labels: positive if time <= h, negative if time > h or NaN.
    Pre-baseline (time < 0) are excluded for that disease.
    """
    assert logits.ndim == 3
    B, D, H = logits.shape
    assert diagnosis_years.values.shape == (B, D)
    assert horizons_years.shape[0] == H
    assert len(disease_order) == D

    # Enforce monotonicity across horizons (logits-space)
    logits_mono = np.maximum.accumulate(logits, axis=2)  # [B, D, H]
    t = diagnosis_years.to_numpy()  # [B, D]
    eligible = (t >= 0) | np.isnan(t)  # drop pre-baseline

    out = {}
    for d in range(D):
        td = t[:, d]
        elig_d = eligible[:, d]
        Xd = logits_mono[:, d, :]  # [B, H]

        per_h = {}
        for k, hk in enumerate(horizons_years):
            pos = (~np.isnan(td)) & (td <= hk)
            neg = np.isnan(td) | (td > hk)
            valid = elig_d & (pos | neg)

            if not np.any(valid):
                per_h[f"{float(hk):.1f}y"] = float("nan")
                continue

            y = pos[valid].astype(int)
            s = Xd[valid, k]
            if y.sum() == 0 or y.sum() == y.size:
                per_h[f"{float(hk):.1f}y"] = float(
                    "nan"
                )  # AUC undefined with a single class
            else:
                try:
                    per_h[f"{float(hk):.1f}y"] = float(roc_auc_score(y, s))
                except Exception:
                    per_h[f"{float(hk):.1f}y"] = float("nan")

        out[disease_order[d]] = per_h

    return out

def _disease_order_from_codes_to_row(codes_to_row: dict[str, int]) -> list[str]:
    return [k for k, _ in sorted(codes_to_row.items(), key=lambda x: int(x[1]))]


def evaluate_logits(
    all_logits: np.ndarray, # shape (N, n_diseases, n_horizons)
    split: str,
    adata: ad.AnnData,
    args: Config,
    wandb_logger: WandbLogger,
):
    """Evaluate logits for all prognosis targets in codes_to_row (row order)."""
    _, mapping_name = get_icd_embedding_names(args, 3)
    codes_to_row_path = os.path.join(args.data_root_path, mapping_name)  # type: ignore[arg-type]
    with open(codes_to_row_path, "r") as f:
        codes_to_row = json.load(f)
    disease_order = _disease_order_from_codes_to_row(codes_to_row)
    n_diseases = len(disease_order)

    if all_logits.shape[1] != n_diseases:
        raise ValueError(
            f"Logits target dim {all_logits.shape[1]} != codes_to_row rows {n_diseases} "
            f"({mapping_name})"
        )
    logits = all_logits
    print(f"Evaluating {n_diseases} prognosis targets ({mapping_name})")

    _, _, diagnosis_years = prepare_diagnosis_frames(adata, disease_order) # (N, D) DataFrame with years from baseline; NaN = no event; <0 = pre-baseline
    if diagnosis_years.shape[1] != n_diseases:
        raise ValueError(
            f"diagnosis_years columns {diagnosis_years.shape[1]} != {n_diseases} targets"
        )
    H = logits.shape[-1]
    horizons_in_years = np.linspace(1, H, H).astype(float)  # [1, 2, ..., H]
    horizon_max = float(horizons_in_years[-1])

    persist_split_artifacts = not (split == "val" and args.task == Task.SURVIVAL)
    if persist_split_artifacts:
        # -----------------------------
        # Persist labels + disease order (full panel and legacy 588-prefix exports)
        # -----------------------------
        diagnosis_years_np = diagnosis_years.to_numpy(dtype=np.float32)

        out_dir = args.checkpoint_dir_path
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, f"diagnosis_years_{split}.npy"), diagnosis_years_np)
        with open(os.path.join(out_dir, f"disease_order_{split}.json"), "w") as f:
            json.dump(disease_order, f)

        n_train_panel = min(588, n_diseases)
        np.save(
            os.path.join(out_dir, f"diagnosis_years_588_{split}.npy"),
            diagnosis_years_np[:, :n_train_panel],
        )
        with open(os.path.join(out_dir, f"disease_order_588_{split}.json"), "w") as f:
            json.dump(disease_order[:n_train_panel], f)

        np.save(
            os.path.join(out_dir, f"global_row_{split}.npy"),
            adata.obs["global_row"].to_numpy(dtype=np.int64),
        )

        # --- Per-disease event counts on val split (events within horizon_max) ---
        # diagnosis_years: DataFrame [B, D] with years from baseline; NaN = no event
        t = diagnosis_years.to_numpy()  # [B, D]

        # Exclude pre-baseline diagnoses (<= 0 years)
        eligible = (t > 0) | np.isnan(t)

        # Event if time <= horizon_max
        is_event = (~np.isnan(t)) & (t <= horizon_max)

        # Only count events for eligible subjects (0 < time <= horizon_max)
        valid_events = eligible & is_event

        # Sum over subjects → number of events per disease
        event_counts = valid_events.sum(axis=0).astype(int)  # [D]

        disease_counts_dict = {
            disease_order[i]: int(event_counts[i])
            for i in range(len(disease_order))
        }

        counts_path = os.path.join(
            args.checkpoint_dir_path, f"per_disease_event_counts_{split}.json"
        )
        with open(counts_path, "w") as f:
            json.dump(disease_counts_dict, f)

    decision_boundary, _auc = compute_global_decision_boundary(
        logits=logits,
        diagnosis_years=diagnosis_years.to_numpy(),
    )

    # Calculate stats for computed optimum decision_boundary
    summary_df_all_horizons = summarize_positive_negative_performance_all_horizons(
        logits=logits,
        diagnosis_years=diagnosis_years,
        horizons_years=horizons_in_years,
        threshold=decision_boundary,
    )

    summary_monotonicity = evaluate_monotonicity(logits)

    # New c-index implementation with fixed censoring
    # Start by filling times with diagnosis times (contains negative times for pre-baseline events and NaN for no events)
    times = diagnosis_years.to_numpy(dtype=np.float32)
    # Set events to 1 if time > 0 and time <= horizon_max, 0 otherwise
    events = (~np.isnan(times)) & (times > 0) & (times <= horizon_max)
    # Fill in times with censoring times for negatives (either death, coma or administrative censoring)
    assessment_date_field = args.assessment_date_field
    assessment_dates = parse_date_safe_vectorized(adata.obs[assessment_date_field]).astype("datetime64[ns]").values
    death_dates = parse_date_safe_vectorized(adata.obs["Date_Death"]).astype("datetime64[ns]").values
    coma_dates = parse_date_safe_vectorized(adata.obs["Date_Coma"]).astype("datetime64[ns]").values
    global_censoring_date = np.datetime64(getattr(args, "time_cutoff", "2022-05-31"))  # Earliest UKB hospital inpatient censoring date (31 May 2022, for Wales cohort)
    # Calculate time differences in years
    death_time_diffs_years = (death_dates - assessment_dates) / pd.Timedelta(days=365.25)
    coma_time_diffs_years = (coma_dates - assessment_dates) / pd.Timedelta(days=365.25)
    global_censoring_time_diffs_years = (global_censoring_date - assessment_dates) / pd.Timedelta(days=365.25)
    # Censoring time is the same for all diagnosis fields (earliest of death/coma per subject)
    censoring_time_diffs_years = np.fmin(death_time_diffs_years, coma_time_diffs_years)
    combined_censoring_time_diffs_years = np.where(np.isnan(censoring_time_diffs_years), global_censoring_time_diffs_years, censoring_time_diffs_years)
    # Broadcast per-subject censoring times across all targets.
    # events/times: (n_subjects, n_targets), combined_censoring_time_diffs_years: (n_subjects,)
    times = np.where(events, times, combined_censoring_time_diffs_years[:, None])

    per_disease_c_index, macro_c_index, micro_c_index = (
        antolini_cindex_fixed(
            torch.from_numpy(logits),
            torch.from_numpy(times),
            torch.from_numpy(events),
            torch.from_numpy(horizons_in_years),
            horizon_max,
        )
    )
    macro_c_index = float(macro_c_index)
    micro_c_index = float(micro_c_index)

    summary_dict = {
        "macro_c_index": macro_c_index,
        "micro_c_index": micro_c_index,
        "decision_boundary": decision_boundary,
        "decision_boundary_auc": float(_auc) if _auc is not None else np.nan,
    }

    # Add monotonicity metrics (prefix keys)
    for key, value in summary_monotonicity.items():
        summary_dict[f"monotonicity_{key}"] = (
            float(value) if value is not None else np.nan
        )

    # ---- Per-disease Harrell c-index (macro + micro) ----
    (
        per_disease_harrell,
        macro_c_index_harrell,
        micro_c_index_harrell,
    ) = harrell_cindex_per_disease_from_logits(
        logits=logits,
        diagnosis_years=diagnosis_years,
        horizons_years=horizons_in_years,
        horizon_max=horizon_max,
        disease_order=disease_order,
    )

    if persist_split_artifacts:
        # Save per-disease Harrell c-index
        path_harrell = os.path.join(
            args.checkpoint_dir_path, f"per_disease_harrell_{split}.json"
        )
        with open(path_harrell, "w") as f:
            json.dump(per_disease_harrell, f)

    # ---- W&B logging ----
    run = wandb_logger.experiment  # Lightning's WandbLogger exposes the underlying Run

    # Log macro/micro Harrell c-index to WandB
    run.log(
        {
            f"{split}/harrell_macro_c_index": macro_c_index_harrell,
            f"{split}/harrell_micro_c_index": micro_c_index_harrell,
        }
    )

    # ---- Persist per-disease c-index JSON ----
    if persist_split_artifacts:
        per_disease_c_index = (
            per_disease_c_index.cpu().numpy()
            if isinstance(per_disease_c_index, torch.Tensor)
            else per_disease_c_index
        )
        per_disease_c_index_dict = {
            disease_order[i]: float(per_disease_c_index[i])
            for i in range(len(per_disease_c_index))
        }
        with open(
            os.path.join(args.checkpoint_dir_path, f"per_disease_c_index_{split}.json"), "w"
        ) as f:
            json.dump(per_disease_c_index_dict, f)

    # ---- Overall mean logits curve ----
    overall_mean_curve = logits.mean(axis=(0, 1))  # [H]
    years = horizons_in_years  # x-axis in years

    # 3a) Nested per-horizon metrics
    nested_metrics = metrics_df_to_nested_wandb(
        summary_df_all_horizons, unit_suffix="y"
    )
    run.summary.update({f"{split}/horizon_metrics": nested_metrics})

    # 3b) Line plots across horizons for key metrics
    log_per_horizon_lineplots(run, summary_df_all_horizons, title_prefix=f"{split}/")

    # 3c) Mean logits curve as line plot (and image for completeness)
    log_mean_logits_curve(
        run, overall_mean_curve, years, title=f"{split}/Overall Mean Logit Trajectory"
    )

    # 3d) Also log the flat summary_dict for convenience
    run.log(
        {
            f"{split}/macro_c_index": macro_c_index,
            f"{split}/micro_c_index": micro_c_index,
            f"{split}/decision_boundary": decision_boundary,
            f"{split}/decision_boundary_auc": (
                float(_auc) if _auc is not None else np.nan
            ),
        }
    )

    per_disease_ipcw = uno_c_index_ipcw_per_disease(
        logits=logits,
        times_years=diagnosis_years,
        horizons_years=horizons_in_years,
        horizon_max=horizon_max,
        disease_order=disease_order,
    )

    if persist_split_artifacts:
        with open(
            os.path.join(args.checkpoint_dir_path, f"per_disease_ipcw_cindex_{split}.json"),
            "w",
        ) as f:
            json.dump(per_disease_ipcw, f)

        # Save per-disease AUC by horizon to JSON
        per_disease_auc_by_h = compute_per_disease_auc_all_horizons(
            logits=logits,
            diagnosis_years=diagnosis_years,
            horizons_years=horizons_in_years,
            disease_order=disease_order,
        )

        path_auc = os.path.join(
            args.checkpoint_dir_path, f"per_disease_auc_by_horizon_{split}.json"
        )
        with open(path_auc, "w") as f:
            json.dump(per_disease_auc_by_h, f)



def metrics_df_to_nested_wandb(df: pd.DataFrame, unit_suffix: str = "y") -> dict:
    """
    Turn a (H x M) DataFrame (index=horizon_years) into a nested dict:
      {"metric": {"1.0y": val, "2.0y": val, ...}, ...}
    """
    out = {}
    horizons = [f"{float(h):.1f}{unit_suffix}" for h in df.index]
    for col in df.columns:
        out[col] = {
            h: float(v) if pd.notnull(v) else np.nan
            for h, v in zip(horizons, df[col].values)
        }
    return out


def log_per_horizon_lineplots(run, df: pd.DataFrame, title_prefix: str = ""):
    """
    Log line plots for key columns in df vs horizon.
    Uses wandb.plot.line on a W&B Table.
    """
    df_plot = df.reset_index().rename(columns={"horizon_years": "horizon"})
    tbl = wandb.Table(dataframe=df_plot)

    plots = {}

    if "auc" in df:
        plots[f"{title_prefix}AUC vs Horizon"] = wandb.plot.line(
            tbl, "horizon", "auc", title=f"{title_prefix}AUC vs Horizon"
        )
    if "detection_rate_after_diag" in df:
        plots[f"{title_prefix}Detection Rate vs Horizon"] = wandb.plot.line(
            tbl,
            "horizon",
            "detection_rate_after_diag",
            title=f"{title_prefix}Detection Rate vs Horizon",
        )
    # True Negative Rate and FPR (if present)
    if "true_negative_rate" in df:
        plots[f"{title_prefix}TNR vs Horizon"] = wandb.plot.line(
            tbl,
            "horizon",
            "true_negative_rate",
            title=f"{title_prefix}True Negative Rate vs Horizon",
        )
    if "false_positive_rate" in df:
        plots[f"{title_prefix}FPR vs Horizon"] = wandb.plot.line(
            tbl,
            "horizon",
            "false_positive_rate",
            title=f"{title_prefix}False Positive Rate vs Horizon",
        )
    if "mean_lead_time_years" in df:
        plots[f"{title_prefix}Mean Lead Time vs Horizon"] = wandb.plot.line(
            tbl,
            "horizon",
            "mean_lead_time_years",
            title=f"{title_prefix}Mean Lead Time vs Horizon",
        )
    if "median_lead_time_years" in df:
        plots[f"{title_prefix}Median Lead Time vs Horizon"] = wandb.plot.line(
            tbl,
            "horizon",
            "median_lead_time_years",
            title=f"{title_prefix}Median Lead Time vs Horizon",
        )

    # Log them in one call
    if plots:
        run.log(plots)


def log_mean_logits_curve(
    run, mean_logits: np.ndarray, horizons_years: np.ndarray, title: str
):
    """
    Log the overall mean logit trajectory as a line plot and as a static image.
    """
    df_curve = pd.DataFrame(
        {
            "horizon": horizons_years.astype(float),
            "mean_logit": mean_logits.astype(float),
        }
    )
    tbl = wandb.Table(dataframe=df_curve)
    run.log(
        {f"{title} (line)": wandb.plot.line(tbl, "horizon", "mean_logit", title=title)}
    )

def harrell_cindex_per_disease_from_logits(
    logits: np.ndarray,              # [B, D, H] cumulative logits
    diagnosis_years: pd.DataFrame,   # [B, D] years; NaN=censored; <0 => exclude
    horizons_years: np.ndarray,      # [H] strictly increasing
    horizon_max: float,              # e.g. horizons_years[-1]
    disease_order: list[str],        # length D; codes per disease index
) -> tuple[
    dict[str, dict[str, float]],     # per_disease metrics
    float, float,                    # macro_c_index, micro_c_index
]:
    """
    For each disease d, compute the Harrell c-index (concordance_index_censored).

    Event/time definition per (subject, disease):
      - time < 0       => excluded completely (pre-baseline)
      - NaN            => censored at horizon_max
      - 0 <= time <= horizon_max => event at 'time'

    Risk scores:
      - last cumulative logit (after enforcing monotonicity)
    """

    assert logits.ndim == 3
    B, D, H = logits.shape
    assert diagnosis_years.values.shape == (B, D)
    assert horizons_years.shape[0] == H
    assert len(disease_order) == D

    # Enforce monotonicity over horizons (logit-space)
    logits_mono = np.maximum.accumulate(logits, axis=2)  # [B, D, H]
    t = diagnosis_years.to_numpy()                       # [B, D]

    per_disease: dict[str, dict[str, float]] = {}

    # For macro aggregates
    c_values = []

    # For micro Harrell: accumulate concordance components
    total_conc = 0.0
    total_disc = 0.0
    total_tied_risk = 0.0
    total_tied_time = 0.0

    for d_idx in range(D):
        disease_code = disease_order[d_idx]
        td = t[:, d_idx]               # [B]
        Xd = logits_mono[:, d_idx, :]  # [B, H]

        # Exclude pre-baseline diagnoses
        eligible = (td >= 0) | np.isnan(td)
        if not np.any(eligible):
            per_disease[disease_code] = {"c_index_harrell": float("nan")}
            continue

        td_e = td[eligible]           # times for eligible subjects
        Xd_e = Xd[eligible, :]        # logits for eligible subjects

        # Event indicator and observed time
        # event if 0 <= time <= horizon_max
        is_event = (~np.isnan(td_e)) & (td_e <= horizon_max)

        # observed time: NaN -> horizon_max (censored), clip within [0, horizon_max]
        time_obs = np.where(np.isnan(td_e), horizon_max, td_e)
        time_obs = np.clip(time_obs, 0.0, horizon_max)

        n_i = time_obs.shape[0]
        if n_i == 0:
            per_disease[disease_code] = {"c_index_harrell": float("nan")}
            continue

        # ---- Harrell c-index for this disease ----
        # Risk = last cumulative logit
        risk = Xd_e[:, -1].astype(float)

        try:
            c_idx, conc, disc, tied_risk, tied_time = concordance_index_censored(
                is_event.astype(bool), time_obs.astype(float), risk
            )
            c_val = float(c_idx)
        except Exception:
            c_val = float("nan")
            conc = disc = tied_risk = tied_time = 0

        per_disease[disease_code] = {"c_index_harrell": c_val}

        if not np.isnan(c_val):
            c_values.append(c_val)

        # accumulate for micro Harrell
        total_conc += float(conc)
        total_disc += float(disc)
        total_tied_risk += float(tied_risk)
        total_tied_time += float(tied_time)

    # ---- Macro aggregate ----
    macro_c_index = float(np.nanmean(c_values)) if len(c_values) > 0 else float("nan")

    # ---- Micro Harrell (pooled over diseases) ----
    denom = total_conc + total_disc + 0.5 * total_tied_risk
    micro_c_index = float(total_conc / denom) if denom > 0 else float("nan")

    return per_disease, macro_c_index, micro_c_index
