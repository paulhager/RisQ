"""RepQuery model"""

from __future__ import annotations

import math
import logging
from pathlib import Path
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, Sequence
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.loggers.wandb import WandbLogger
from config_types import Config, TemporalSamplingStrategy
from dataset.utils import (
    filter_and_pad_vectorized,
    get_icd_embedding_names,
    npAnnData,
    parse_icd_hierarchy_levels,
)
from models.utils import MetricTracker, evaluate_logits
from models.TransformerUtils import init_ft_transformer
from models.Transformers import CrossAttnDecoder


class RepQuery(pl.LightningModule):

    def __init__(self, args: Config, adata: npAnnData) -> None:
        super().__init__()

        self._set_hparams(args)  # type: ignore[arg-type]
        self.args = args
        self.feature_dropout_p = float(
            getattr(self.args, "feature_dropout_p", 0.0)
        )
        self.structured_feature_dropout_p = float(
            getattr(self.args, "structured_feature_dropout_p", 0.0)
        )
        self.modality_dropout_p = float(
            getattr(self.args, "modality_dropout_p", 0.0)
        )

        self._validate_dropout_configuration()

        adata_var_names = list(adata.var.index)
        self._initialize_structured_dropout_mask(adata_var_names)
        self._initialize_modality_dropout_masks(adata_var_names)

        self.use_pos_weight = bool(getattr(self.args, "repquery_use_pos_weight", False))
        self.pos_weight_clip_max = float(getattr(self.args, "repquery_pos_weight_clip_max", 50.0))
        self.pos_weight_eps = float(getattr(self.args, "repquery_pos_weight_eps", 1.0))

        self._bce = None  # will become BCEWithLogitsLoss when pos_weight is set

        adata_var = adata.var

        (
            ft_transformer_encoder,
            self.num_indices,
            self.bin_indices,
            self.cat_indices,
        ) = init_ft_transformer(
            self.args,
            adata_var,
            use_cls=False,
            use_predictor=False,
        )

        backbone_kwargs = ft_transformer_encoder.backbone_kwargs
        self.args.hidden_dim = backbone_kwargs["hidden_dim"]
        self.args.num_heads = backbone_kwargs["num_heads"]
        self.args.attention_dropout = backbone_kwargs["attention_dropout"]
        self.args.ffn_dropout = backbone_kwargs["ffn_dropout"]
        self.args.ffn_hidden_dim = backbone_kwargs["ffn_hidden_dim"]

        self.cross_attn_decoder = CrossAttnDecoder(
            d_model=self.args.decoder_hidden_dim,
            num_heads=self.args.decoder_num_heads,
            num_layers=self.args.decoder_n_layers,
            dim_feedforward=(
                self.args.decoder_ffn_hidden_dim
                if self.args.decoder_ffn_hidden_dim is not None
                else self.args.hidden_dim * self.args.decoder_ffn_hidden_dim_multiplier
            ),
            attn_dropout=self.args.decoder_attention_dropout,
            resid_dropout=self.args.decoder_ffn_dropout,
            activation=self.args.activation,
        )

        self.register_buffer(
            "num_indices_tensor",
            torch.from_numpy(self.num_indices.astype(bool)),
            persistent=False,
        )
        self.register_buffer(
            "bin_indices_tensor",
            torch.from_numpy(self.bin_indices.astype(bool)),
            persistent=False,
        )
        self.register_buffer(
            "cat_indices_tensor",
            torch.from_numpy(self.cat_indices.astype(bool)),
            persistent=False,
        )

        self.num_embeddings = ft_transformer_encoder.num_embeddings
        self.bin_embeddings = ft_transformer_encoder.bin_embeddings
        self.cat_embeddings = ft_transformer_encoder.cat_embeddings

        self.cls_embedding = nn.Parameter(
            torch.zeros(self.args.n_cls_tokens, self.args.hidden_dim)
        )

        self.encoder = ft_transformer_encoder.backbone

        self.projector = nn.Linear(
            self.args.hidden_dim,
            self.args.decoder_hidden_dim,
        )

        self.predictor = nn.Linear(
            self.args.decoder_hidden_dim,
            1,
        )

        self._load_icd_embeddings()
        self._load_icd_prognosis_embeddings()
        self._load_meds_embeddings()

        self.negative_sample_weight = float(
            getattr(self.args, "repquery_negative_sample_weight", 1.0)
        )
        if not 0.0 <= self.negative_sample_weight <= 1.0:
            raise ValueError("repquery_negative_sample_weight must be between 0 and 1.")

        self.loss_trackers = {
            "train": MetricTracker(args.task),
            "val": MetricTracker(args.task),
        }

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_feature_names(path: Union[str, Path]) -> list[str]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Feature list file not found: '{file_path}'")
        return [line.strip() for line in file_path.read_text().splitlines() if line.strip()]

    @staticmethod
    def _build_feature_mask(
        feature_names: set[str], adata_var_names: Sequence[str]
    ) -> np.ndarray:
        return np.array([name in feature_names for name in adata_var_names], dtype=bool)

    @staticmethod
    def _validate_dropout_probability(name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}.")

    def _validate_dropout_configuration(self) -> None:
        self._validate_dropout_probability("feature_dropout_p", self.feature_dropout_p)
        self._validate_dropout_probability(
            "structured_feature_dropout_p", self.structured_feature_dropout_p
        )
        self._validate_dropout_probability(
            "modality_dropout_p", self.modality_dropout_p
        )

        if self.structured_feature_dropout_p > 0.0 and self.modality_dropout_p > 0.0:
            raise ValueError(
                "structured_feature_dropout_p and modality_dropout_p cannot both "
                "be > 0. Disable the legacy structured dropout when using the new "
                "modality-dropout path."
            )

    def _initialize_structured_dropout_mask(
        self, adata_var_names: Sequence[str]
    ) -> None:
        keep_mask = np.ones(len(adata_var_names), dtype=bool)
        structured_keep_path = getattr(
            self.args, "structured_feature_dropout_keep_path", None
        )
        if structured_keep_path is not None and self.structured_feature_dropout_p > 0:
            keep_names = set(self._read_feature_names(structured_keep_path))
            keep_mask = self._build_feature_mask(keep_names, adata_var_names)

        self.register_buffer(
            "_structured_dropout_keep_mask",
            torch.from_numpy(keep_mask),
            persistent=False,
        )

    @classmethod
    def _load_modality_dropout_masks(
        cls,
        groups_dir: Path,
        protected_path: Path,
        adata_var_names: Sequence[str],
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        if not groups_dir.is_dir():
            raise NotADirectoryError(
                f"modality_dropout_groups_dir is not a directory: '{groups_dir}'"
            )

        adata_var_name_set = set(adata_var_names)
        protected_names = set(cls._read_feature_names(protected_path))
        protected_mask = cls._build_feature_mask(protected_names, adata_var_names)

        missing_protected = protected_names - adata_var_name_set
        if missing_protected:
            logging.warning(
                "Modality dropout protected list %s contains %d features not found "
                "in adata.var; they will be ignored.",
                protected_path,
                len(missing_protected),
            )
        if protected_names and not protected_mask.any():
            raise ValueError(
                "No protected modality-dropout features were found in adata.var. "
                f"Checked '{protected_path}'."
            )

        group_paths = sorted(
            path
            for path in groups_dir.glob("*_only.txt")
            if path.resolve() != protected_path.resolve()
        )
        if not group_paths:
            raise ValueError(
                f"No modality group files found in '{groups_dir}' after excluding "
                f"'{protected_path.name}'."
            )

        group_names: list[str] = []
        group_masks: list[np.ndarray] = []

        for group_path in group_paths:
            group_names_set = set(cls._read_feature_names(group_path))
            protected_overlap = group_names_set & protected_names
            if protected_overlap:
                overlap_examples = sorted(protected_overlap)[:10]
                raise ValueError(
                    f"Modality group '{group_path.name}' overlaps with protected "
                    f"features from '{protected_path.name}'. Examples: "
                    f"{overlap_examples}"
                )

            missing_group_names = group_names_set - adata_var_name_set
            if missing_group_names:
                logging.warning(
                    "Modality group %s contains %d features not found in adata.var; "
                    "they will be ignored.",
                    group_path,
                    len(missing_group_names),
                )

            group_mask = cls._build_feature_mask(group_names_set, adata_var_names)
            if not group_mask.any():
                logging.info(
                    "Skipping modality group %s because it does not match any "
                    "features in adata.var after alignment.",
                    group_path,
                )
                continue

            group_names.append(group_path.stem)
            group_masks.append(group_mask)

        if not group_masks:
            raise ValueError(
                "No non-protected modality-dropout groups remain after aligning "
                f"'{groups_dir}' to the current adata.var."
            )

        group_mask_matrix = np.stack(group_masks, axis=0)
        overlapping_indices = np.where(group_mask_matrix.sum(axis=0) > 1)[0]
        if overlapping_indices.size > 0:
            examples = []
            for idx in overlapping_indices[:5]:
                overlapping_groups = [
                    group_names[group_idx]
                    for group_idx in np.where(group_mask_matrix[:, idx])[0]
                ]
                examples.append(f"{adata_var_names[idx]} -> {overlapping_groups}")
            raise ValueError(
                "Modality dropout groups must be disjoint after aligning to "
                f"adata.var. Examples: {examples}"
            )

        return group_names, protected_mask, group_mask_matrix

    def _initialize_modality_dropout_masks(
        self, adata_var_names: Sequence[str]
    ) -> None:
        n_vars = len(adata_var_names)
        protected_mask = np.zeros(n_vars, dtype=bool)
        group_mask_matrix = np.zeros((0, n_vars), dtype=bool)
        drop_icd_group_mask = np.zeros(0, dtype=bool)
        drop_meds_group_mask = np.zeros(0, dtype=bool)
        self._modality_dropout_group_names: list[str] = []

        if self.modality_dropout_p > 0.0:
            groups_dir_value = getattr(self.args, "modality_dropout_groups_dir", None)
            if groups_dir_value is None:
                raise ValueError(
                    "modality_dropout_groups_dir must be set when modality_dropout_p "
                    "is > 0."
                )
            groups_dir = Path(groups_dir_value)
            protected_path_value = getattr(
                self.args, "modality_dropout_protected_path", None
            )
            protected_path = (
                Path(protected_path_value)
                if protected_path_value is not None
                else groups_dir / "ehr_only.txt"
            )

            (
                self._modality_dropout_group_names,
                protected_mask,
                group_mask_matrix,
            ) = self._load_modality_dropout_masks(
                groups_dir=groups_dir,
                protected_path=protected_path,
                adata_var_names=adata_var_names,
            )
            logging.info(
                "Loaded %d modality-dropout groups from %s.",
                len(self._modality_dropout_group_names),
                groups_dir,
            )

            group_name_to_index = {
                group_name: idx
                for idx, group_name in enumerate(self._modality_dropout_group_names)
            }
            drop_icd_group_mask = np.zeros(
                len(self._modality_dropout_group_names), dtype=bool
            )
            drop_meds_group_mask = np.zeros(
                len(self._modality_dropout_group_names), dtype=bool
            )

            icd_group_name = getattr(
                self.args, "modality_dropout_icd_group_name", None
            )
            if (
                icd_group_name is not None
                and icd_group_name in group_name_to_index
            ):
                drop_icd_group_mask[group_name_to_index[icd_group_name]] = True

            meds_group_name = getattr(
                self.args, "modality_dropout_meds_group_name", None
            )
            if (
                meds_group_name is not None
                and meds_group_name in group_name_to_index
            ):
                drop_meds_group_mask[group_name_to_index[meds_group_name]] = True

        self.register_buffer(
            "_modality_dropout_protected_mask",
            torch.from_numpy(protected_mask),
            persistent=False,
        )
        self.register_buffer(
            "_modality_dropout_group_masks",
            torch.from_numpy(group_mask_matrix),
            persistent=False,
        )
        self.register_buffer(
            "_modality_dropout_drop_icd_group_mask",
            torch.from_numpy(drop_icd_group_mask),
            persistent=False,
        )
        self.register_buffer(
            "_modality_dropout_drop_meds_group_mask",
            torch.from_numpy(drop_meds_group_mask),
            persistent=False,
        )

    def set_pos_weight(self, pos_weight: torch.Tensor) -> None:
        if pos_weight.dim() != 1:
            raise ValueError(f"pos_weight must be 1D (n_targets,), got {tuple(pos_weight.shape)}")

        pos_weight = pos_weight.detach().float().to(next(self.parameters()).device)

        if hasattr(self, "pos_weight") and isinstance(getattr(self, "pos_weight"), torch.Tensor):
            self.pos_weight = pos_weight
        else:
            self.register_buffer("pos_weight", pos_weight, persistent=False)

        self._bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.pos_weight)

    def _load_icd_embeddings(self) -> None:
        try:
            path = os.path.join(self.args.data_root_path, self.args.icd_embeddings_name)  # type: ignore[arg-type]
            embeddings = torch.from_numpy(np.load(path)).float()

            self.icd_embeddings_projector = nn.Linear(
                embeddings.shape[1], self.args.hidden_dim
            )
        except Exception as e:
            print(f"Error loading ICD embeddings: {e}")
            embeddings = None
            self.icd_embeddings_projector = None

        self.icd_embeddings: torch.Tensor
        self.register_buffer("icd_embeddings", embeddings)

    def _load_icd_prognosis_embeddings(self) -> None:
        hierarchy_levels = parse_icd_hierarchy_levels(
            getattr(self.args, "icd_hierarchy_level", None)
        )
        embedding_tensors: list[torch.Tensor] = []

        for level in hierarchy_levels:
            embedding_name, _ = get_icd_embedding_names(self.args, level)
            path = os.path.join(
                self.args.data_root_path, embedding_name
            )  # type: ignore[arg-type]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"ICD prognosis embedding file not found for level {level}: '{path}'"
                )
            tensor = torch.from_numpy(np.load(path)).float()
            embedding_tensors.append(tensor)

        if not embedding_tensors:
            raise ValueError(
                "No ICD prognosis embeddings loaded. Check icd_hierarchy_level configuration."
            )

        expected_dim = embedding_tensors[0].shape[1]
        for tensor in embedding_tensors[1:]:
            if tensor.shape[1] != expected_dim:
                raise ValueError(
                    "All ICD prognosis embedding matrices must share the same feature dimension."
                )

        embeddings = (
            torch.cat(embedding_tensors, dim=0)
            if len(embedding_tensors) > 1
            else embedding_tensors[0]
        )

        self.icd_prognosis_embeddings: torch.Tensor
        self.register_buffer("icd_prognosis_embeddings", embeddings)

        self.icd_prognosis_embeddings_projector = nn.Linear(
            self.icd_prognosis_embeddings.shape[1], self.args.decoder_hidden_dim
        )

    def _load_meds_embeddings(self) -> None:
        try:
            path = os.path.join(self.args.data_root_path, self.args.meds_embeddings_name)  # type: ignore[arg-type]
            embeddings = torch.from_numpy(np.load(path)).float()

            self.meds_embeddings_projector = nn.Linear(
                embeddings.shape[1], self.args.hidden_dim
            )

        except Exception as e:
            print(f"Error loading medication embeddings: {e}")
            embeddings = None
            self.meds_embeddings_projector = None

        self.meds_embeddings: torch.Tensor
        self.register_buffer("meds_embeddings", embeddings)

    def forward(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        split: str,
    ) -> Dict[str, Any]:
        (
            x,
            icd_multi_hot,
            meds_multi_hot,
            prognosis_targets, # shape (B, n_targets), with -1 for targets before assessment zero (or no GP record), 0 for negatives, and positive integers for months to event for positive cases
            observed_timespan_months,
        ) = batch

        device = self.cls_embedding.device
        x = x.to(device)
        icd_multi_hot = icd_multi_hot.to(device) if icd_multi_hot is not None else None
        meds_multi_hot = (
            meds_multi_hot.to(device) if meds_multi_hot is not None else None
        )

        bottleneck = self._encode(x, icd_multi_hot, meds_multi_hot)
        # ------------------------------------------------------------------
        # Decoding and Prognosis prediction
        # ------------------------------------------------------------------
        prediction_tokens = self.icd_prognosis_embeddings_projector(
            self.icd_prognosis_embeddings
        ).expand(x.shape[0], -1, -1)
        # Add random temporal timespan to prediction tokens to create a query for the decoder
        random_temporal_timespan = self._sample_temporal_timespan(
            prognosis_targets=prognosis_targets,
            observed_timespan_months=observed_timespan_months,
        ) # shape (B, n_targets), with integer month samples between 0 and observed timespan (inclusive)
        pe = self._generate_temporal_token(
            random_temporal_timespan, self.args.decoder_hidden_dim
        )
        pred_logits = self.cross_attn_decoder(bottleneck, prediction_tokens + pe, None)
        preds = self.predictor(pred_logits) # shape (B, n_targets, 1)
        # For positive cases, check if date of diagnosis is within the random temporal timespan
        # (and we thus want a positive prediction)
        pos_mask = prognosis_targets > 0
        target = torch.zeros_like(prognosis_targets, dtype=torch.float32, device=device) # shape (B, n_targets)
        target[pos_mask] = (
            prognosis_targets[pos_mask] < random_temporal_timespan[pos_mask] # True if real diagnosis is within the sampled timespan, False otherwise
        ).float()
        logits = preds.squeeze(-1)  # (B, n_targets)

        if self.use_pos_weight and (self._bce is not None):
            loss = self._bce(logits, target)  # (B, n_targets)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        # Do not stack negative_sample_weight when using pos_weight
        if (not self.use_pos_weight) and self.negative_sample_weight != 1.0 and split == "train":
            weights = torch.ones_like(target)
            negative_mask = target == 0
            weights[negative_mask] = self.negative_sample_weight
            loss = loss * weights
        
        mask = prognosis_targets != -1
        if getattr(self.args, "repquery_average_loss_over_targets", False):
            # loss has shape (B, n_targets) with axis 0 being the batch dimension and axis 1 being the target dimension
            # First, average over batch, so we get mean loss per target (ignoring -1 targets which indicate no GP record or diagnosis before assessment)
            # Then, average over targets
            loss[~mask] = np.nan # set -1 targets to NaN so they are ignored in mean
            loss = loss.nanmean(axis=0).nanmean() # mean over targets, then mean over batch
        else:
            loss = loss[mask].mean()
        self.log(f"RepQuery.{split}.loss", loss, on_epoch=True, on_step=False)

        return {"loss": loss}

    # ------------------------------------------------------------------
    # Encoding pipeline
    # ------------------------------------------------------------------
    def _encode(
        self,
        x: Tensor,
        icd_multi_hot: Optional[Tensor],
        meds_multi_hot: Optional[Tensor],
    ) -> Tensor:
        ordered_tokens, padding_mask = self._order_and_filter(
            x, icd_multi_hot, meds_multi_hot
        )
        ordered_tokens, padding_mask = self._prepend_cls(ordered_tokens, padding_mask)

        # apply dropout
        ordered_tokens, padding_mask = self._apply_feature_dropout(
            ordered_tokens, padding_mask
        )

        encoded_tokens = self.encoder(  # type: ignore[arg-type]
            ordered_tokens,
            src_key_padding_mask=padding_mask,
        )
        bottleneck = self._extract_bottleneck(encoded_tokens)

        return bottleneck

    def _sample_temporal_timespan(
        self,
        prognosis_targets: Tensor, # shape (B, n_targets), with -1 for targets before assessment zero, 0 for negatives, and positive integers for months to event for positive cases
        observed_timespan_months: Tensor, # shape (B, n_targets) with the observed timespan in months for each subject and target
    ) -> Tensor:
        """Sample temporal query tokens according to the configured strategy."""

        device = prognosis_targets.device
        batch_size, n_targets = prognosis_targets.shape

        observed = observed_timespan_months.clamp_min(1.0)
        sampling_max_months = getattr(self.args, "temporal_sampling_max_months", None)
        if sampling_max_months is not None:
            observed = torch.minimum(
                observed,
                torch.full_like(observed, float(sampling_max_months)),
            ).clamp_min(1.0)

        # Shape (B, n_targets), with integer month samples between 0 and observed timespan (inclusive)
        uniform_samples = torch.floor(
            torch.rand(batch_size, n_targets, device=device) * observed
        ).long()

        strategy = getattr(
            self.args,
            "temporal_sampling_strategy",
            TemporalSamplingStrategy.UNIFORM,
        )

        center_mask = prognosis_targets > 0
        if sampling_max_months is not None:
            center_mask = center_mask & (prognosis_targets <= float(sampling_max_months))

        if strategy == TemporalSamplingStrategy.GAUSSIAN:
            if center_mask.any():
                std = float(getattr(self.args, "temporal_sampling_std_months", 6.0))
                mean = prognosis_targets[center_mask].float()
                std_tensor = torch.full(
                    size=mean.shape,
                    fill_value=std,
                    dtype=torch.float32,
                    device=device,
                )
                gaussian_samples = torch.normal(mean=mean, std=std_tensor)
                gaussian_samples = gaussian_samples.clamp_min(0.0)
                max_support = observed[center_mask].float()
                gaussian_samples = torch.minimum(gaussian_samples, max_support) # Gaussian samples are between 0 and observed timespan (inclusive)
                uniform_samples[center_mask] = torch.floor(gaussian_samples).long()

        if strategy == TemporalSamplingStrategy.DIAG_CENTERED:
            if center_mask.any():
                diag_times = prognosis_targets[center_mask].float().clamp_min(0.0)
                max_support = observed[center_mask].float()
                diag_times = torch.minimum(diag_times, max_support)

                before_mask = torch.rand_like(diag_times) < 0.5

                before_range = diag_times
                before_samples = torch.rand_like(diag_times) * before_range # between 0 and diag_times

                after_range = torch.clamp(max_support - diag_times, min=0.0)
                after_samples = diag_times + torch.rand_like(diag_times) * after_range # between diag_times and max_support

                centered_samples = torch.where(
                    before_mask, before_samples, after_samples
                )
                uniform_samples[center_mask] = torch.floor(centered_samples).long()

        fixed_p = float(getattr(self.args, "temporal_fixed_horizon_p", 0.0))
        fixed_horizons = getattr(self.args, "temporal_fixed_horizons_months", [])
        if fixed_p > 0.0 and len(fixed_horizons) > 0:
            use_fixed = torch.rand(batch_size, n_targets, device=device) < fixed_p

            horizons = torch.tensor(fixed_horizons, device=device, dtype=torch.long)
            horizon_idx = torch.randint(
                low=0,
                high=horizons.numel(),
                size=(batch_size, n_targets),
                device=device,
            )
            fixed_samples = horizons[horizon_idx]

            jitter = int(getattr(self.args, "temporal_fixed_horizon_jitter_months", 0))
            if jitter > 0:
                fixed_samples = fixed_samples + torch.randint(
                    low=-jitter,
                    high=jitter + 1,
                    size=(batch_size, n_targets),
                    device=device,
                )

            fixed_samples.clamp_min_(1)
            observed_int = torch.floor(observed).long().clamp_min(1)
            fixed_samples = torch.minimum(fixed_samples, observed_int)

            uniform_samples = torch.where(use_fixed, fixed_samples, uniform_samples)

        # Ensure at least 1 month is sampled to avoid data leakage for positives with very short timespan (since 0 means "diagnosis before or at assessment" in prognosis_targets)
        uniform_samples.clamp_min_(1)

        return uniform_samples

    def _update_best_epoch_loss(self, split: str) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is None or getattr(trainer, "sanity_checking", False):
            return
        if split not in self.loss_trackers:
            return

        metric_key = f"RepQuery.{split}.loss"
        metric_value = trainer.callback_metrics.get(metric_key)

        if metric_value is None:
            return

        if isinstance(metric_value, torch.Tensor):
            metric_scalar = metric_value.detach().float().item()
        elif hasattr(metric_value, "item"):
            metric_scalar = float(metric_value.item())
        else:
            metric_scalar = float(metric_value)

        self.loss_trackers[split].update("loss", metric_scalar)
        best_loss = self.loss_trackers[split].get_best_score("loss")

        if best_loss is None:
            return

        device = getattr(self, "device", torch.device("cpu"))
        best_tensor = torch.tensor(best_loss, device=device)

        self.log(
            f"RepQuery.{split}.loss.best",
            best_tensor,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_train_epoch_end(self) -> None:  # pragma: no cover
        super().on_train_epoch_end()
        self._update_best_epoch_loss("train")
    
    def on_train_start(self) -> None:  # pragma: no cover
        super().on_train_start()

        # Only if enabled
        if not self.use_pos_weight:
            return

        # Already initialized (e.g. set manually)
        if self._bce is not None:
            return

        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return

        train_loader = trainer.train_dataloader
        # train_dataloader can be list/dict depending on setup
        if isinstance(train_loader, (list, tuple)):
            train_loader = train_loader[0]
        elif isinstance(train_loader, dict):
            train_loader = next(iter(train_loader.values()))

        pos_weight = RepQuery.compute_pos_weight_from_loader(
            train_loader=train_loader,
            device=torch.device("cpu"),
            eps=self.pos_weight_eps,
            clip_max=self.pos_weight_clip_max,
        )
        self.set_pos_weight(pos_weight)

    def on_validation_epoch_end(self) -> None:  # pragma: no cover
        super().on_validation_epoch_end()
        self._update_best_epoch_loss("val")

    # =====================================================================
    # IG SUPPORT: build token embeddings + metadata (ICD/MEDS code ids)
    # =====================================================================

    @staticmethod
    def _make_code_id_block(select_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Builds a padded (B, max_k) tensor of code ids from a boolean selection mask.

        Inputs:
          select_mask: Bool tensor shaped (B, N_codes). True means "code present".

        Outputs:
          code_ids_block: Long tensor shaped (B, max_k) with code ids, padded with -1.
          padding_mask:   Bool tensor shaped (B, max_k), True where padding.
        """
        # select_mask: (B, N_codes)
        B, N = select_mask.shape

        # Count how many codes each subject has -> (B,)
        per_subject_counts = select_mask.sum(dim=1)

        # If nobody has any codes, max_k becomes 0 -> handle gracefully
        max_k = int(per_subject_counts.max().item()) if per_subject_counts.numel() > 0 else 0

        # Prepare outputs
        code_ids_block = torch.full(
            (B, max_k), -1, device=select_mask.device, dtype=torch.long
        )

        # Padding mask: True for padding positions
        idx = torch.arange(max_k, device=select_mask.device).expand(B, -1)
        padding_mask = idx >= per_subject_counts.unsqueeze(1)

        # Compute "rank" of each True within a row (0..k-1) to place code ids
        # rank[b, j] = how many Trues up to column j, minus 1. Only meaningful where select_mask is True.
        rank = select_mask.long().cumsum(dim=1) - 1  # (B, N_codes)

        # Get coordinates of True entries (row-major)
        coords = torch.nonzero(select_mask, as_tuple=False)  # (nnz, 2) with columns [b, code_id]
        if coords.numel() == 0:
            return code_ids_block, padding_mask

        b = coords[:, 0]          # (nnz,)
        code_id = coords[:, 1]    # (nnz,)
        pos = rank[b, code_id]    # (nnz,) position in 0..k-1 within that subject

        # Scatter code ids into the padded block
        code_ids_block[b, pos] = code_id

        return code_ids_block, padding_mask

    def _order_and_filter_with_meta(
        self,
        x: Tensor,
        icd_multi_hot: Optional[Tensor],
        meds_multi_hot: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Same as _order_and_filter, but also returns metadata mapping tokens back to:
          - ICD code ids
          - MEDS code ids

        Returns:
          ordered_tokens:       (B, T, hidden_dim)
          ordered_padding_mask: (B, T)
          meta: dict with:
              - icd_code_ids:  (B, T) long, -1 where not an ICD token
              - meds_code_ids: (B, T) long, -1 where not a MEDS token
              - x_feature_ids: (B, T) long, original x feature index for tabular tokens, else -1
        """
        x, icd_multi_hot, meds_multi_hot = self._apply_modality_dropout(
            self._apply_structured_feature_dropout(x),
            icd_multi_hot,
            meds_multi_hot,
        )

        token_embeddings: list[Tensor] = []
        padding_masks: list[Tensor] = []

        # We track code id tensors aligned with token_embeddings concatenation.
        # For non-ICD/MEDS/x tokens we fill with -1.
        token_icd_ids: list[Tensor] = []
        token_meds_ids: list[Tensor] = []
        token_x_feature_ids: list[Tensor] = []

        B = x.shape[0]
        device = x.device

        # --------------------------
        # ICD tokens (variable count)
        # --------------------------
        if self.icd_embeddings is not None and icd_multi_hot is not None:
            E_icd = self.icd_embeddings.shape[1]

            use_temporal = (
                bool(getattr(self.args, "use_temporal_token", False))
                and icd_multi_hot.dtype != torch.bool
            )

            # Select which codes are present
            icd_select = (icd_multi_hot > 0) if use_temporal else icd_multi_hot.bool()  # (B, N_icd_codes)

            # Create padded code-id block (B, max_icd) and its padding mask
            icd_code_ids_block, icd_padding_mask = self._make_code_id_block(icd_select)  # both (B, max_icd)

            max_icd = icd_code_ids_block.shape[1]

            # Build the embedding block (B, max_icd, E_icd)
            icd_block = torch.zeros(B, max_icd, E_icd, device=device)
            # Fill only non-padding positions with their embedding vectors
            if max_icd > 0:
                valid_pos = ~icd_padding_mask  # (B, max_icd)
                # Map (b, pos) -> code_id
                code_ids = icd_code_ids_block[valid_pos]  # (nnz,)
                # Lookup embeddings
                icd_block[valid_pos] = self.icd_embeddings[code_ids]  # (nnz, E_icd)

            # Add temporal PE (if enabled) using the months from icd_multi_hot[b, code_id]
            if use_temporal and max_icd > 0:
                months_block = torch.zeros((B, max_icd), device=device, dtype=torch.float32)

                coords = torch.nonzero(icd_select, as_tuple=False)  # (nnz, 2): [b, code_id]
                if coords.numel() > 0:
                    b_idx = coords[:, 0]
                    c_idx = coords[:, 1]
                    # position of each selected code within its row (0..k-1)
                    pos = (icd_select.long().cumsum(dim=1) - 1)[b_idx, c_idx]
                    months_block[b_idx, pos] = icd_multi_hot[b_idx, c_idx].to(torch.float32)

                months_pe = self._generate_temporal_token(months_block, E_icd)
                icd_block = icd_block + months_pe

            # Project ICD embeddings into hidden_dim
            projected_icd = self.icd_embeddings_projector(icd_block)  # (B, max_icd, hidden_dim)

            token_embeddings.append(projected_icd)
            padding_masks.append(icd_padding_mask)

            # Meta alignment for this token chunk
            token_icd_ids.append(icd_code_ids_block)  # (B, max_icd)
            token_meds_ids.append(torch.full((B, max_icd), -1, device=device, dtype=torch.long))
            token_x_feature_ids.append(torch.full((B, max_icd), -1, device=device, dtype=torch.long))

        # ---------------------------
        # MEDS tokens (variable count)
        # ---------------------------
        if self.meds_embeddings is not None and meds_multi_hot is not None and meds_multi_hot.numel() > 0:
            E_meds = self.meds_embeddings.shape[1]

            meds_select = meds_multi_hot.bool()  # (B, N_meds_codes)

            meds_code_ids_block, meds_padding_mask = self._make_code_id_block(meds_select)  # (B, max_meds)
            max_meds = meds_code_ids_block.shape[1]

            meds_block = torch.zeros(B, max_meds, E_meds, device=device)
            if max_meds > 0:
                valid_pos = ~meds_padding_mask
                code_ids = meds_code_ids_block[valid_pos]
                meds_block[valid_pos] = self.meds_embeddings[code_ids]

            projected_meds = self.meds_embeddings_projector(meds_block)  # (B, max_meds, hidden_dim)

            token_embeddings.append(projected_meds)
            padding_masks.append(meds_padding_mask)

            token_icd_ids.append(torch.full((B, max_meds), -1, device=device, dtype=torch.long))
            token_meds_ids.append(meds_code_ids_block)
            token_x_feature_ids.append(torch.full((B, max_meds), -1, device=device, dtype=torch.long))

        # ---------------------------
        # Tabular x embeddings
        # ---------------------------
        feature_padding_masks: list[Tensor] = []
        for argname, module, indices in [
            ("x_num", self.num_embeddings, self.num_indices),
            ("x_bin", self.bin_embeddings, self.bin_indices),
            ("x_cat", self.cat_embeddings, self.cat_indices),
        ]:
            if module is None:
                continue

            feature_slice = x[..., indices]  # (B, n_feat_of_type)
            feature_mask = torch.isnan(feature_slice)  # (B, n_feat_of_type)
            feature_slice = feature_slice.clone()
            feature_slice[feature_mask] = 0  # replace NaNs with 0 before embedding

            if argname == "x_cat":
                feature_slice = feature_slice.to(torch.int64)  # categories must be int indices

            embeddings = module(feature_slice)  # (B, n_feat_of_type, hidden_dim)

            token_embeddings.append(embeddings)
            feature_padding_masks.append(feature_mask)

            # Add meta for these tokens: they are neither ICD nor MEDS
            L = embeddings.shape[1]
            token_icd_ids.append(torch.full((B, L), -1, device=device, dtype=torch.long))
            token_meds_ids.append(torch.full((B, L), -1, device=device, dtype=torch.long))
            feature_ids_np = np.flatnonzero(np.asarray(indices, dtype=bool)).astype(np.int64)
            feature_ids = torch.from_numpy(feature_ids_np).to(device=device, dtype=torch.long)
            token_x_feature_ids.append(feature_ids.unsqueeze(0).expand(B, -1))

        if feature_padding_masks:
            padding_masks.extend(feature_padding_masks)

        # ---------------------------
        # Concatenate everything
        # ---------------------------
        if token_embeddings:
            concatenated_embeddings = torch.cat(token_embeddings, dim=1)  # (B, T_raw, hidden_dim)
            concatenated_mask = torch.cat(padding_masks, dim=1)           # (B, T_raw)

            concat_icd_ids = torch.cat(token_icd_ids, dim=1)              # (B, T_raw)
            concat_meds_ids = torch.cat(token_meds_ids, dim=1)            # (B, T_raw)
            concat_x_feature_ids = torch.cat(token_x_feature_ids, dim=1)  # (B, T_raw)
        else:
            concatenated_embeddings = self._empty_token(x)
            concatenated_mask = torch.zeros((B, 0), device=device, dtype=torch.bool)
            concat_icd_ids = torch.zeros((B, 0), device=device, dtype=torch.long)
            concat_meds_ids = torch.zeros((B, 0), device=device, dtype=torch.long)
            concat_x_feature_ids = torch.zeros((B, 0), device=device, dtype=torch.long)

        # filter_and_pad_vectorized removes padded tokens and packs remaining tokens
        ordered_tokens, ordered_padding_mask, _ = filter_and_pad_vectorized(
            concatenated_embeddings,
            concatenated_mask,
        )
        # Keep metadata in the same packed order as ordered_tokens.
        _, _, ordered_icd_ids = filter_and_pad_vectorized(
            concatenated_embeddings,
            concatenated_mask,
            mask=concat_icd_ids,
        )
        _, _, ordered_meds_ids = filter_and_pad_vectorized(
            concatenated_embeddings,
            concatenated_mask,
            mask=concat_meds_ids,
        )
        _, _, ordered_x_feature_ids = filter_and_pad_vectorized(
            concatenated_embeddings,
            concatenated_mask,
            mask=concat_x_feature_ids,
        )

        meta = {
            "icd_code_ids": ordered_icd_ids,
            "meds_code_ids": ordered_meds_ids,
            "x_feature_ids": ordered_x_feature_ids,
        }

        return ordered_tokens, ordered_padding_mask, meta

    def build_preencoder_tokens(
        self,
        x: Tensor,
        icd_multi_hot: Optional[Tensor],
        meds_multi_hot: Optional[Tensor],
        include_cls: bool = True,
        deterministic: bool = True,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Public API for IG:

        Returns the actual token embeddings that go into the transformer encoder, plus:
          - padding_mask
          - meta mapping token positions to ICD/MEDS code ids and tabular feature ids

        deterministic:
          - if True: do NOT apply feature dropout (so IG is stable)
          - if False: respects self.training / feature_dropout_p (usually not wanted for IG)
        """
        device = self.cls_embedding.device

        # Move inputs to model device
        x = x.to(device)
        if icd_multi_hot is not None:
            icd_multi_hot = icd_multi_hot.to(device)
        if meds_multi_hot is not None:
            meds_multi_hot = meds_multi_hot.to(device)

        # Build tokens + meta
        tokens, padding_mask, meta = self._order_and_filter_with_meta(x, icd_multi_hot, meds_multi_hot)

        # Optionally prepend CLS tokens
        if include_cls:
            tokens, padding_mask = self._prepend_cls(tokens, padding_mask)

            # Also prepend -1 to code-id meta for CLS positions (CLS is not a code)
            B = tokens.shape[0]
            n_cls = int(self.args.n_cls_tokens)
            cls_pad = torch.full((B, n_cls), -1, device=device, dtype=torch.long)

            meta["icd_code_ids"] = torch.cat([cls_pad, meta["icd_code_ids"]], dim=1)
            meta["meds_code_ids"] = torch.cat([cls_pad, meta["meds_code_ids"]], dim=1)
            meta["x_feature_ids"] = torch.cat([cls_pad, meta["x_feature_ids"]], dim=1)

        # For IG, we want deterministic token stream -> skip dropout
        if (not deterministic) and self.feature_dropout_p > 0.0:
            tokens, padding_mask = self._apply_feature_dropout(tokens, padding_mask)

        return tokens, padding_mask, meta

    def predict_logits_at_horizon_years_from_tokens(
        self,
        tokens: Tensor,
        padding_mask: Tensor,
        horizon_years: Union[int, float, Tensor],
    ) -> Tensor:
        """
        Public API for token-level IG:

        Given pre-encoder token embeddings (B, T, hidden_dim) + padding mask,
        compute deterministic logits (B, n_targets) at a fixed horizon.
        """
        device = self.cls_embedding.device

        tokens = tokens.to(device)               # ensure correct device
        padding_mask = padding_mask.to(device)   # ensure correct device

        # Encode: identical to _encode, but skip token construction
        encoded_tokens = self.encoder(tokens, src_key_padding_mask=padding_mask)
        bottleneck = self._extract_bottleneck(encoded_tokens)  # (B, n_cls_tokens, hidden_dim)

        B = tokens.shape[0]
        n_targets = int(self.icd_prognosis_embeddings.shape[0])

        # Prediction tokens
        prediction_tokens = self.icd_prognosis_embeddings_projector(
            self.icd_prognosis_embeddings
        ).expand(B, -1, -1)  # (B, n_targets, d_dec)

        # Horizon in months (IG implementation assumes years API)
        if isinstance(horizon_years, (int, float)):
            horizon_months = float(horizon_years) * 12.0
            horizon_block = torch.full((B, n_targets), horizon_months, device=device, dtype=torch.float32)
        else:
            hm = horizon_years.to(device).float() * 12.0
            if hm.numel() == 1:
                horizon_block = torch.full((B, n_targets), float(hm.item()), device=device, dtype=torch.float32)
            else:
                hm = hm.view(B, 1)
                horizon_block = hm.expand(B, n_targets)

        pe = self._generate_temporal_token(horizon_block, self.args.decoder_hidden_dim)
        query_tokens = prediction_tokens + pe  # (B, n_targets, d_dec)

        decoded = self.cross_attn_decoder(bottleneck, query_tokens, None)
        logits = self.predictor(decoded).squeeze(-1)  # (B, n_targets)

        return logits

    # =====================================================================
    # PURPOSE: deterministic logits at a fixed horizon (months / years),
    #          reusing the exact decoding path of generate_and_save_1year_bin_logits,
    #          but without torch.no_grad and without looping over all bins.
    # =====================================================================
    # Core deterministic forward for a single fixed horizon in months
    def predict_logits_at_horizon_months(
        self,
        x: Tensor,
        icd_multi_hot: Optional[Tensor],
        meds_multi_hot: Optional[Tensor],
        horizon_months: Union[int, float, Tensor],
    ) -> Tensor:
        """
        Deterministic prognosis logits for a *fixed* horizon.
        Returns logits shaped (B, n_targets).

        This mirrors the deterministic evaluation path used in
        `generate_and_save_1year_bin_logits()` (decoder query = prognosis tokens + fixed PE),
        but is usable for IG (i.e., supports gradients w.r.t. inputs).
        """
        device = self.cls_embedding.device

        # Move to device
        x = x.to(device)
        if icd_multi_hot is not None:
            icd_multi_hot = icd_multi_hot.to(device)
        if meds_multi_hot is not None:
            meds_multi_hot = meds_multi_hot.to(device)

        B = x.shape[0]
        n_targets = int(self.icd_prognosis_embeddings.shape[0])  # number of prognosis tokens/targets

        # Encode inputs -> bottleneck (B, n_cls_tokens, hidden_dim)
        bottleneck = self._encode(x, icd_multi_hot, meds_multi_hot)

        # Build prediction tokens (B, n_targets, decoder_hidden_dim)
        prediction_tokens = self.icd_prognosis_embeddings_projector(
            self.icd_prognosis_embeddings
        ).expand(B, -1, -1)

        # Build a temporal block with the same horizon for all targets in the batch
        if isinstance(horizon_months, (int, float)):
            horizon_block = torch.full(
                (B, n_targets),
                float(horizon_months),
                device=device,
                dtype=torch.float32,
            )
        else:
            # Tensor input: allow shape (B,) or (B, 1) or scalar tensor
            hm = horizon_months.to(device)
            if hm.numel() == 1:
                horizon_block = torch.full(
                    (B, n_targets),
                    float(hm.item()),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                # Expect per-sample horizon (B,) -> broadcast across targets
                hm = hm.view(B, 1).float()
                horizon_block = hm.expand(B, n_targets)

        # Add temporal positional encoding to query tokens
        pe = self._generate_temporal_token(horizon_block, self.args.decoder_hidden_dim)
        query_tokens = prediction_tokens + pe

        # Decode and predict
        decoded = self.cross_attn_decoder(bottleneck, query_tokens, None)  # (B, n_targets, d_dec)
        preds = self.predictor(decoded).squeeze(-1)  # (B, n_targets)

        return preds

    # Thin wrapper that converts years -> months (*12) and calls the months function
    def predict_logits_at_horizon_years(
        self,
        x: Tensor,
        icd_multi_hot: Optional[Tensor],
        meds_multi_hot: Optional[Tensor],
        horizon_years: Union[int, float, Tensor],
    ) -> Tensor:
        """
        Convenience wrapper: horizon in years -> months.
        Returns logits shaped (B, n_targets).
        """
        if isinstance(horizon_years, (int, float)):
            return self.predict_logits_at_horizon_months(
                x, icd_multi_hot, meds_multi_hot, horizon_months=float(horizon_years) * 12.0
            )
        else:
            return self.predict_logits_at_horizon_months(
                x, icd_multi_hot, meds_multi_hot, horizon_months=horizon_years.float() * 12.0
            )

    # multiple horizons
    def predict_logits_at_horizons_years(
        self,
        x: Tensor,
        icd_multi_hot: Optional[Tensor],
        meds_multi_hot: Optional[Tensor],
        horizons_years: Sequence[Union[int, float]],
    ) -> Tensor:
        """
        Multiple horizons in one call.
        Returns logits shaped (B, n_targets, H) where H=len(horizons_years).
        Computed in a loop to keep memory predictable.
        """
        outs = []
        for h in horizons_years:
            outs.append(
                self.predict_logits_at_horizon_years(x, icd_multi_hot, meds_multi_hot, h)
            )  # each (B, n_targets)
        return torch.stack(outs, dim=-1)  # (B, n_targets, H)

    # Adapter specifically for dataloader batch format
    def logits_from_batch_for_ig(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        horizon_years: Union[int, float],
        output_type: str = "logit",
    ) -> Tensor:
        """
        Helper for IG/eval code: takes your dataset batch tuple and produces
        deterministic horizon logits or probabilities.

        Returns:
          - logits/probs shaped (B, n_targets)
        """
        x, icd_multi_hot, meds_multi_hot, _, _ = batch

        logits = self.predict_logits_at_horizon_years(
            x, icd_multi_hot, meds_multi_hot, horizon_years=horizon_years
        )

        if output_type == "logit":
            return logits
        if output_type == "prob":
            return torch.sigmoid(logits)

        raise ValueError(f"Invalid output_type={output_type}, expected 'logit' or 'prob'")

    def generate_and_save_1year_bin_logits(
        self,
        dataloader: DataLoader,
        split: str,
        wandb_logger: WandbLogger,
        save_logits: bool = False,
        save_embeddings: bool = False,
    ) -> None:
        one_year_pred_logits_list = []
        # Determine global max years
        max_years = 0
        for batch in tqdm(dataloader, desc="Determining global max years"):
            (
                _,
                _,
                _,
                _,
                observed_timespan_months, # shape (B, n_targets), contains the observed timespan in months for each subject and target
            ) = batch
            max_months = int(observed_timespan_months.max().item())
            max_years = max(max_years, math.ceil(max_months / 12))
        bottleneck_embeddings_list = []
        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc=f"Generating 1-year bin logits for {split}"
            ):
                (
                    x,
                    icd_multi_hot,
                    meds_multi_hot,
                    _,
                    _,
                ) = batch

                # Move all to GPU
                x = x.to(self.device)
                icd_multi_hot = (
                    icd_multi_hot.to(self.device) if icd_multi_hot is not None else None
                )
                meds_multi_hot = (
                    meds_multi_hot.to(self.device)
                    if meds_multi_hot is not None
                    else None
                )

                # Generate bottleneck tokens
                bottleneck = self._encode(x, icd_multi_hot, meds_multi_hot)
                if save_embeddings:
                    bottleneck_embeddings_list.append(bottleneck.detach().cpu().numpy())

                # Generate query tokens
                prediction_tokens = self.icd_prognosis_embeddings_projector(
                    self.icd_prognosis_embeddings
                ).expand(x.shape[0], -1, -1) # (B, n_targets, d_dec)

                # First bin should be 12 months (1 year), last bin should be max_years*12, step size is 12 months.
                bins_1year = (torch.linspace(0, max_years, max_years + 1) * 12 + 12)[
                    :-1
                ].expand(prediction_tokens.shape[0], prediction_tokens.shape[1], -1)
                pe = self._generate_temporal_token(
                    bins_1year, self.args.decoder_hidden_dim
                ) # (B, n_targets, max_years, d_dec)
                binned_one_year_prediction_tokens = prediction_tokens.unsqueeze(
                    2
                ).expand(
                    prediction_tokens.shape[0],
                    prediction_tokens.shape[1],
                    bins_1year.shape[-1],
                    prediction_tokens.shape[-1],
                ) + pe.to(
                    self.device
                ) # (B, n_targets, max_years, d_dec)

                # Calculate prediction logits
                yearly_pred_logits_list = []
                for i in range(max_years):
                    one_year_prediction_tokens = self.cross_attn_decoder(
                        bottleneck, binned_one_year_prediction_tokens[:, :, i, :], None
                    )
                    one_year_pred_logits = self.predictor(one_year_prediction_tokens)

                    yearly_pred_logits_list.append(
                        one_year_pred_logits.detach().cpu().numpy()
                    )
                one_year_pred_logits_list.append(
                    np.concatenate(yearly_pred_logits_list, axis=2)
                )

        one_year_pred_logits_concat = np.concatenate(one_year_pred_logits_list, axis=0)

        # Retrieve global_row once for both logits and embeddings
        # Requires dataloader.shuffle=False
        adata_view = dataloader.dataset.data
        global_row = adata_view.obs["global_row"].to_numpy(dtype=np.int64)

        assert one_year_pred_logits_concat.shape[0] == global_row.shape[0], \
            "Logits and global_row lengths do not match"

        if save_logits:
            if self.args.pretrained_weights_path:
                ckpt_path = Path(self.args.pretrained_weights_path)
                logits_path = os.path.join(
                    ckpt_path.parent.absolute(), f"1year_bin_logits_{split}.npz"
                )
            else:
                logits_path = os.path.join(
                    self.args.checkpoint_dir_path, f"1year_bin_logits_{split}.npz"
                )

            Path(logits_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                logits_path,
                logits=one_year_pred_logits_concat,
                global_row=global_row,
            )

        if save_embeddings:
            bottleneck_embeddings = np.concatenate(bottleneck_embeddings_list, axis=0)
            # bottleneck_embeddings.shape == (N, n_cls_tokens, hidden_dim)

            if bottleneck_embeddings.shape[0] != global_row.shape[0]:
                raise ValueError(
                    f"Mismatch between bottleneck_embeddings (N={bottleneck_embeddings.shape[0]}) "
                    f"and global_row (N={global_row.shape[0]})."
                )

            bottleneck_path = os.path.join(
                self.args.checkpoint_dir_path,
                f"bottleneck_embeddings_{split}.npz",
            )
            Path(bottleneck_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                bottleneck_path,
                bottleneck_tokens=bottleneck_embeddings,
                global_row=global_row,
            )

        evaluate_logits(
            one_year_pred_logits_concat, # shape (N, n_targets, max_years)
            split,
            dataloader.dataset.data,
            self.args,
            wandb_logger,
        )

    def _generate_temporal_token(self, temporal_block: Tensor, d_in: int) -> Tensor:
        div_term = torch.exp(
            torch.arange(0, d_in, 2, device=temporal_block.device, dtype=torch.float32)
            * (-(math.log(10000.0) / d_in))
        )
        pos = temporal_block.unsqueeze(-1)
        pe = torch.empty(
            *temporal_block.shape,
            d_in,
            device=temporal_block.device,
            dtype=torch.float32,
        )
        pe.zero_()
        pe[..., 0::2] = torch.sin(pos * div_term)
        pe[..., 1::2] = torch.cos(pos * div_term)
        return pe

    def _order_and_filter(
        self,
        x: Tensor,
        icd_multi_hot: Optional[Tensor],
        meds_multi_hot: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        x, icd_multi_hot, meds_multi_hot = self._apply_modality_dropout(
            self._apply_structured_feature_dropout(x),
            icd_multi_hot,
            meds_multi_hot,
        )

        token_embeddings: list[Tensor] = []
        padding_masks: list[Tensor] = []

        B = x.shape[0]
        device = x.device

        if self.icd_embeddings is not None and icd_multi_hot is not None:
            E_icd = self.icd_embeddings.shape[1]
            use_temporal = (
                bool(getattr(self.args, "use_temporal_token", False))
                and icd_multi_hot.dtype != torch.bool
            )
            icd_select = (icd_multi_hot > 0) if use_temporal else icd_multi_hot.bool()
            per_subject_counts = icd_select.sum(dim=1)
            max_icd = int(per_subject_counts.max().item())

            icd_block = torch.zeros(B, max_icd, E_icd, device=device)
            idx = torch.arange(max_icd, device=device).expand(B, -1)
            padding_mask = idx >= per_subject_counts.unsqueeze(1)
            icd_block[~padding_mask] = self.icd_embeddings.expand(x.shape[0], -1, -1)[
                icd_select
            ]

            # Add sinusoidal positional encoding as temporal tokens
            if use_temporal:
                months_block = torch.zeros_like(idx, dtype=torch.float32)
                months_block[~padding_mask] = icd_multi_hot[icd_select].to(
                    months_block.dtype
                )
                months_pe = self._generate_temporal_token(months_block, E_icd)
                icd_block = icd_block + months_pe

            projected_icd = self.icd_embeddings_projector(icd_block)
            token_embeddings.append(projected_icd)
            padding_masks.append(padding_mask)

        if self.meds_embeddings is not None and meds_multi_hot is not None and meds_multi_hot.numel() > 0:
            E_meds = self.meds_embeddings.shape[1]
            meds_select = meds_multi_hot.bool()
            per_subject_counts = meds_select.sum(dim=1)
            max_meds = int(per_subject_counts.max().item())

            meds_block = torch.zeros(B, max_meds, E_meds, device=device)
            idx = torch.arange(max_meds, device=device).expand(B, -1)
            padding_mask = idx >= per_subject_counts.unsqueeze(1)
            meds_block[~padding_mask] = self.meds_embeddings.expand(x.shape[0], -1, -1)[
                meds_select
            ]

            projected_meds = self.meds_embeddings_projector(meds_block)
            token_embeddings.append(projected_meds)
            padding_masks.append(padding_mask)

        feature_padding_masks: list[Tensor] = []
        for argname, module, indices in [
            ("x_num", self.num_embeddings, self.num_indices),
            ("x_bin", self.bin_embeddings, self.bin_indices),
            ("x_cat", self.cat_embeddings, self.cat_indices),
        ]:
            if module is None:
                continue

            feature_slice = x[..., indices]
            feature_mask = torch.isnan(feature_slice)
            feature_slice = feature_slice.clone()
            feature_slice[feature_mask] = 0

            if argname == "x_cat":
                feature_slice = feature_slice.to(torch.int64)

            embeddings = module(feature_slice)

            token_embeddings.append(embeddings)
            feature_padding_masks.append(feature_mask)

        if feature_padding_masks:
            padding_masks.extend(feature_padding_masks)

        if token_embeddings:
            concatenated_embeddings = torch.cat(token_embeddings, dim=1)
            concatenated_mask = torch.cat(padding_masks, dim=1)
        else:
            concatenated_embeddings = self._empty_token(x)
            concatenated_mask = torch.zeros((B, 0), device=x.device, dtype=torch.bool)

        ordered_tokens, ordered_padding_mask, _ = filter_and_pad_vectorized(
            concatenated_embeddings,
            concatenated_mask,
        )

        return ordered_tokens, ordered_padding_mask

    def _apply_structured_feature_dropout(self, x: Tensor) -> Tensor:
        if not self.training or self.structured_feature_dropout_p <= 0.0:
            return x
        if torch.rand(1, device=x.device).item() >= self.structured_feature_dropout_p:
            return x

        x = x.clone()
        x[:, ~self._structured_dropout_keep_mask] = float("nan")
        return x

    def _apply_modality_dropout(
        self,
        x: Tensor,
        icd_multi_hot: Optional[Tensor],
        meds_multi_hot: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if not self.training or self.modality_dropout_p <= 0.0:
            return x, icd_multi_hot, meds_multi_hot
        if self._modality_dropout_group_masks.numel() == 0:
            return x, icd_multi_hot, meds_multi_hot

        n_groups = self._modality_dropout_group_masks.shape[0]
        dropped_groups = torch.rand(n_groups, device=x.device) < self.modality_dropout_p
        if not dropped_groups.any():
            return x, icd_multi_hot, meds_multi_hot

        dropped_feature_mask = self._modality_dropout_group_masks[dropped_groups].any(dim=0)
        drop_icd = bool(
            self._modality_dropout_drop_icd_group_mask.numel() > 0
            and self._modality_dropout_drop_icd_group_mask[dropped_groups].any()
        )
        drop_meds = bool(
            self._modality_dropout_drop_meds_group_mask.numel() > 0
            and self._modality_dropout_drop_meds_group_mask[dropped_groups].any()
        )
        if not dropped_feature_mask.any() and not drop_icd and not drop_meds:
            return x, icd_multi_hot, meds_multi_hot

        if dropped_feature_mask.any():
            x = x.clone()
            x[:, dropped_feature_mask] = float("nan")
        if drop_icd and icd_multi_hot is not None:
            icd_multi_hot = torch.zeros_like(icd_multi_hot)
        if drop_meds and meds_multi_hot is not None:
            meds_multi_hot = torch.zeros_like(meds_multi_hot)
        return x, icd_multi_hot, meds_multi_hot

    def _prepend_cls(
        self, ordered_tokens: Tensor, padding_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        cls_tokens = self.cls_embedding.unsqueeze(0).expand(
            ordered_tokens.shape[0], -1, -1
        )
        cls_padding = torch.zeros(
            padding_mask.shape[0],
            self.args.n_cls_tokens,
            dtype=torch.bool,
            device=padding_mask.device,
        )

        ordered_tokens = torch.cat([cls_tokens, ordered_tokens], dim=1)
        padding_mask = torch.cat([cls_padding, padding_mask], dim=1)

        return ordered_tokens, padding_mask

    def _extract_bottleneck(self, encoded_tokens: Tensor) -> Tensor:
        return encoded_tokens[:, :self.args.n_cls_tokens]

    def _empty_token(self, x: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], 0, self.args.hidden_dim, device=x.device)

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Any:  # pragma: no cover
        return self.forward(batch, "train")

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Any:  # pragma: no cover
        return self.forward(batch, "val")

    def configure_optimizers(self) -> Any:  # pragma: no cover
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        scheduler_config = self._build_scheduler(optimizer)

        if scheduler_config is None:
            return optimizer

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def _build_scheduler(
        self, optimizer: Optimizer
    ) -> Optional[Dict[str, Any]]:  # pragma: no cover
        """Create the learning-rate scheduler with linear warmup."""

        scheduler_name = getattr(self.args, "scheduler", "warmup")
        if scheduler_name == "warmup":
            warmup_epochs = max(int(getattr(self.args, "warmup_epochs", 0)), 0)

            if warmup_epochs <= 0:
                return None

            def lr_lambda(current_epoch: int) -> float:
                if current_epoch >= warmup_epochs:
                    return 1.0
                return float(current_epoch + 1) / float(warmup_epochs)

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

            return {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "linear_warmup",
            }

        if scheduler_name in {"none", "constant"}:
            return None

        raise ValueError(
            "Valid schedulers are 'warmup', 'none', or 'constant' for RepQuery."
        )

    def _apply_feature_dropout(
        self, tokens: Tensor, padding_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Randomly drop whole tokens (features) during training by turning them
        into padding tokens.

        tokens:        (B, T, D)
        padding_mask:  (B, T) with True for padding positions
        """
        if not self.training or self.feature_dropout_p <= 0.0:
            return tokens, padding_mask

        B, T, D = tokens.shape
        device = tokens.device

        # Never drop CLS tokens
        n_cls = self.args.n_cls_tokens
        if n_cls >= T:
            return tokens, padding_mask

        # Candidate positions for dropout (exclude already padded tokens)
        non_cls_mask = ~padding_mask[:, n_cls:]  # (B, T - n_cls)

        # Sample dropout for non-CLS, non-padded tokens
        drop_rand = torch.rand(B, T - n_cls, device=device)
        drop_mask = (drop_rand < self.feature_dropout_p) & non_cls_mask  # (B, T - n_cls)

        # Update padding mask: dropped tokens become padding
        new_padding_mask = padding_mask.clone()
        new_padding_mask[:, n_cls:] = padding_mask[:, n_cls:] | drop_mask

        # Zero out embeddings for dropped tokens
        tokens = tokens.clone()
        tokens[:, n_cls:][drop_mask] = 0.0

        return tokens, new_padding_mask
    
    @staticmethod
    @torch.no_grad()
    def compute_pos_weight_from_loader(
        train_loader,
        device: torch.device,
        eps: float = 1.0,
        clip_max: float = 50.0,
    ) -> torch.Tensor:
        """
        Computes per-target pos_weight[j] = (N_neg[j] + eps) / (N_pos[j] + eps),
        ignoring missing labels (prognosis_targets == -1).

        Positive event indicator: prognosis_targets > 0.

        Safeguard:
        - If N_pos[j] == 0 (no positives in TRAIN), set pos_weight[j] = 1.0
            to avoid huge weights driven purely by eps.
        """
        pos_counts = None
        valid_counts = None

        for batch in train_loader:
            prognosis_targets = batch[3]  # (B, n_targets)
            prognosis_targets = prognosis_targets.to(device)
            valid = prognosis_targets != -1
            event = (prognosis_targets > 0) & valid  # (B, n_targets)

            pos = event.sum(dim=0).cpu().numpy()
            val = valid.sum(dim=0).cpu().numpy()

            if pos_counts is None:
                pos_counts = pos
                valid_counts = val
            else:
                pos_counts += pos
                valid_counts += val

        pos_counts = pos_counts.astype(np.float64)
        valid_counts = valid_counts.astype(np.float64)
        neg_counts = valid_counts - pos_counts

        # Base formula with smoothing
        pos_weight = (neg_counts + eps) / (pos_counts + eps)

        # If no positives exist for a target in TRAIN, pos_weight is irrelevant -> keep neutral
        no_pos = pos_counts == 0
        pos_weight[no_pos] = 1.0

        pos_weight = pos_weight ** 0.5 # the alpha could be a hyperparam or just be set to a value smaller than 1, e.g. 0.5 for squareroot

        # Pre-clip debug
        qs_raw = np.quantile(pos_weight, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
        frac_would_clip = float((pos_weight >= clip_max).mean())
        print(
            "[pos_weight raw] "
            f"would_clip={int((pos_weight >= clip_max).sum())} ({frac_would_clip:.3f}) "
            f"min={qs_raw[0]:.3g} median={qs_raw[1]:.3g} p90={qs_raw[2]:.3g} "
            f"p95={qs_raw[3]:.3g} p99={qs_raw[4]:.3g} max={qs_raw[5]:.3g}"
        )

        qs_pos = np.quantile(pos_counts, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
        print(
            "[pos_counts] "
            f"min={qs_pos[0]:.0f} median={qs_pos[1]:.0f} p90={qs_pos[2]:.0f} "
            f"p95={qs_pos[3]:.0f} p99={qs_pos[4]:.0f} max={qs_pos[5]:.0f}"
        )


        # Clip extremes
        pos_weight = np.minimum(pos_weight, clip_max).astype(np.float32)

        # ----------------------------
        # DEBUG / SUMMARY PRINTING
        # ----------------------------
        pw = pos_weight  # np.float32 (n_targets,)

        frac_clipped = float((pw >= (clip_max - 1e-6)).mean())
        n_targets = pw.shape[0]
        n_clipped = int((pw >= (clip_max - 1e-6)).sum())
        n_no_pos = int((pos_counts == 0).sum())

        qs = np.quantile(pw, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])

        print(
            "[pos_weight] "
            f"n_targets={n_targets} "
            f"no_pos={n_no_pos} "
            f"clipped={n_clipped} ({frac_clipped:.3f}) "
            f"min={qs[0]:.3g} "
            f"median={qs[1]:.3g} "
            f"p90={qs[2]:.3g} "
            f"p95={qs[3]:.3g} "
            f"p99={qs[4]:.3g} "
            f"max={qs[5]:.3g} "
            f"(clip_max={clip_max}, eps={eps})"
        )

        # Roughly, clip triggers when pos_counts < valid_counts / clip_max
        thresh = np.median(valid_counts) / float(clip_max)
        print(f"[pos_weight] median valid_count={np.median(valid_counts):.0f} -> clipping ~ pos < {thresh:.1f}")

        return torch.tensor(pos_weight, device=device)
