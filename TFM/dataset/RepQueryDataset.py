from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataset.utils import (
    npAnnData,
    icd_precompute,
    icd_build_representation,
    meds_precompute,
    meds_build_representation,
    icd_prognosis_precompute,
)
from config_types import Config


class RepQueryDataset(Dataset):
    def __init__(self, data: npAnnData, args: Config) -> None:
        self.data = data
        self.args = args

        # ---- X tensor (avoid copies if possible) ----
        # If data.X is a NumPy array (dense), this is zero-copy; otherwise falls back to np.asarray
        X_np = np.asarray(data.X, dtype=np.float32)
        self.data_tensor = torch.from_numpy(X_np)  # shape: [N, D]

        # ---- ICD prognosis precompute ----
        self.icd_prognosis_labels = icd_prognosis_precompute(self.data.obs, self.args)

        # ---- ICD mapping (precompute once) ----
        self.use_temporal_token: bool = bool(self.args.use_temporal_token)
        (
            self.num_icd_codes,
            self.diag_cols,
            self.diag_col_to_row,
            self.assessment_col,
            self._per_subject_icd_rows,
            self._per_subject_icd_months,
        ) = icd_precompute(self.data.obs, self.args)
        self.disable_icd = bool(getattr(self.args, "disable_icd", False))

        # ---- Medication mapping (via utils) ----
        if getattr(self.args, "disable_meds", False):
            self.num_meds_codes = 0
            self._meds_indptr = None
            self._meds_indices = None
            self._meds_names = None
            self._meds_name_to_row = None
            self._global_rows = np.arange(len(data), dtype=np.int64)
        else:
            (
                self.num_meds_codes,
                self._meds_indptr,
                self._meds_indices,
                self._meds_names,
                self._meds_name_to_row,
                self._global_rows,
            ) = meds_precompute(self.data.obs, self.args)

        # ---- Precompute observed timespan ----
        # Compute censoring dates per subject and disease, store as matrix
        # For positives, use the global censoring date
        # For negatives, use the minimum of global censoring date and per-subject censoring date (death/coma)        
        # Global censoring date in months from assessment date
        assessment_dates = pd.to_datetime(
            self.data.obs[self.args.assessment_date_field].astype(str)
        )
        global_censoring_months = torch.from_numpy(
            (
                (
                    pd.to_datetime(self.args.time_cutoff)
                    - assessment_dates
                ).dt.days
                / 30.4375
            ).to_numpy()
        )  # shape (N,) with months from assessment date to global cutoff

        # Per-subject censoring date is the minimum of death and coma dates (if present), or NaT if neither is present
        death_col = "Date_Death"
        coma_col = "Date_Coma"
        per_subject_censoring_date = pd.to_datetime(
            # Can't use min with skipna, which is just the default behavior
            np.fmin(
                pd.to_datetime(self.data.obs[death_col].astype(str)),
                pd.to_datetime(self.data.obs[coma_col].astype(str)),
            )
        )
        per_subject_death_coma_months = torch.from_numpy(
            (
                (
                    per_subject_censoring_date
                    - assessment_dates
                ).dt.days
                / 30.4375
            ).to_numpy()
        )  # shape (N,) with months from assessment date to minimum of death/coma censoring (or NaT if neither)
        observed_timespan_months = torch.where(
            torch.isnan(per_subject_death_coma_months),
            global_censoring_months,
            torch.minimum(global_censoring_months, per_subject_death_coma_months),
        )  # shape (N,) with months from assessment date to minimum of global cutoff and death/coma censoring (ignoring NaT)
        # Prognosis targets (supervision) can differ from ICD history vocabulary size when
        # icd_embeddings_* (input) and icd_codes_* (prognosis) use different code_to_row maps.
        n_targets = int(self.icd_prognosis_labels.shape[0])
        self.observed_timespan_months = (
            observed_timespan_months.unsqueeze(1).expand(-1, n_targets).clone()
        )  # shape (N, n_targets) with months from assessment date to minimum of global cutoff and death/coma censoring for each subject and target
        # For positives, use the global cutoff
        positive_mask = self.icd_prognosis_labels.T > 0
        # Override observed timespan for positives to be the global censoring months (since we want to sample temporal tokens between assessment date and global cutoff for positives)
        self.observed_timespan_months[positive_mask] = (
            global_censoring_months.unsqueeze(1)
            .expand(-1, n_targets)[positive_mask]
            .double()
        )
        # self.observed_timespan_months is now a tensor of shape (N, n_targets) with the observed timespan in months for each subject and target,
        # using global cutoff for positives and minimum of global cutoff and death/coma censoring for negatives

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Features (already torch)
        subject_tensor = self.data_tensor[index]  # [D], float32

        # ICD representation (multi-hot or months-to-event vector)
        icd_repr = icd_build_representation(
            index,
            self.num_icd_codes,
            self._per_subject_icd_rows,
            self._per_subject_icd_months,
            self.use_temporal_token,
        )
        if self.disable_icd:
            icd_repr = torch.zeros_like(icd_repr)

        meds_repr = meds_build_representation(
            index,
            self.num_meds_codes,
            self._meds_indptr,
            self._meds_indices,
            self._meds_names,
            self._meds_name_to_row,
            self._global_rows,
        )

        prognosis_target = torch.from_numpy(self.icd_prognosis_labels[:, index])

        # Return per-subject, per-target observed timespan in months
        observed_timespan = self.observed_timespan_months[
            index
        ]  # shape (n_targets,) with observed timespan in months for each target for this subject

        return (
            subject_tensor.view(-1),
            icd_repr,
            meds_repr,
            prognosis_target,
            observed_timespan,
        )

    def __len__(self) -> int:
        return len(self.data)
