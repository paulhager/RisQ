from __future__ import annotations

import os
import logging
import json
import re
from typing import Dict, List, Optional, Sequence

import pandas as pd
import anndata as ad
import numpy as np
import tensorstore as ts
import torch
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold

from config_types import Config, Task
from omegaconf import OmegaConf

_VALID_ICD_HIERARCHY_LEVELS = [1, 2, 3]
META_COLUMNS = ["Date_Death", "Date_Coma", "Has_GP_Records", "53-0.0", "global_row"]


def parse_icd_hierarchy_levels(raw_levels: Optional[Sequence[int] | int]) -> list[int]:
    """Validate and normalize ICD hierarchy configuration to sorted unique levels."""

    if raw_levels is None:
        raise ValueError("icd_hierarchy_level is required.")

    # Handle single integer input
    if isinstance(raw_levels, int):
        levels = [raw_levels]
    elif isinstance(raw_levels, Sequence) and not isinstance(raw_levels, (str, bytes)):
        try:
            levels = [int(level) for level in raw_levels]
        except (TypeError, ValueError) as exc:
            raise TypeError("icd_hierarchy_level must contain integers only.") from exc
    else:
        raise TypeError(
            "icd_hierarchy_level must be an integer or a sequence of integers."
        )

    normalized = sorted(set(levels))
    invalid = [
        level for level in normalized if level not in _VALID_ICD_HIERARCHY_LEVELS
    ]
    if invalid:
        raise ValueError(
            f"Unsupported icd_hierarchy_level values: {invalid}. Valid options are {_VALID_ICD_HIERARCHY_LEVELS}."
        )
    return normalized


def get_icd_embedding_names(args: Config, level: int) -> tuple[str, str]:
    """Return the embedding and mapping file names for a given ICD hierarchy level."""

    if level == 1:
        return args.icd_chapter_embeddings_name, args.icd_chapter_to_row_name
    if level == 2:
        return args.icd_block_range_embeddings_name, args.icd_block_range_to_row_name
    if level == 3:
        return args.icd_codes_embeddings_name, args.icd_codes_to_row_name
    raise ValueError(
        f"Unsupported ICD hierarchy level '{level}'. Valid options are {_VALID_ICD_HIERARCHY_LEVELS}."
    )


def filter_and_pad_vectorized(
    x: torch.Tensor,
    key_padding_mask: torch.Tensor,
    mask: torch.Tensor = torch.Tensor([]),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Calculate the number of valid features per subject and the maximum number of features
    valid_features_per_subject = (~key_padding_mask).sum(dim=1)
    max_features = int(valid_features_per_subject.max().item())

    # Initialize tensors for padded data and new mask
    x_filtered = torch.zeros(
        x.shape[0], max_features, *x.shape[2:], device=x.device, dtype=x.dtype
    )

    # Create a tensor of indices for gathering
    idx = torch.arange(x_filtered.size(1), device=x.device).expand(x.size(0), -1)
    key_padding_mask_filtered = idx >= valid_features_per_subject.unsqueeze(1)

    # Apply mask and copy data to x_filtered
    x_filtered[~key_padding_mask_filtered] = x[~key_padding_mask]

    if mask.nelement() > 0:
        new_mask = torch.zeros(
            x_filtered.shape[0],
            x_filtered.shape[1],
            dtype=mask.dtype,
            device=mask.device,
        )
        new_mask[~key_padding_mask_filtered] = mask[~key_padding_mask]
        return x_filtered, key_padding_mask_filtered, new_mask

    return x_filtered, key_padding_mask_filtered, torch.Tensor([])


def one_hot_encode_anndata(adata_splits: list[npAnnData]) -> list[npAnnData]:
    """Efficiently one-hot encodes categorical variables across multiple AnnData splits.

    Args:
        adata_splits: A list of AnnData objects (e.g., [target_adata_train, target_adata_val, target_adata_test])

    Returns:
        A list of AnnData objects with one-hot encoded categorical variables.
    """
    adata = npAnnData.concat(*adata_splits)
    cat_indices = (
        (adata.var["value_type"] == "Categorical single")
        & (adata.var["possible_preprocessing"] != "replace zero with NaN")
    ).to_numpy()
    cat_adata = adata[:, cat_indices]
    cat_vals = cat_adata.X

    unique_values = []
    for i in range(cat_vals.shape[1]):
        unique_values.append(np.unique(cat_vals[:, i][~np.isnan(cat_vals[:, i])]))

    encoder = OneHotEncoder(
        sparse_output=False, categories=unique_values, drop="if_binary"
    )
    cat_vals_ohe = encoder.fit_transform(cat_vals)
    n_unique_vals = [len(vals) for vals in unique_values]

    new_var_names = []
    for j, n_cat in enumerate(n_unique_vals):
        if n_cat > 2:
            for i in range(n_cat):
                new_var_names.append(f"{cat_adata.var_names[j]}_cat{i}")
        else:
            new_var_names.append(f"{cat_adata.var_names[j]}_bin")

    non_cat_adata = adata[:, ~cat_indices]
    non_cat_vals = non_cat_adata.X

    # Add _num to the non-categorical variable names so we can select the indices if needed
    non_cat_adata.var_names = [
        f"{var_name}_num" for var_name in non_cat_adata.var_names
    ]
    new_var_names.extend(non_cat_adata.var_names)

    # Merge
    new_data_matrix = np.hstack((cat_vals_ohe, non_cat_vals)).astype(np.float32)
    new_adata = npAnnData(X=new_data_matrix, uns=adata_splits[0].uns)
    new_adata.var_names = new_var_names
    new_adata.var_names_make_unique()
    new_adata.obs = adata.obs.copy()

    # Split
    new_adata_splits = []
    start = 0
    for adata_split in adata_splits:
        end = start + len(adata_split)
        new_adata_splits.append(new_adata[start:end])
        start = end

    return new_adata_splits


def split_indices(
    obs_indices, args: Config, adata: npAnnData
) -> tuple[list[int], list[int], list[int]]:
    val_size = args.val_size
    test_size = args.test_size

    dataset_seed = args.dataset_seed if args.dataset_seed else args.seed

    use_stratified = (
        len(args.target_outcomes) == 1
        and (args.task == Task.CLASSIFICATION or args.task == Task.SURVIVAL)
        and args.stratify
    )

    if args.cross_validation_n_folds == 0:
        # Original logic for single split
        train_obs_indices, val_test_obs_indices = train_test_split(
            obs_indices,
            test_size=(val_size + test_size),
            random_state=dataset_seed,
            stratify=(
                adata.obs[args.target_outcomes].iloc[obs_indices]
                if use_stratified
                else None
            ),
        )
        val_obs_indices, test_obs_indices = train_test_split(
            val_test_obs_indices,
            test_size=(test_size / (val_size + test_size)),
            random_state=dataset_seed,
            stratify=(
                adata.obs[args.target_outcomes].iloc[val_test_obs_indices]
                if use_stratified
                else None
            ),
        )
    else:
        if args.cross_validation_current_fold is None:
            raise ValueError(
                "cross_validation_current_fold must be set when cross_validation_n_folds > 0"
            )

        fold = int(args.cross_validation_current_fold)
        n_folds = int(args.cross_validation_n_folds)
        if fold < 0 or fold >= n_folds:
            raise ValueError(
                f"cross_validation_current_fold must be in [0, {n_folds - 1}], got {fold}"
            )

        cv_mode = getattr(args, "cv_mode", "fixed_test_inner")

        if cv_mode == "fixed_test_inner":
            # Legacy behavior: fixed test split, then K-fold on remaining data
            train_val_indices, test_obs_indices = train_test_split(
                obs_indices,
                test_size=test_size,
                random_state=dataset_seed,
                stratify=(
                    adata.obs[args.target_outcomes].iloc[obs_indices]
                    if use_stratified
                    else None
                ),
            )

            if use_stratified:
                splitter = StratifiedKFold(
                    n_splits=n_folds,
                    shuffle=True,
                    random_state=dataset_seed,
                )
                splits = list(
                    splitter.split(
                        train_val_indices,
                        adata.obs[args.target_outcomes].iloc[train_val_indices],
                    )
                )
            else:
                splitter = KFold(
                    n_splits=n_folds,
                    shuffle=True,
                    random_state=dataset_seed,
                )
                splits = list(splitter.split(train_val_indices))

            train_idx, val_idx = splits[fold]
            train_obs_indices = np.array(train_val_indices)[train_idx].tolist()
            val_obs_indices = np.array(train_val_indices)[val_idx].tolist()
        elif cv_mode == "outer_oof":
            # New behavior: outer K-fold test split + one random inner train/val split
            if use_stratified:
                outer_splitter = StratifiedKFold(
                    n_splits=n_folds,
                    shuffle=True,
                    random_state=dataset_seed,
                )
                outer_splits = list(
                    outer_splitter.split(
                        obs_indices, adata.obs[args.target_outcomes].iloc[obs_indices]
                    )
                )
            else:
                outer_splitter = KFold(
                    n_splits=n_folds,
                    shuffle=True,
                    random_state=dataset_seed,
                )
                outer_splits = list(outer_splitter.split(obs_indices))

            learning_idx, test_idx = outer_splits[fold]
            learning_obs_indices = np.array(obs_indices)[learning_idx].tolist()
            test_obs_indices = np.array(obs_indices)[test_idx].tolist()

            inner_val_size = args.cv_inner_val_size
            if inner_val_size is None:
                denom = max(1e-8, 1 - test_size)
                inner_val_size = min(max(val_size / denom, 1e-6), 1 - 1e-6)

            train_obs_indices, val_obs_indices = train_test_split(
                learning_obs_indices,
                test_size=inner_val_size,
                random_state=dataset_seed + fold,
                stratify=(
                    adata.obs[args.target_outcomes].iloc[learning_obs_indices]
                    if use_stratified
                    else None
                ),
            )
        else:
            raise ValueError(
                f"Unsupported cv_mode='{cv_mode}'. Supported modes: fixed_test_inner, outer_oof"
            )

    # Print checksum over the splits to verify that they are consistent across runs
    logging.info(
        f"Split checksums before dropping: train={sum(train_obs_indices)}, val={sum(val_obs_indices)}, test={sum(test_obs_indices)}"
    )

    return train_obs_indices, val_obs_indices, test_obs_indices


def _subsample_train_indices_nested_by_gp(
    train_obs_indices: list[int],
    gp_mask: np.ndarray,
    train_fraction: float,
    subset_seed: int,
) -> tuple[list[int], dict[str, float | int]]:
    """Build a nested train subset while preserving the GP/no-GP composition."""

    if not 0 < train_fraction <= 1:
        raise ValueError(
            f"low_data_split must be in (0, 1] when nested_low_data_split_preserve_gp is enabled, got {train_fraction}."
        )

    train_obs_array = np.asarray(train_obs_indices, dtype=int)
    train_gp_indices = train_obs_array[gp_mask[train_obs_array]]
    train_no_gp_indices = train_obs_array[~gp_mask[train_obs_array]]

    rng = np.random.default_rng(subset_seed)
    permuted_gp = rng.permutation(train_gp_indices)
    permuted_no_gp = rng.permutation(train_no_gp_indices)

    n_gp = min(len(permuted_gp), int(np.ceil(train_fraction * len(permuted_gp))))
    n_no_gp = min(
        len(permuted_no_gp), int(np.ceil(train_fraction * len(permuted_no_gp)))
    )

    selected_train_obs_indices = np.concatenate(
        [permuted_gp[:n_gp], permuted_no_gp[:n_no_gp]]
    ).tolist()

    return selected_train_obs_indices, {
        "requested_fraction": train_fraction,
        "requested_total_pool": len(train_obs_indices),
        "requested_gp_pool": len(train_gp_indices),
        "requested_no_gp_pool": len(train_no_gp_indices),
        "selected_total": len(selected_train_obs_indices),
        "selected_gp": n_gp,
        "selected_no_gp": n_no_gp,
    }


def _mean_impute(
    adata_train: npAnnData, adata_val: npAnnData, adata_test: npAnnData
) -> tuple[npAnnData, npAnnData, npAnnData]:
    """
    Impute missing values in three groups:
    - 'Numeric' features (value_type != 'Categorical single'): mean imputation.
    - Medication/diagnosis features (possible_preprocessing == 'replace zero with NaN'): NaN -> 0.
    - Other categorical single features: most-frequent imputation.

    If there are no numeric features, the mean imputation block is skipped to avoid
    passing a (n_samples, 0) array to SimpleImputer.
    """

    # ----- 1) Mean imputation for 'numeric' features -----
    numeric_indices = adata_train.var["value_type"] != "Categorical single"

    if numeric_indices.any():
        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")

        imp_mean.fit(adata_train.X[:, numeric_indices])

        adata_train.update_X_inplace(
            (slice(None), numeric_indices),
            imp_mean.transform(adata_train.X[:, numeric_indices]),
        )
        adata_val.update_X_inplace(
            (slice(None), numeric_indices),
            imp_mean.transform(adata_val.X[:, numeric_indices]),
        )
        adata_test.update_X_inplace(
            (slice(None), numeric_indices),
            imp_mean.transform(adata_test.X[:, numeric_indices]),
        )
    else:
        logging.info(
            "No non-categorical features (value_type != 'Categorical single'); "
            "skipping mean imputation."
        )

    # ----- 2) Meds/diagnoses: NaN -> 0 (unchanged) -----
    imp_most_frequent = SimpleImputer(
        missing_values=np.nan, strategy="most_frequent", keep_empty_features=True
    )

    # Don't impute medicine and diagnoses. Just set the NaN to 0
    med_diag_indices = (
        adata_train.var["possible_preprocessing"] == "replace zero with NaN"
    )
    adata_train.update_X_inplace(
        (slice(None), med_diag_indices),
        np.nan_to_num(adata_train.X[:, med_diag_indices], nan=0),
    )
    adata_val.update_X_inplace(
        (slice(None), med_diag_indices),
        np.nan_to_num(adata_val.X[:, med_diag_indices], nan=0),
    )
    adata_test.update_X_inplace(
        (slice(None), med_diag_indices),
        np.nan_to_num(adata_test.X[:, med_diag_indices], nan=0),
    )

    # ----- 3) Categorical single features: most frequent (unchanged) -----
    non_numeric_indices = (adata_train.var["value_type"] == "Categorical single") & (
        adata_train.var["possible_preprocessing"] != "replace zero with NaN"
    )

    if non_numeric_indices.any():
        imp_most_frequent.fit(adata_train.X[:, non_numeric_indices])

        adata_train.update_X_inplace(
            (slice(None), non_numeric_indices),
            imp_most_frequent.transform(adata_train.X[:, non_numeric_indices]),
        )
        adata_val.update_X_inplace(
            (slice(None), non_numeric_indices),
            imp_most_frequent.transform(adata_val.X[:, non_numeric_indices]),
        )
        adata_test.update_X_inplace(
            (slice(None), non_numeric_indices),
            imp_most_frequent.transform(adata_test.X[:, non_numeric_indices]),
        )

    return adata_train, adata_val, adata_test


def create_MultiDisease_train_val_test_split(
    adata: npAnnData,
    args: Config,
) -> tuple[
    tuple[npAnnData, npAnnData, npAnnData],
    tuple[list[int], list[int], list[int]],
]:
    # For validation and test require GP records
    gp_mask = (adata.obs["Has_GP_Records"] == True).values
    obs_indices = np.where(gp_mask)[0].tolist()
    args.stratify = False
    train_obs_indices, val_obs_indices, test_obs_indices = split_indices(
        obs_indices, args, adata
    )

    # Add positive cases from non-gp records to train set (negative cases with obs = -1 will be dropped further below)
    train_obs_indices = train_obs_indices + np.where(~gp_mask)[0].tolist()

    low_data_summary = None
    if args.low_data_split and args.nested_low_data_split_preserve_gp:
        train_obs_indices, low_data_summary = _subsample_train_indices_nested_by_gp(
            train_obs_indices=train_obs_indices,
            gp_mask=gp_mask,
            train_fraction=args.low_data_split,
            subset_seed=args.seed,
        )

    adata_train = adata[np.array(train_obs_indices)]
    adata_val = adata[np.array(val_obs_indices)]
    adata_test = adata[np.array(test_obs_indices)]

    if low_data_summary is not None:
        logging.info(
            "Nested low-data train subset: fraction=%.4f pool_total=%d pool_gp=%d pool_no_gp=%d selected_total=%d selected_gp=%d selected_no_gp=%d",
            low_data_summary["requested_fraction"],
            low_data_summary["requested_total_pool"],
            low_data_summary["requested_gp_pool"],
            low_data_summary["requested_no_gp_pool"],
            low_data_summary["selected_total"],
            low_data_summary["selected_gp"],
            low_data_summary["selected_no_gp"],
        )

    # Remove empty features
    empty_features = adata_train.var.index[np.isnan(adata_train.X).all(axis=0)].tolist()
    logging.info(f"Removing {len(empty_features)} empty features with all NaN values in training set: {empty_features}")
    adata_train = adata_train[:, ~adata_train.var.index.isin(empty_features)]
    adata_val = adata_val[:, ~adata_val.var.index.isin(empty_features)]
    adata_test = adata_test[:, ~adata_test.var.index.isin(empty_features)]

    if args.model_name in ["XGBoost", "TabPFN"]:
        if getattr(args, "xgb_use_imputation_onehot", True):
            adata_train, adata_val, adata_test = _mean_impute(adata_train, adata_val, adata_test)
            (adata_train, adata_val, adata_test) = one_hot_encode_anndata([adata_train, adata_val, adata_test])
        else:
            # rawcat mode: keep NaNs + keep categorical columns as-is (no one-hot)
            pass

        # Remove censored/invalid subjects (row-wise)
        valid_train = (adata_train.obs[args.target_outcomes] != -1).to_numpy().all(axis=1)
        valid_val   = (adata_val.obs[args.target_outcomes] != -1).to_numpy().all(axis=1)
        valid_test  = (adata_test.obs[args.target_outcomes] != -1).to_numpy().all(axis=1)

        logging.info(
            f"Removed {(~valid_train).sum()} patients with diagnosis before/on assessment or censored within horizon from train set"
        )
        logging.info(
            f"Removed {(~valid_val).sum()} patients with diagnosis before/on assessment or censored within horizon from val set"
        )
        logging.info(
            f"Removed {(~valid_test).sum()} patients with diagnosis before/on assessment or censored within horizon from test set"
        )

        if low_data_summary is not None:
            logging.info(
                "Nested low-data train subset after target filtering: final_valid_train=%d",
                int(valid_train.sum()),
            )

        adata_train = adata_train[valid_train]
        adata_val   = adata_val[valid_val]
        adata_test  = adata_test[valid_test]

        # Per-split positive/negative counts (logged + exported; sanity-checks that each split has positives)
        OmegaConf.set_struct(args, False)
        split_data_dict = {"train_total": None, "train_positives": None, "val_total": None, "val_positives": None, "test_total": None, "test_positives": None}
        for target in args.target_outcomes:
            for split_name, split_data in zip(['train', 'val', 'test'], [adata_train, adata_val, adata_test]):
                positives = (split_data.obs[target] == 1).sum()
                negatives = (split_data.obs[target] == 0).sum()
                pos_freq = float(positives / (positives + negatives))
                logging.info(f"{split_name.capitalize()} positive frequency for {target}: {pos_freq}")
                logging.info(f"{split_name.capitalize()} sample size for {target}: {(positives + negatives)}")
                logging.info(f"{split_name.capitalize()} positives samples: {positives}")
                split_data_dict[f"{split_name}_total"] = (positives + negatives)
                split_data_dict[f"{split_name}_positives"] = positives
                if positives == 0:
                    raise ValueError(f"No positive cases for target {target} in {split_name} set after splitting. Stopping.")
        # Store split_data_dict in csv file
        os.makedirs(os.path.join(args.data_root_path, "split_counts"), exist_ok=True)
        split_data_df = pd.DataFrame([split_data_dict])
        split_data_df.to_csv(os.path.join(args.data_root_path, f"split_counts/{args.model_name}_{args.task}_{args.target_outcomes[0]}.csv"), index=False)
        OmegaConf.set_struct(args, True)
    
    logging.info(
        f"Split checksums (incl. no-GP data): train={sum(train_obs_indices)}, val={sum(val_obs_indices)}, test={sum(test_obs_indices)}"
    )
    logging.info(
        f"Split sizes (incl. no-GP data): train={adata_train.shape[0]}, val={adata_val.shape[0]}, test={adata_test.shape[0]}"
    )

    return (adata_train, adata_val, adata_test), (
        train_obs_indices,
        val_obs_indices,
        test_obs_indices,
    )


class npAnnData(ad.AnnData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._convert_X_to_numpy()

    def _convert_X_to_numpy(self):
        if self.X is not None and not isinstance(self.X, np.ndarray):
            self.X = self.X.toarray()  # type: ignore # Convert sparse matrix to numpy array

    @property
    def X(self) -> np.ndarray:
        x = super().X
        if not isinstance(x, np.ndarray):
            raise ValueError("X must be a numpy array")
        return x

    @X.setter
    def X(self, value: np.ndarray):
        self._X = value

    def update_X_inplace(self, indices, values):
        copy = self.X.copy()
        copy[indices] = values
        if self.is_view:
            self._init_as_actual(self.copy())
        self.X = copy

    def __getitem__(self, index) -> npAnnData:
        result = super().__getitem__(index)
        if isinstance(result, ad.AnnData):
            return npAnnData(
                result.X, result.obs, result.var, result.uns, result.obsm, result.varm
            )
        return result

    def concat(*adatas: npAnnData) -> npAnnData:
        result = ad.concat(adatas, merge="same")  # type: ignore
        return npAnnData(
            result.X, result.obs, result.var, result.uns, result.obsm, result.varm
        )

    def copy(self) -> npAnnData:
        result = super().copy()
        return npAnnData(
            result.X, result.obs, result.var, result.uns, result.obsm, result.varm
        )


# Convert date strings to datetime objects vectorized
def parse_date_safe_vectorized(date_series):
    """Safely parse date series, return None for NaN/invalid dates"""
    date_series = date_series.astype(str)
    # Replace empty strings and 'nan' with actual NaN
    date_series = date_series.replace(["", "nan", "NaN"], pd.NA)
    # Convert to datetime, coerce errors to NaT (Not a Time)
    return pd.to_datetime(date_series, format="%Y-%m-%d", errors="coerce")


def create_targets_binary_classification(adata: npAnnData, args: Config) -> npAnnData:
    """
    Creates multilabel classification targets from date-based diagnosis data.

    Args:
        adata: The anndata object containing observation data with date fields
        args: Configuration object with multilabel settings

    Returns:
        adata_with_targets: Updated adata object with multilabel targets added to obs
        target_columns: List of target column names that were created
    """

    # Get assessment date field
    assessment_date_field = args.assessment_date_field
    horizon = args.years_to_diag
    minimum_cases = args.minimum_cases

    if assessment_date_field not in adata.obs.columns:
        raise ValueError(
            f"Assessment date field '{assessment_date_field}' not found in adata.obs"
        )

    diagnosis_fields = args.target_outcomes

    print(f"Found {len(diagnosis_fields)} potential diagnosis fields")

    # Parse dates
    assessment_dates = parse_date_safe_vectorized(adata.obs[assessment_date_field])
    death_dates = parse_date_safe_vectorized(adata.obs["Date_Death"])
    coma_dates = parse_date_safe_vectorized(adata.obs["Date_Coma"])

    # Parse all diagnosis dates at once
    diagnosis_data = adata.obs[diagnosis_fields].copy()
    for col in diagnosis_fields:
        diagnosis_data[col] = parse_date_safe_vectorized(diagnosis_data[col])

    # Create target columns
    target_columns = [f"target_{diag_field}" for diag_field in diagnosis_fields]

    # Initialize target matrix - all zeros by default
    target_matrix = np.zeros((len(adata), len(diagnosis_fields)), dtype=np.float32)

    # Convert to numpy arrays for vectorized operations
    assessment_dates_np = assessment_dates.astype("datetime64[ns]").values
    diagnosis_dates_np = diagnosis_data.astype("datetime64[ns]").values
    death_dates_np = death_dates.astype("datetime64[ns]").values
    coma_dates_np = coma_dates.astype("datetime64[ns]").values
    # Earliest UKB hospital inpatient censoring date (31 May 2022, for Wales cohort)
    global_censoring_date = np.datetime64(getattr(args, "time_cutoff", "2022-05-31"))

    # Calculate time differences
    diag_time_diffs = diagnosis_dates_np - assessment_dates_np[:, np.newaxis]
    death_time_diffs = death_dates_np - assessment_dates_np
    coma_time_diffs = coma_dates_np - assessment_dates_np
    global_censoring_time_diffs = global_censoring_date - assessment_dates_np
    # Convert timedelta to years
    diag_time_diffs_years = diag_time_diffs / pd.Timedelta(days=365.25)
    death_time_diffs_years = death_time_diffs / pd.Timedelta(days=365.25)
    coma_time_diffs_years = coma_time_diffs / pd.Timedelta(days=365.25)
    global_censoring_time_diffs_years = global_censoring_time_diffs / pd.Timedelta(days=365.25)

    # Flatten diag_time_diffs_years to not blow up mask dimensions, as we're only training on one target anyways
    diag_time_diffs_years = diag_time_diffs_years[:, 0]

    # Set horizon to the time until global censoring if it's shorter than the specified years_to_diag horizon
    effective_horizon = np.minimum(horizon, global_censoring_time_diffs_years)
    print(f"Setting effective_horizon to global censoring time for {(effective_horizon < horizon).sum()} subjects ({((effective_horizon < horizon).mean() * 100):.2f}%) because their assessment date is less than {horizon} years from the global censoring date")

    # Create masks for different conditions
    diag_within_timeframe_mask = (diag_time_diffs_years <= effective_horizon) & (
        diag_time_diffs_years > 0
    )
    print(f"Shape of diag_within_timeframe_mask: {diag_within_timeframe_mask.shape}")
    diag_outside_timeframe_mask = diag_time_diffs_years > effective_horizon
    diag_before_assessment_mask = diag_time_diffs_years <= 0
    censored_within_timeframe_mask = (
        (death_time_diffs_years <= effective_horizon) & (death_time_diffs_years > 0)
    ) | ((coma_time_diffs_years <= effective_horizon) & (coma_time_diffs_years > 0))
    censored_mask = (
        censored_within_timeframe_mask & ~diag_within_timeframe_mask.flatten()
    )

    # Set target values based on conditions
    # Positive labels: diagnosed within timeframe after assessment
    target_matrix[diag_within_timeframe_mask] = 1

    # Negative labels: diagnosed outside timeframe after assessment (already 0.0 by default)
    target_matrix[diag_outside_timeframe_mask] = 0

    # Invalid labels: diagnosed before or on assessment date
    target_matrix[diag_before_assessment_mask] = -1

    # Invalid labels: censored within timeframe without diagnosis
    target_matrix[censored_mask] = -1
    # If we don't use the gp adjusted loss, we also exclude those with no gp records (no penalization for cancer targets)
    if (
        len(args.target_outcomes) == 1
        and args.target_outcomes[0] in adata.uns["source_code_frac_primary_only"]
    ):
        gp_exclusion_mask = np.zeros(len(adata), dtype=bool)
        if args.include_gp == "positives_only":
            print("WARNING: Excluding those with no gp records and no diagnosis")
            # Exclude negatives where we don't have GP records
            gp_exclusion_mask = (
                adata.obs["Has_GP_Records"] == False
            ).values & ~diag_within_timeframe_mask.flatten()
        elif args.include_gp == "none":
            print("WARNING: Excluding all those with no gp records")
            # Exclude all without GP records
            gp_exclusion_mask = (adata.obs["Has_GP_Records"] == False).values
        elif args.include_gp == "all":
            print("WARNING: Not excluding those with no gp records")
        else:
            raise ValueError(f"Invalid value for include_gp: {args.include_gp}")
        target_matrix[gp_exclusion_mask] = -1

    # No diagnosis recorded cases remain 0.0 (already default)

    # Add target columns to adata.obs efficiently
    target_df = pd.DataFrame(
        target_matrix, columns=target_columns, index=adata.obs.index
    )
    adata.obs = pd.concat([adata.obs, target_df], axis=1)

    # Filter out targets with less than minimum_cases
    target_columns_filtered = [
        col for col in target_columns if (adata.obs[col] > 0).sum() >= minimum_cases
    ]
    print(
        f"Filtered out {len(target_columns) - len(target_columns_filtered)} targets with less than {minimum_cases} cases"
    )
    print(f"Keeping {len(target_columns_filtered)} targets")

    # Update num_targets in args
    args.num_targets = len(target_columns_filtered)
    args.target_outcomes = (
        target_columns_filtered  # Update target_outcomes to use new multilabel targets
    )

    return adata


def create_targets_survival(adata: npAnnData, args: Config) -> npAnnData:
    """
    Creates survival analysis targets from date-based diagnosis data.

    For each diagnosis field, creates two new columns:
    - event_{diag_field}: Boolean indicator (True=event occurred, False=censored)
    - time_{diag_field}: Time-to-event in years (from assessment to diagnosis or censoring)

    Args:
        adata: The anndata object containing observation data with date fields
        args: Configuration object with survival analysis settings

    The function adds event and time columns to adata.obs for survival analysis.
    Subjects with diagnosis before assessment are excluded (set to NaN).
    Subjects with no recorded diagnosis are censored at the follow-up time.
    """

    assessment_date_field = args.assessment_date_field
    diagnosis_fields = args.target_outcomes
    minimum_cases = args.minimum_cases

    print(f"Found {len(diagnosis_fields)} potential diagnosis fields")

    # Parse dates
    assessment_dates = parse_date_safe_vectorized(adata.obs[assessment_date_field])
    death_dates = parse_date_safe_vectorized(adata.obs["Date_Death"])
    coma_dates = parse_date_safe_vectorized(adata.obs["Date_Coma"])

    # Check if any assessment dates are invalid
    if assessment_dates.isna().any():
        invalid_indices = assessment_dates.isna()
        raise ValueError(
            f"Subjects at indices {invalid_indices[invalid_indices].index.tolist()} do not have valid assessment dates"
        )

    # Parse all diagnosis dates at once
    diagnosis_data = adata.obs[diagnosis_fields].copy()
    for col in diagnosis_fields:
        diagnosis_data[col] = parse_date_safe_vectorized(diagnosis_data[col])

    # Create target columns
    event_columns = [f"event_{diag_field}" for diag_field in diagnosis_fields]
    time_columns = [f"time_{diag_field}" for diag_field in diagnosis_fields]

    # Initialize target matrix - all zeros by default
    event_matrix = np.zeros((len(adata), len(diagnosis_fields)), dtype=np.float32)
    time_matrix = np.zeros((len(adata), len(diagnosis_fields)), dtype=np.float32)

    # Convert to numpy arrays for vectorized operations
    assessment_dates_np = assessment_dates.astype("datetime64[ns]").values
    diagnosis_dates_np = diagnosis_data.astype("datetime64[ns]").values
    death_dates_np = death_dates.astype("datetime64[ns]").values
    coma_dates_np = coma_dates.astype("datetime64[ns]").values

    # Create boolean masks for vectorized operations
    # Check where diagnosis dates are not NaN
    valid_diag_mask = ~pd.isna(diagnosis_dates_np)

    # Check where diagnosis came after assessment (vectorized comparison)
    after_assessment_mask = diagnosis_dates_np > assessment_dates_np[:, np.newaxis]
    before_assessment_mask = diagnosis_dates_np <= assessment_dates_np[:, np.newaxis]
    censored_mask_deaths = (death_dates_np > assessment_dates_np) # & ~valid_diag_mask
    censored_mask_comas = (coma_dates_np > assessment_dates_np) # & ~valid_diag_mask
    censored_mask = censored_mask_deaths | censored_mask_comas

    # Calculate time differences in days (vectorized)
    time_diffs_events = diagnosis_dates_np - assessment_dates_np[:, np.newaxis]
    # Negatives are treated as censored at earliest UKB hospital inpatient censoring date (31 May 2022, for Wales cohort)
    negative_censoring_date = np.datetime64(getattr(args, "time_cutoff", "2022-05-31"))
    time_diffs_negative_censoring = negative_censoring_date - assessment_dates_np[:, np.newaxis]
    # Censoring time is the same for all diagnosis fields (earliest of death/coma/global censoring per subject)
    censoring_dates = np.fmin(death_dates_np, coma_dates_np)
    censoring_dates = np.fmin(censoring_dates, negative_censoring_date)
    time_diffs_censoring = censoring_dates[:, np.newaxis] - assessment_dates_np[:, np.newaxis]
    # Convert timedelta to years
    time_diffs_events_years = time_diffs_events / pd.Timedelta(days=365.25)
    time_diffs_censoring_years = time_diffs_censoring / pd.Timedelta(days=365.25)
    time_diffs_negative_censoring_years = time_diffs_negative_censoring / pd.Timedelta(days=365.25)

    # How event_matrix and time_matrix are set:
    # Positives: event=1, time=years to diagnosis
    # Negatives: event=0, time=99 (censored at time point 99)
    # Diagnosed before/on assessment: event=-1, time=-1 (invalid, will be kicked)
    # Censored within timeframe without diagnosis: event=0, time=years to death or coma
    #
    # Everything with event == -1 will be kicked
    # Everything with event != 1 (so event == 0) will be fed into XGBoost with negative time (indicating censoring at that time point)

    # Set event values
    event_matrix[~valid_diag_mask] = 0
    event_matrix[censored_mask] = 0
    event_matrix[after_assessment_mask] = 1
    event_matrix[before_assessment_mask] = -1

    # Set time values
    time_matrix[~valid_diag_mask] = time_diffs_negative_censoring_years[~valid_diag_mask]
    time_matrix[censored_mask] = time_diffs_censoring_years[censored_mask]
    time_matrix[after_assessment_mask] = time_diffs_events_years[after_assessment_mask]
    time_matrix[before_assessment_mask] = -1

    # Mark samples with negative censoring time diff as invalid, so they are excluded downstream.
    # This is probably caused by using the earliest censoring date (Wales) for all subjects.
    neg_time_mask = (~valid_diag_mask) & (time_matrix < 0)
    if neg_time_mask.any():
        print(f"WARNING: Found {neg_time_mask.sum()} subjects with negative time to censoring. Marking as invalid.")
        print(f"Indices with negative time to censoring: {np.where(neg_time_mask)[0].tolist()}")
        event_matrix[neg_time_mask] = -1
        time_matrix[neg_time_mask] = -1.0

    if (
        not args.gp_adjusted_loss
        and len(args.target_outcomes) == 1
        and args.target_outcomes[0] in adata.uns["source_code_frac_primary_only"]
    ):
        
        gp_exclusion_mask = np.zeros(len(adata), dtype=bool)
        if args.include_gp == "positives_only":
            print("WARNING: Excluding those with no gp records and no diagnosis")
            # Exclude negatives where we don't have GP records
            gp_exclusion_mask = (
                adata.obs["Has_GP_Records"] == False
            ).values & (event_matrix.flatten() == 0)
        elif args.include_gp == "none":
            print("WARNING: Excluding all those with no gp records")
            # Exclude all without GP records
            gp_exclusion_mask = (adata.obs["Has_GP_Records"] == False).values
        elif args.include_gp == "all":
            print("WARNING: Not excluding those with no gp records")
        else:
            raise ValueError(f"Invalid value for include_gp: {args.include_gp}. Must be one of 'positives_only', 'none', or 'all'.")
        
        if gp_exclusion_mask.any():
            print(f"WARNING: Found {gp_exclusion_mask.sum()} subjects with no GP records. Marking as invalid.")

        event_matrix[gp_exclusion_mask] = -1
        time_matrix[gp_exclusion_mask] = -1

    target_df = pd.DataFrame(
        np.concatenate([event_matrix, time_matrix], axis=1),
        columns=(event_columns + time_columns),
        index=adata.obs.index,
    )
    adata.obs = pd.concat([adata.obs, target_df], axis=1)

    # Filter out targets with less than minimum_cases
    target_columns_filtered = [
        col for col in event_columns if (adata.obs[col] > 0).sum() >= minimum_cases
    ]
    print(
        f"Filtered out {len(event_columns) - len(target_columns_filtered)} targets with less than {minimum_cases} cases"
    )
    print(f"Keeping {len(target_columns_filtered)} targets")

    # Update num_targets in args
    args.num_targets = len(target_columns_filtered)
    args.target_outcomes = (
        target_columns_filtered  # Update target_outcomes to use new multilabel targets
    )

    return adata


# ---------------------------- ICD utilities ----------------------------
def icd_precompute(obs: pd.DataFrame, args: Config) -> tuple[
    int,
    list[str],
    dict[str, int],
    Optional[str],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """Precompute per-subject ICD indices and months-to-event.

    Returns:
        num_icd_codes, diag_cols, diag_col_to_row, assessment_col,
        per_subject_rows, per_subject_months
    """
    num_icd_codes = 0
    assessment_col: Optional[str] = None
    diag_cols: list[str] = []
    diag_col_to_row: dict[str, int] = {}
    per_subject_rows: list[torch.Tensor]
    per_subject_months: list[torch.Tensor]

    try:
        codes_to_row_path = os.path.join(
            args.data_root_path, args.icd_embeddings_codes_to_row_name  # type: ignore[arg-type]
        )
        with open(codes_to_row_path, "r") as f:
            icd_code_to_row: dict[str, int] = json.load(f)
        num_icd_codes = len(icd_code_to_row)

        # Identify obs diag columns like '130622-0.0' or ICD root codes
        obs_cols = list(obs.columns)
        META_COLUMNS = [
            "Date_Death",
            "Date_Coma",
            "Has_GP_Records",
            "53-0.0",
            "global_row",
        ]
        diag_cols = [c for c in obs_cols if c not in META_COLUMNS]

        # Keep only those that exist in mapping
        diag_col_to_row = {}
        kept_cols: list[str] = []
        for col in diag_cols:
            base_code = col.split("-")[0]
            if base_code in icd_code_to_row:
                kept_cols.append(col)
                diag_col_to_row[col] = icd_code_to_row[base_code]
        diag_cols = kept_cols

        assessment_col = getattr(args, "assessment_date_field", "53-0.0")

        if num_icd_codes > 0 and assessment_col in obs.columns and len(diag_cols) > 0:
            # String-based mask for before filtering
            df_diag_str = obs[diag_cols].astype("string")
            assess_str = obs[assessment_col].astype("string")
            before = (
                df_diag_str.le(assess_str, axis=0).fillna(False).to_numpy(dtype=bool)
            )

            col_to_icdrow = np.array(
                [diag_col_to_row[c] for c in diag_cols], dtype=np.int64
            )

            subj_i, diag_j = np.where(before)
            icd_rows = col_to_icdrow[diag_j]

            N = len(obs)
            counts = np.bincount(subj_i, minlength=N)
            order = np.argsort(subj_i, kind="stable")
            icd_rows_sorted = icd_rows[order]
            splits = np.cumsum(counts)[:-1]
            grouped_rows = np.split(icd_rows_sorted, splits) if N > 0 else []

            per_subject_rows = [
                (
                    torch.from_numpy(g.astype(np.int64))
                    if len(g) > 0
                    else torch.empty(0, dtype=torch.int64)
                )
                for g in grouped_rows
            ]

            if args.use_temporal_token:
                # Datetime for months delta
                df_diag_dt = (
                    obs[diag_cols]
                    .astype("object")
                    .apply(lambda x: pd.to_datetime(x, errors="coerce"))
                )
                assess_dt = obs[assessment_col].apply(
                    lambda x: pd.to_datetime(x, errors="coerce")
                )

                diag_days = df_diag_dt.to_numpy(dtype="datetime64[D]")
                assess_days = assess_dt.to_numpy(dtype="datetime64[D]")
                delta_days = (assess_days[:, None] - diag_days).astype("timedelta64[D]")
                months_all = delta_days.astype(np.float32) / np.float32(30.4375)
                months_vals = months_all[subj_i, diag_j]

                months_sorted = months_vals[order]
                grouped_months = np.split(months_sorted, splits) if N > 0 else []
                per_subject_months = [
                    (
                        torch.from_numpy(g.astype(np.float32))
                        if len(g) > 0
                        else torch.empty(0, dtype=torch.float32)
                    )
                    for g in grouped_months
                ]
            else:
                per_subject_months = [
                    torch.empty(0, dtype=torch.float32) for _ in range(len(obs))
                ]
        else:
            raise ValueError("ICD precompute: No valid ICD columns or assessment column found in obs")
            per_subject_rows = [
                torch.empty(0, dtype=torch.long) for _ in range(len(obs))
            ]
            per_subject_months = [
                torch.empty(0, dtype=torch.float32) for _ in range(len(obs))
            ]
    except Exception as e:
        print(f"Error loading/precomputing ICD embeddings (utils): {e}")
        num_icd_codes = 0
        diag_cols = []
        diag_col_to_row = {}
        assessment_col = None
        per_subject_rows = [torch.empty(0, dtype=torch.long) for _ in range(len(obs))]
        per_subject_months = [
            torch.empty(0, dtype=torch.float32) for _ in range(len(obs))
        ]

    return (
        num_icd_codes,
        diag_cols,
        diag_col_to_row,
        assessment_col,
        per_subject_rows,
        per_subject_months,
    )


def icd_build_representation(
    index: int,
    num_icd_codes: int,
    per_subject_rows: Optional[list[torch.Tensor]],
    per_subject_months: Optional[list[torch.Tensor]],
    use_temporal_token: bool,
) -> torch.Tensor:
    """Build the ICD representation for a subject at `index`.

    If `use_temporal_token` is False: returns a bool multi-hot of size [num_icd_codes].
    If `use_temporal_token` is True: returns a float32 months vector of size [num_icd_codes],
      where 0.0 means no event; if multiple entries per code exist, uses the minimum months.
    """
    if num_icd_codes <= 0:
        return (
            torch.zeros(0, dtype=torch.float32)
            if use_temporal_token
            else torch.zeros(0, dtype=torch.bool)
        )

    rows = (
        per_subject_rows[index]
        if per_subject_rows is not None
        else torch.empty(0, dtype=torch.long)
    )
    if not use_temporal_token:
        icd_multi_hot = torch.zeros(num_icd_codes, dtype=torch.bool)
        if rows.numel() > 0:
            icd_multi_hot[rows] = True
        return icd_multi_hot

    months = (
        per_subject_months[index]
        if per_subject_months is not None
        else torch.empty(0, dtype=torch.float32)
    )
    icd_months = torch.zeros(num_icd_codes, dtype=torch.float32)
    if rows.numel() > 0:
        # Minimum months per ICD row (closest event)
        for r, m in zip(rows.tolist(), months.tolist()):
            prev = icd_months[r].item()
            if prev == 0.0 or m < prev:
                icd_months[r] = m
    return icd_months


# ---------------------------- Medication utilities ----------------------------
def meds_precompute(obs: pd.DataFrame, args: Config) -> tuple[
    int,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[dict[str, int]],
    np.ndarray,
]:
    """Load medication CSR structure and mapping, and compute global row mapping.

    Returns:
        num_meds_codes: Number of unique medications with embeddings
        indptr: CSR row pointer array (subjects+1)
        indices: CSR column indices (medication columns)
        names: Array of medication names for columns (length = n_cols)
        name_to_row: Mapping from med name -> row in embeddings file
        global_rows: Array mapping dataset view index to global CSR row
    """
    num_meds_codes = 0
    indptr: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None
    names: Optional[np.ndarray] = None
    name_to_row: Optional[dict[str, int]] = None

    try:
        meds_oh_path = os.path.join(args.data_root_path, args.meds_OH_name)  # type: ignore[arg-type]
        oh_npz = np.load(meds_oh_path)
        indptr = oh_npz["indptr"]
        indices = oh_npz["indices"]

        meds_names_path = os.path.join(args.data_root_path, args.meds_names_name)  # type: ignore[arg-type]
        names = np.load(meds_names_path, allow_pickle=True)

        meds_codes_to_row_path = os.path.join(
            args.data_root_path, args.meds_embeddings_codes_to_row_name  # type: ignore[arg-type]
        )
        with open(meds_codes_to_row_path, "r") as f:
            name_to_row = json.load(f)

        # Determine number of unique embedding rows (not the number of names)
        if name_to_row is not None and len(name_to_row) > 0:
            try:
                rows_iter = (int(v) for v in name_to_row.values())
                max_row = max(rows_iter)
                num_meds_codes = int(max_row) + 1
            except Exception:
                # Fallback: load embeddings to infer row count if mapping values are not numeric
                meds_emb_path = os.path.join(args.data_root_path, args.meds_embeddings_name)  # type: ignore[arg-type]
                try:
                    num_meds_codes = int(np.load(meds_emb_path, mmap_mode="r").shape[0])
                except Exception:
                    num_meds_codes = 0
        else:
            num_meds_codes = 0
    except Exception as e:
        print(f"Error loading/precomputing medication embeddings (utils): {e}")
        indptr = None
        indices = None
        names = None
        name_to_row = None
        num_meds_codes = 0

    # Determine mapping to global row for each view row
    try:
        if "global_row" in obs.columns:
            global_rows = obs["global_row"].to_numpy(dtype=np.int64)
        else:
            global_rows = np.arange(len(obs), dtype=np.int64)
    except Exception:
        global_rows = np.arange(len(obs), dtype=np.int64)

    return (
        num_meds_codes,
        indptr,
        indices,
        names,
        name_to_row,
        global_rows,
    )


def meds_build_representation(
    index: int,
    num_meds_codes: int,
    indptr: Optional[np.ndarray],
    indices: Optional[np.ndarray],
    names: Optional[np.ndarray],
    name_to_row: Optional[dict[str, int]],
    global_rows: np.ndarray,
) -> torch.Tensor:
    """Build the medication multi-hot for a subject at `index`.

    Returns a bool multi-hot of size [num_meds_codes]. Empty if unavailable.
    """
    if (
        num_meds_codes <= 0
        or indptr is None
        or indices is None
        or names is None
        or name_to_row is None
    ):
        print("No medication data, returning zero medication")
        return torch.zeros(0, dtype=torch.bool)

    try:
        global_row = int(global_rows[index])
    except Exception:
        global_row = int(index)

    if global_row < 0 or global_row + 1 >= int(indptr.shape[0]):
        raise ValueError(f"Patient index {global_row} out of bounds for medication data with indptr length {len(indptr)}")
        return torch.zeros(num_meds_codes, dtype=torch.bool)

    start = int(indptr[global_row])
    end = int(indptr[global_row + 1])
    if end <= start:
        #print(f"Patient medication bounds invalid, returning zero medication")
        return torch.zeros(num_meds_codes, dtype=torch.bool)

    cols = indices[start:end]
    subject_med_names = names[cols]

    multi_hot = torch.zeros(num_meds_codes, dtype=torch.bool)
    for name in subject_med_names:
        row = name_to_row.get(str(name))
        if row is not None and 0 <= int(row) < num_meds_codes:
            multi_hot[int(row)] = True
    return multi_hot


# ---------------------------- ICD prognosis pretraining utilities ----------------------------
ICD10_CHAPTERS: list[tuple[str, str, str]] = [
    ("C00", "D49", "Neoplasms"),
    (
        "D50",
        "D89",
        "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism",
    ),
    ("E00", "E89", "Endocrine, nutritional and metabolic diseases"),
    ("F01", "F99", "Mental, Behavioral and Neurodevelopmental disorders"),
    ("G00", "G99", "Diseases of the nervous system"),
    ("H00", "H59", "Diseases of the eye and adnexa"),
    ("H60", "H95", "Diseases of the ear and mastoid process"),
    ("I00", "I99", "Diseases of the circulatory system"),
    ("J00", "J99", "Diseases of the respiratory system"),
    ("K00", "K95", "Diseases of the digestive system"),
    ("L00", "L99", "Diseases of the skin and subcutaneous tissue"),
    ("M00", "M99", "Diseases of the musculoskeletal system and connective tissue"),
    ("N00", "N99", "Diseases of the genitourinary system"),
]

ICD10_BLOCKS: list[tuple[str, str, str]] = [
    # C00–D49 — Neoplasms
    ("C00", "C14", "Malignant neoplasms of lip, oral cavity and pharynx"),
    ("C15", "C26", "Malignant neoplasms of digestive organs"),
    ("C30", "C39", "Malignant neoplasms of respiratory and intrathoracic organs"),
    ("C40", "C41", "Malignant neoplasms of bone and articular cartilage"),
    ("C43", "C44", "Melanoma and other malignant neoplasms of skin"),
    ("C45", "C49", "Malignant neoplasms of mesothelial and soft tissue"),
    ("C50", "C50", "Malignant neoplasm of breast"),
    ("C51", "C58", "Malignant neoplasms of female genital organs"),
    ("C60", "C63", "Malignant neoplasms of male genital organs"),
    ("C64", "C68", "Malignant neoplasms of urinary tract"),
    (
        "C69",
        "C72",
        "Malignant neoplasms of eye, brain and other parts of central nervous system",
    ),
    ("C73", "C75", "Malignant neoplasms of thyroid and other endocrine glands"),
    (
        "C76",
        "C80",
        "Malignant neoplasms of ill-defined, secondary and unspecified sites",
    ),
    (
        "C81",
        "C96",
        "Malignant neoplasms of lymphoid, haematopoietic and related tissue",
    ),
    ("D00", "D09", "In situ neoplasms"),
    ("D10", "D36", "Benign neoplasms"),
    ("D37", "D48", "Neoplasms of uncertain or unknown behaviour"),
    # D50–D89 — Blood and immune disorders
    ("D50", "D53", "Nutritional anaemias"),
    ("D55", "D59", "Haemolytic anaemias"),
    ("D60", "D64", "Aplastic and other anaemias"),
    ("D65", "D69", "Coagulation defects, purpura and other haemorrhagic conditions"),
    ("D70", "D77", "Other diseases of blood and blood-forming organs"),
    ("D80", "D89", "Certain disorders involving the immune mechanism"),
    # E00–E89 — Endocrine, nutritional and metabolic diseases
    ("E00", "E07", "Disorders of thyroid gland"),
    ("E10", "E14", "Diabetes mellitus"),
    (
        "E15",
        "E16",
        "Other disorders of glucose regulation and pancreatic internal secretion",
    ),
    ("E20", "E35", "Disorders of other endocrine glands"),
    ("E40", "E46", "Malnutrition"),
    ("E50", "E64", "Other nutritional deficiencies"),
    ("E65", "E68", "Obesity and other hyperalimentation"),
    ("E70", "E90", "Metabolic disorders"),
    # F01–F99 — Mental, behavioural and neurodevelopmental disorders
    ("F00", "F09", "Organic, including symptomatic, mental disorders"),
    (
        "F10",
        "F19",
        "Mental and behavioural disorders due to psychoactive substance use",
    ),
    ("F20", "F29", "Schizophrenia, schizotypal and delusional disorders"),
    ("F30", "F39", "Mood [affective] disorders"),
    ("F40", "F48", "Neurotic, stress-related and somatoform disorders"),
    ("F50", "F59", "Behavioural syndromes associated with physiological disturbances"),
    ("F60", "F69", "Disorders of adult personality and behaviour"),
    ("F70", "F79", "Intellectual disabilities"),
    ("F80", "F89", "Pervasive and specific developmental disorders"),
    (
        "F90",
        "F98",
        "Behavioural and emotional disorders with onset usually in childhood and adolescence",
    ),
    ("F99", "F99", "Unspecified mental disorder"),
    # G00–G99 — Nervous system
    ("G00", "G09", "Inflammatory diseases of the central nervous system"),
    ("G10", "G14", "Systemic atrophies primarily affecting the central nervous system"),
    ("G20", "G26", "Extrapyramidal and movement disorders"),
    ("G30", "G32", "Other degenerative diseases of the nervous system"),
    ("G35", "G37", "Demyelinating diseases of the central nervous system"),
    ("G40", "G47", "Episodic and paroxysmal disorders"),
    ("G50", "G59", "Nerve, nerve root and plexus disorders"),
    (
        "G60",
        "G65",
        "Polyneuropathies and other disorders of the peripheral nervous system",
    ),
    ("G70", "G73", "Diseases of myoneural junction and muscle"),
    ("G80", "G83", "Cerebral palsy and other paralytic syndromes"),
    ("G89", "G99", "Other disorders of the nervous system"),
    # H00–H59 — Eye and adnexa
    ("H00", "H05", "Disorders of eyelid, lacrimal system and orbit"),
    ("H10", "H13", "Disorders of conjunctiva"),
    ("H15", "H22", "Disorders of sclera, cornea, iris and ciliary body"),
    ("H25", "H28", "Disorders of lens"),
    ("H30", "H36", "Disorders of choroid and retina"),
    ("H40", "H42", "Glaucoma"),
    ("H43", "H45", "Disorders of vitreous body and globe"),
    ("H46", "H48", "Disorders of optic nerve and visual pathways"),
    (
        "H49",
        "H52",
        "Disorders of ocular muscles, binocular movement, accommodation and refraction",
    ),
    ("H53", "H54", "Visual disturbances and blindness"),
    ("H55", "H59", "Other disorders of eye and adnexa"),
    # H60–H95 — Ear and mastoid process
    ("H60", "H62", "Diseases of external ear"),
    ("H65", "H75", "Diseases of middle ear and mastoid"),
    ("H80", "H83", "Diseases of inner ear"),
    ("H90", "H95", "Other disorders of ear"),
    # I00–I99 — Circulatory system
    ("I00", "I02", "Acute rheumatic fever"),
    ("I05", "I09", "Chronic rheumatic heart diseases"),
    ("I10", "I15", "Hypertensive diseases"),
    ("I20", "I25", "Ischaemic heart diseases"),
    ("I26", "I28", "Pulmonary heart disease and diseases of pulmonary circulation"),
    ("I30", "I52", "Other forms of heart disease"),
    ("I60", "I69", "Cerebrovascular diseases"),
    ("I70", "I79", "Diseases of arteries, arterioles and capillaries"),
    ("I80", "I89", "Diseases of veins, lymphatic vessels and lymph nodes"),
    ("I95", "I99", "Other and unspecified disorders of circulatory system"),
    # J00–J99 — Respiratory system
    ("J00", "J06", "Acute upper respiratory infections"),
    ("J09", "J18", "Influenza and pneumonia"),
    ("J20", "J22", "Other acute lower respiratory infections"),
    ("J30", "J39", "Other diseases of upper respiratory tract"),
    ("J40", "J47", "Chronic lower respiratory diseases"),
    ("J60", "J70", "Lung diseases due to external agents"),
    ("J80", "J84", "Other respiratory diseases principally affecting the interstitium"),
    ("J85", "J86", "Suppurative and necrotic conditions of lower respiratory tract"),
    ("J90", "J94", "Other diseases of pleura"),
    ("J95", "J99", "Other diseases of the respiratory system"),
    # K00–K95 — Digestive system
    ("K00", "K14", "Diseases of oral cavity, salivary glands and jaws"),
    ("K20", "K31", "Diseases of oesophagus, stomach and duodenum"),
    ("K35", "K38", "Diseases of appendix"),
    ("K40", "K46", "Hernia"),
    ("K50", "K52", "Noninfective enteritis and colitis"),
    ("K55", "K64", "Other diseases of intestines"),
    ("K65", "K67", "Diseases of peritoneum"),
    ("K70", "K77", "Diseases of liver"),
    ("K80", "K87", "Disorders of gallbladder, biliary tract and pancreas"),
    ("K90", "K93", "Other diseases of the digestive system"),
    ("K94", "K95", "Other postprocedural digestive disorders"),
    # L00–L99 — Skin and subcutaneous tissue
    ("L00", "L08", "Infections of the skin and subcutaneous tissue"),
    ("L10", "L14", "Bullous disorders"),
    ("L20", "L30", "Dermatitis and eczema"),
    ("L40", "L45", "Papulosquamous disorders"),
    ("L50", "L54", "Urticaria and erythema"),
    ("L55", "L59", "Radiation-related disorders of the skin"),
    ("L60", "L75", "Disorders of skin appendages"),
    ("L80", "L99", "Other disorders of the skin and subcutaneous tissue"),
    # M00–M99 — Musculoskeletal system and connective tissue
    ("M00", "M25", "Arthropathies"),
    ("M30", "M36", "Systemic connective tissue disorders"),
    ("M40", "M54", "Dorsopathies"),
    ("M60", "M79", "Soft tissue disorders"),
    ("M80", "M94", "Osteopathies and chondropathies"),
    (
        "M95",
        "M99",
        "Other disorders of the musculoskeletal system and connective tissue",
    ),
    # N00–N99 — Genitourinary system
    ("N00", "N08", "Glomerular diseases"),
    ("N10", "N16", "Renal tubulo-interstitial diseases"),
    ("N17", "N19", "Renal failure"),
    ("N20", "N23", "Urolithiasis"),
    ("N25", "N29", "Other disorders of kidney and ureter"),
    ("N30", "N39", "Other diseases of urinary system"),
    ("N40", "N51", "Diseases of male genital organs"),
    ("N60", "N64", "Disorders of breast"),
    ("N70", "N77", "Inflammatory diseases of female pelvic organs"),
    ("N80", "N98", "Noninflammatory disorders of female genital tract"),
    ("N99", "N99", "Other disorders of genitourinary system"),
]


def _icd_root3(icd_code: str) -> str:
    """Return a 3-character alphanumeric root like 'A00', 'O9A'.

    Pads with '0' if fewer than 3 available characters after the leading letter.
    """
    code = re.sub(r"[^A-Za-z0-9]", "", icd_code.upper())
    if not code:
        return ""
    letter_match = re.search(r"[A-Z]", code)
    if not letter_match:
        return ""
    start = letter_match.start()
    tail = code[start:]
    root = tail[:3]
    if len(root) < 3:
        root = root + ("0" * (3 - len(root)))
    return root


def _load_field_id_to_icd_mapping(args: Config) -> dict[str, str]:
    """Load mapping from UKBB field IDs to ICD10 codes.

    Returns a dict mapping field_id (as string without visit suffix) -> icd10 code string.
    If the mapping file is unavailable or missing expected columns, returns empty dict.
    """
    mapping: dict[str, str] = {}
    try:
        mapping_path = os.path.join(args.data_root_path, args.ukbb_field_id_to_value_name)  # type: ignore[arg-type]
        df = pd.read_csv(mapping_path)

        field_col = "FieldID"
        icd_col = "Value"

        # Drop rows with missing values
        df = df[[field_col, icd_col]].dropna()

        for _, row in df.iterrows():
            field_id = str(row[field_col]).strip()
            icd_code = str(row[icd_col]).strip().upper()
            if field_id and icd_code:
                mapping[field_id] = icd_code
    except Exception as e:
        print(f"Warning: failed to load field_id->ICD mapping: {e}")

    return mapping


def _icd_code_to_group(icd_code: str, level: int) -> str:
    """Convert a full ICD10 code into a hierarchy group label.

    level=1 -> ICD-10 chapter range (e.g., 'A00-B99') per provided table
    level=2 -> ICD-10 block range (e.g., 'I60-I69') per ICD10_BLOCKS
    level>=3 -> letter + first two digits (e.g., 'I63') i.e. leave as is
    """
    if level >= 3:
        return icd_code
    code = re.sub(r"[^A-Za-z0-9]", "", icd_code.upper())
    if not code:
        return ""
    letter_match = re.search(r"[A-Z]", code)
    if not letter_match:
        return ""
    letter = letter_match.group(0)

    # Chapter-based grouping for level 1
    if level <= 1:
        root3 = _icd_root3(code)
        if root3:
            for start, end, _desc in ICD10_CHAPTERS:
                if start <= root3 <= end:
                    return f"{start}-{end}"
        # Fallback to letter if not matched
        return letter

    # Block-based grouping for level 2
    if level == 2:
        root3 = _icd_root3(code)
        if root3:
            try:
                from dataset.utils import ICD10_BLOCKS  # ensure access if moved
            except Exception:
                # ICD10_BLOCKS defined above; fallback if import fails
                pass
            for start, end, _desc in ICD10_BLOCKS:
                if start <= root3 <= end:
                    return f"{start}-{end}"
        # Fallback to letter+first digit grouping if not matched
        digits = re.findall(r"\d", code[code.index(letter) + 1 :])
        return f"{letter}{digits[0]}" if digits else letter

    # For deeper levels, use numeric digit-based groups
    digits = re.findall(r"\d", code[code.index(letter) + 1 :])
    if level == 2 or len(digits) == 1:
        return f"{letter}{digits[0]}" if digits else letter
    return f"{letter}{digits[0]}{digits[1]}"


def icd_prognosis_precompute(obs: pd.DataFrame, args: Config) -> np.ndarray:
    """Precompute ICD prognosis payloads as fractions of the assessment-to-cutoff interval."""
    assessment_col = getattr(args, "assessment_date_field", "53-0.0")
    if assessment_col not in obs.columns:
        raise ValueError(f"Assessment date field '{assessment_col}' not found in obs")
    assess_dt = parse_date_safe_vectorized(obs[assessment_col])
    assess_np = assess_dt.astype("datetime64[ns]").values

    cutoff_ts = pd.to_datetime(getattr(args, "time_cutoff", None))
    if pd.isna(cutoff_ts):
        raise ValueError("`args.time_cutoff` must be a valid date string.")
    assess_np = assess_dt.values.astype("datetime64[D]")
    cutoff_np = np.asarray(cutoff_ts.to_datetime64()).astype("datetime64[D]")
    timeframe_days = cutoff_np - assess_np  # shape [N], timedelta64[D]

    obs_cols = list(obs.columns)
    META_COLUMNS = ["Date_Death", "Date_Coma", "Has_GP_Records", "53-0.0", "global_row"]
    diag_cols = [c for c in obs_cols if c not in META_COLUMNS]

    target_ids = getattr(args, "repquery_target_field_ids", None)
    if target_ids:
        target_ids_set = set(target_ids)
        diag_cols = [c for c in diag_cols if c in target_ids_set]

    field_to_icd = _load_field_id_to_icd_mapping(args)
    col_to_icd: dict[str, str] = {}

    for c in diag_cols:
        col_to_icd[c] = field_to_icd[c]

    resolved_diag_cols = list(col_to_icd.keys())

    death_series = parse_date_safe_vectorized(obs["Date_Death"])
    coma_series = parse_date_safe_vectorized(obs["Date_Coma"])
    death_np = death_series.astype("datetime64[ns]").values
    coma_np = coma_series.astype("datetime64[ns]").values
    death_delta_days = (death_np - assess_np).astype("timedelta64[D]")
    coma_delta_days = (coma_np - assess_np).astype("timedelta64[D]")
    has_gp_records = obs["Has_GP_Records"]

    hierarchy_levels = parse_icd_hierarchy_levels(
        getattr(args, "icd_hierarchy_level", None)
    )
    num_subjects = len(obs)
    diag_date_cache: dict[str, pd.Series] = {}
    all_level_targets: list[np.ndarray] = []

    for level in hierarchy_levels:
        level_col_to_group: dict[str, str] = {}
        for c, icd_code in col_to_icd.items():
            group = _icd_code_to_group(icd_code, level)
            if group:
                level_col_to_group[c] = group

        valid_diag_cols = [c for c in resolved_diag_cols if c in level_col_to_group]

        classes = sorted({level_col_to_group[c] for c in valid_diag_cols})
        group_to_idx = {group: idx for idx, group in enumerate(classes)}

        if valid_diag_cols:
            diag_parsed = {
                c: diag_date_cache.setdefault(c, parse_date_safe_vectorized(obs[c]))
                for c in valid_diag_cols
            }
            diag_df = pd.DataFrame(diag_parsed, index=obs.index).apply(
                pd.to_datetime, errors="coerce"
            )
            diag_delta_df = diag_df.sub(assess_dt, axis=0)
            deltas_days = diag_delta_df.to_numpy(dtype="timedelta64[D]")
            col_group_indices = np.array(
                [group_to_idx[level_col_to_group[c]] for c in valid_diag_cols],
                dtype=np.int64,
            )
            class_masks = [col_group_indices == idx for idx in range(len(classes))]
        else:
            raise ValueError(
                f"No valid diagnosis columns found for hierarchy level {level}. "
            )
            deltas_days = np.empty((num_subjects, 0), dtype="timedelta64[D]")
            class_masks = []

        per_class_values: list[np.ndarray] = []
        if class_masks:
            for class_mask in class_masks:
                if not class_mask.any():
                    per_class_values.append(np.full(num_subjects, -1, dtype=np.float32))
                    continue
                class_deltas = deltas_days[:, class_mask]
                class_min_delta_days = (
                    pd.DataFrame(class_deltas)
                    .min(axis=1, skipna=True)
                    .to_numpy(dtype="timedelta64[D]")
                )

                within_timeframe_mask = class_min_delta_days <= timeframe_days # Diagnosis after assessment and within global timeframe
                censor_mask = (
                    (class_min_delta_days <= np.timedelta64(0, "D")) | # Diagnosis before or at assessment, or
                    # Masks for including GP records:
                    # - Negative with no GP record -> kick only negatives without GP records; keep positives regardless of GP record
                    # - No GP record at all -> kick all without GP records (positives and negatives)
                    ((~within_timeframe_mask & ~has_gp_records) if args.include_gp == "positives_only" else (np.zeros(num_subjects, dtype=bool) if args.include_gp == "all" else (~has_gp_records)))
                )
                target = np.zeros(num_subjects, dtype=np.float32) # Default to 0 (no diagnosis or diagnosis after timeframe)
                months = class_min_delta_days.astype(np.float32) / 30.4375
                target[within_timeframe_mask] = months[within_timeframe_mask] # Set months until diagnosis for diagnoses within timeframe
                target[censor_mask] = -1
                per_class_values.append(target)
        else:
            per_class_values = []

        fallback_target = np.full(num_subjects, -1, dtype=np.float32)

        embedding_name, mapping_name = get_icd_embedding_names(args, level)
        codes_to_row_path = os.path.join(
            args.data_root_path, mapping_name
        )  # type: ignore[arg-type]
        with open(codes_to_row_path, "r") as f:
            codes_to_row = json.load(f)

        rows_to_codes: dict[int, str] = {}
        for code, row_idx in codes_to_row.items():
            try:
                row_idx_int = int(row_idx)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Row index for code '{code}' in '{mapping_name}' must be an integer, got '{row_idx}'."
                ) from exc
            rows_to_codes[row_idx_int] = code

        ordered_labels = [rows_to_codes[idx] for idx in sorted(rows_to_codes)]

        level_targets: list[np.ndarray] = []
        for label in ordered_labels:
            target_vector: Optional[np.ndarray] = None
            if label in group_to_idx:
                idx = group_to_idx[label]
                if idx < len(per_class_values):
                    target_vector = per_class_values[idx]
            elif label in level_col_to_group:
                group_label = level_col_to_group[label]
                idx = group_to_idx.get(group_label)
                if idx is not None and idx < len(per_class_values):
                    target_vector = per_class_values[idx]

            if target_vector is None:
                level_targets.append(fallback_target.copy())
            else:
                level_targets.append(target_vector)

        if level_targets:
            all_level_targets.append(np.stack(level_targets, axis=0))
        else:
            all_level_targets.append(np.empty((0, num_subjects), dtype=np.float32))

    if not all_level_targets:
        return np.empty((0, num_subjects), dtype=np.float32)
    if len(all_level_targets) == 1:
        # Length == 1 means only one hierarchy level
        return all_level_targets[0]
    
    return np.concatenate(all_level_targets, axis=0)

def _concat_features(view: npAnnData, extra_block: np.ndarray, prefix: str, value_type: str) -> npAnnData:
    """
    Horizontally concatenate additional feature block to a view while preserving var metadata.

    - Keeps all existing columns in view.var (e.g., value_type, possible_preprocessing, etc.)
    - Appends rows for the new features with safe default metadata so XGBoost can still
      identify original categoricals correctly.
    """
    # Sanity check: same number of rows
    if view.n_obs != extra_block.shape[0]:
        raise ValueError(
            f"_concat_features: row mismatch: view.n_obs={view.n_obs}, "
            f"extra_block.shape[0]={extra_block.shape[0]}"
        )

    # Stack X and extra features horizontally
    new_X = np.hstack([view.X, extra_block]).astype(np.float32)

    # New feature names
    n_extra = int(extra_block.shape[1])
    extra_names = [f"{prefix}{i}" for i in range(n_extra)]

    # --- Preserve var metadata ---
    # Copy existing var (keep all metadata columns)
    base_var = view.var.copy()

    # Create metadata rows for appended features with defaults
    extra_var = pd.DataFrame(index=extra_names, columns=base_var.columns)

    # Fill in conservative defaults where possible:
    # - make sure appended features are treated as numeric by downstream logic
    if "value_type" in extra_var.columns:
        extra_var["value_type"] = value_type

    # - ICD/MEDS are one-hot/binary-ish; do NOT mark as "replace zero with NaN"
    if "possible_preprocessing" in extra_var.columns:
        extra_var["possible_preprocessing"] = ""

    # - if present, set categorical options to 0/1; this field isn't required for XGBoost,
    #   but some code may rely on it for other models.
    if "n_categorical_options" in extra_var.columns:
        extra_var["n_categorical_options"] = 1 if value_type == "Categorical single" else 1 # We use 1 for binary categoricals and numerics

    # Concatenate var tables
    new_var = pd.concat([base_var, extra_var], axis=0)

    # Ensure unique var names (AnnData expects unique var_names)
    if new_var.index.has_duplicates:
        # make duplicates unique by appending an incrementing suffix
        seen: dict[str, int] = {}
        new_index = []
        for name in new_var.index.tolist():
            if name not in seen:
                seen[name] = 0
                new_index.append(name)
            else:
                seen[name] += 1
                new_index.append(f"{name}_{seen[name]}")
        new_var.index = new_index

    return npAnnData(
        X=new_X,
        obs=view.obs.copy(),
        var=new_var,
        uns=view.uns,
    )
