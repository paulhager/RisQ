import re
from dataclasses import dataclass, field
from databind.core import ExtraKeys
from typing import List, Dict, Optional
from omegaconf import MISSING

from enum import Enum
from hydra.core.config_store import ConfigStore


class Task(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    SURVIVAL = "survival"
    PROGNOSIS = "prognosis"


class MAELossWeighting(Enum):
    EQUAL = "equal"
    INVERSE = "inverse"
    EMA = "EMA"
    LEARNED = "learned"


class Activation(Enum):
    RELU = "relu"
    GELU = "gelu"
    SELU = "selu"
    LEAKY_RELU = "leakyrelu"


class TemporalSamplingStrategy(Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    DIAG_CENTERED = "diag_centered"


def _parse_enum_value(enum_class, value: str):
    """
    Parse string value to enum, handling case-insensitive conversion.
    Uses existing parsing functions for known enums, otherwise attempts
    case-insensitive matching against enum values.
    """
    # Generic case-insensitive enum parsing
    value_lower = value.lower()
    for enum_member in enum_class.__members__.values():
        if enum_member.value.lower() == value_lower:
            return enum_member

    # If no match found, raise descriptive error
    valid_values = [member.value for member in enum_class.__members__.values()]
    raise ValueError(
        f"'{value}' is not a valid {enum_class.__name__}. Valid options: {valid_values}"
    )


def parse_list(s: str) -> list[str]:
    # Extract everything that looks like a token inside [ ... ]
    return re.findall(r"[^,\[\]\s]+", s)


@ExtraKeys()
@dataclass
class Config:
    # Note: Hydra will handle defaults from the YAML config files
    # This field is not used when Hydra loads the config from YAML

    # Core settings
    seed: int = 2024
    dataset_seed: int = 2024
    check_val_every_n_epoch: int = 1
    patience: int = 20
    low_data_split: Optional[float] = None
    # When enabled, low_data_split builds nested train subsets separately within
    # GP and no-GP subjects, preserving their proportion across fractions.
    nested_low_data_split_preserve_gp: bool = False
    balance_train: bool = False
    cross_validation_n_folds: int = 0
    cross_validation_current_fold: Optional[int] = None
    # CV mode when cross_validation_n_folds > 0:
    # - "fixed_test_inner": fixed test split + K-fold over remaining data (legacy behavior)
    # - "outer_oof": outer K-fold test split + one random inner train/val split per outer fold
    cv_mode: str = "fixed_test_inner"
    # Inner validation fraction used only for cv_mode="outer_oof". If None, derived from val_size/test_size.
    cv_inner_val_size: Optional[float] = None
    mini_batch_size: Optional[int] = None
    vars_to_remove_path: Optional[str] = None
    vars_to_keep_path: Optional[str] = None
    stratify: bool = False
    subsample_final_validation: bool = True
    use_pos_weight_bce: bool = False
    train_pos_weight_bce: Dict[str, float] = field(default_factory=dict)
    use_focal_loss: bool = True
    focal_loss_alpha: float = 0.85
    focal_loss_gamma: float = 2.0
    val_size: float = 0.15
    test_size: float = 0.15
    eval_full_dataset_no_split: bool = False
    debug: bool = False

    # Load pretrained weights
    pretrained_weights_path: Optional[str] = None
    average_pool: bool = False
    max_pool: bool = False

    # Testing and validation
    test: bool = True
    eval_full_dataset_no_split: bool = False
    input_size: Optional[int] = None
    num_targets: Optional[int] = None

    # FT
    shared_num_weights_tokenizer: bool = False

    # Schedule free optimizer
    schedulefree: bool = False

    # Wandb
    use_wandb: bool = True
    # W&B entity (team/user). Leave None to use the logged-in account's default
    # entity. Can also be set via the WANDB_ENTITY environment variable.
    wandb_entity: Optional[str] = None
    wandb_project_name: str = "UKBB_Foundation"
    wandb_mode: str = "online"
    wandb_id: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Lightning
    fast_dev_run: bool = False
    enable_progress_bar: bool = False

    # Hyperparameter sweep
    hp_sweep: bool = False
    launch_sweep: bool = False
    hp_sweep_n_trials: int = 20  # 100
    prune: bool = True  # False
    wandb_sweep_id: str = ""
    wandb_n_agents: int = 1
    sweep_method: str = "bayes"  # random
    classification_main_metric: str = "auc"  # "aucpr", "auc", "logloss"
    survival_main_metric: str = "cox-nloglik"  # "cox-nloglik"

    checkpoint_path: Optional[str] = None

    # Target
    target_outcomes: List[str] = field(default_factory=lambda: ["130708"])
    target_file_name: str = "all_selected_inputs_assessment0_eid_index_targets_preprocessed_gqnorm_filtered.h5ad"
    task: Task = Task.SURVIVAL

    # Disease Prognosis Classification
    years_to_diag: float = 15.0  # 5.0
    eval_horizons: List[float] = field(default_factory=lambda: [2.0, 5.0, 10.0, 15.0])
    assessment_date_field: str = "53-0.0"
    minimum_cases: int = 10
    include_gp: str = "positives_only"  # "none", "positives_only", "all"
    gp_adjusted_loss: bool = False
    no_gp_adjusted_loss_weight: float = 0.75

    # Paths
    data_root_path: str = MISSING
    wandb_dir_path: str = MISSING
    checkpoint_dir_path: str = MISSING
    num_workers: int = MISSING
    paths_name: str = MISSING

    # use icd and med embeddings for non-lightning models
    use_icd_baseline_onehot: bool = True
    use_meds_baseline_onehot: bool = True

    # ICD Embeddings
    icd_embeddings_name: str = "ukbb_icd10_codes_embeddings.npy"
    icd_embeddings_codes_to_row_name: str = "ukbb_icd10_codes_to_row.json"
    icd_codes_embeddings_name: str = "ukbb_icd10_codes_embeddings.npy"
    icd_codes_to_row_name: str = "ukbb_icd10_codes_to_row.json"
    icd_chapter_embeddings_name: str = "ukbb_icd10_chapter_embeddings.npy"
    icd_chapter_to_row_name: str = "ukbb_icd10_chapter_to_row.json"
    icd_block_range_embeddings_name: str = "ukbb_icd10_block_embeddings.npy"
    icd_block_range_to_row_name: str = "ukbb_icd10_block_to_row.json"
    use_temporal_token: bool = True
    disable_icd: bool = False

    # Medication Embeddings
    disable_meds: bool = False
    meds_embeddings_name: str = "ukbb_meds_embeddings.npy"
    meds_embeddings_codes_to_row_name: str = "ukbb_meds_names_to_row.json"
    meds_OH_name: str = "ukbb_medications_OH_csr.npz"
    meds_names_name: str = "ukbb_medications_names.npy"
    meds_mappings_name: str = "ukbb_medications_mapping.csv"

    # ICD Prognosis Pretraining Task
    ukbb_field_id_to_value_name: str = "ukbb_field_id_to_value.csv"
    icd_hierarchy_level: List[int] = field(
        default_factory=lambda: [3]
    )  # field(default_factory=lambda: [1, 2, 3])
    icd_prognosis_loss_weight: float = 1.0  # 0 is off, >0 is on
    save_logits: bool = False
    save_embeddings: bool = False
    # Save per-patient predictions for the test split of a standard (non-CV) run.
    save_test_predictions: bool = False
    test_predictions_output_dir: Optional[str] = None
    # Save fold-wise out-of-fold predictions (test split of each outer fold)
    save_outer_oof_predictions: bool = False
    oof_output_dir: Optional[str] = None

    # General
    model_name: str = MISSING
    mean_impute: bool = MISSING
    epochs: int = MISSING
    batch_size: int = MISSING

    # Architecture
    n_layers: int = MISSING
    hidden_dim: int = MISSING
    dropout: float = 0.1

    # Optimizer & Scheduler
    lr: float = MISSING
    weight_decay: float = 1e-5
    scheduler: str = "warmup"
    warmup_epochs: int = 10

    # RepQuery
    time_cutoff: str = "2022-05-31"
    generate_and_save_repquery_logits: bool = True
    temporal_sampling_strategy: TemporalSamplingStrategy = (
        TemporalSamplingStrategy.UNIFORM  # TemporalSamplingStrategy.DIAG_CENTERED
    )
    temporal_sampling_std_months: float = 12.0
    temporal_sampling_max_months: Optional[int] = None
    temporal_fixed_horizon_p: float = 0.0
    temporal_fixed_horizons_months: List[int] = field(default_factory=list)
    temporal_fixed_horizon_jitter_months: int = 0
    repquery_negative_sample_weight: float = 1.0
    feature_dropout_p: float = 0.0
    structured_feature_dropout_p: float = 0.0
    structured_feature_dropout_keep_path: Optional[str] = None
    modality_dropout_p: float = 0.6
    modality_dropout_groups_dir: Optional[str] = "configs/vars_to_keep/modality_groups"
    modality_dropout_protected_path: Optional[str] = (
        "configs/vars_to_keep/modality_groups/ehr_only.txt"
    )
    modality_dropout_icd_group_name: Optional[str] = "disease_history_only"
    modality_dropout_meds_group_name: Optional[str] = "medications_only"
    repquery_use_pos_weight: bool = False
    repquery_pos_weight_clip_max: float = 50.0
    repquery_pos_weight_eps: float = 1.0
    repquery_average_loss_over_targets: bool = False
    repquery_target_field_ids: Optional[List[str]] = None

    # XGBoost
    xgb_use_imputation_onehot: bool = False  # True
    calculate_permutation_importance: bool = False
    calculate_gain: bool = True #False
    max_depth: int = 6 #5
    eta: float = 0.3 #0.01
    gamma: float = 0.0
    scale_pos_weight: bool = False
    tree_method: str = "hist"  # "auto", gpu_hist
    max_delta_step: float = 0.0
    min_child_weight: float = 1.0 #5.0
    subsample: float = 1.0 #0.75
    colsample_bylevel: float = 1.0
    colsample_bytree: float = 1.0
    lambda_l2: float = 1.0
    alpha_l1: float = 0.0 #1.0
    save_xgb_model: bool = False
    save_sklearn_model: bool = False

    # Integrated gradients
    # Master switch
    run_integrated_gradients: bool = (
        False  # If True: run IG post-training (evaluation-time)
    )

    # It uses the already-loaded model and the eval_loader you pass.
    # ig_split is only used for naming output artifacts.
    ig_split: str = "val"

    # What outputs to explain
    ig_targets_mode: str = "all"  # Supported: "all" | "list"
    ig_targets_list: Optional[List[int]] = field(
        default_factory=lambda: [488, 502, 569, 572, 576, 577, 578, 97, 357, 131, 211]
    )  # Used only if ig_targets_mode="list"

    # Horizons / objective
    ig_horizons_years: List[int] = field(
        default_factory=lambda: [2]
    )  # List of horizons to explain
    ig_output_type: str = "logit"  # "logit" | "prob"

    # Which inputs to attribute
    ig_attr_x: bool = True       # Tabular x attribution
    ig_attr_icd: bool = False     # Token-embedding attribution for ICD tokens (requires RepQuery token hooks)
    ig_attr_meds: bool = False    # Token-embedding attribution for meds tokens (requires RepQuery token hooks)
    ig_x_attr_mode: str = "token"  # "token" | "raw" | "both"

    # Baselines (only these are used)
    ig_baseline_x: str = "mean_train"            # "mean_train" | "median_train" | "zero"
    ig_baseline_tokens: str = "zero"             # MUST be "zero" (only supported token baseline right now)
    ig_missing_aggregation: str = "zero"         # "zero" | "nan"; controls whether absent/missing inputs contribute 0 or are excluded from cohort-level means

    # Compute controls
    ig_steps: int = 20  # Number of IG steps
    ig_num_samples: int = (
        2000  # Max samples per (target,horizon), balanced case-control
    )
    ig_seed: int = 2024
    ig_batch_size: int = 128
    ig_token_batch_size: int = 128  # Optional smaller batch for token-space IG

    # Output controls
    ig_topk_features: int = 1370  # Top-K x features saved per (target,horizon)
    ig_topk_codes: int = 50  # Top-K code_ids saved per (target,horizon) for ICD/MEDS
    ig_out_dir: Optional[str] = (
        None  # Default: <checkpoint_dir_path>/integrated_gradients
    )
    ig_save_format: str = "csv"  # "parquet" | "csv"
    ig_save_patient_level: bool = False
    ig_patient_topk: int = 100
    ig_patient_id_col: str = "global_row"
    ig_stratify_by: List[str] = field(default_factory=list)  # e.g. ["sex"], ["age_bin"], or ["sex_age_bin"]
    ig_age_field: str = "21003-0.0"
    ig_sex_field: str = "31-0.0"
    ig_age_bin_edges: List[float] = field(
        default_factory=lambda: [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    )

    # TabPFN
    subsample_size: int = 5000

    # Transformer
    num_heads: int = 0
    attention_dropout: float = -1
    ffn_hidden_dim: Optional[int] = None
    ffn_hidden_dim_multiplier: int = 0
    ffn_dropout: float = -1
    norm_first: bool = True
    use_gate: bool = True
    activation: Activation = Activation.GELU
    use_positional_embedding: bool = False
    d_bottleneck: Optional[int] = None
    torch_transformer: bool = False

    # Transformer / RepQuery shared fields
    loss_weighting: MAELossWeighting = MAELossWeighting.INVERSE
    use_cls_token: bool = True
    use_cls_token_in_decoder: bool = True
    cls_orthogonality_loss: bool = False
    cls_token_only_decoding: bool = True
    n_cls_tokens: int = 1
    use_projection: bool = False
    reconstruct_all: bool = False
    correlation_loss: bool = False
    correlation_loss_alpha: float = 0.5
    decoder_n_layers: int = 0
    decoder_hidden_dim: int = 0
    decoder_num_heads: int = 0
    decoder_attention_dropout: float = -1
    decoder_ffn_hidden_dim: Optional[int] = None
    decoder_ffn_hidden_dim_multiplier: int = 0
    decoder_ffn_dropout: float = -1
    n_subjects: Optional[int] = None
    max_num_features: Optional[int] = None



# Register the configuration schemas with Hydra's ConfigStore
# This enables native enum parsing and type validation
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
