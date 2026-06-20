import os
import gc
import json
from os.path import join
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import OmegaConf
from pathlib import Path
from typing import Any, Union, Optional, cast
from multiprocessing import Queue as mp_queue
import logging
import glob

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from config_types import Task, Config, _parse_enum_value, MAELossWeighting, parse_list
from omegaconf.listconfig import ListConfig
from enum import Enum
from models.utils import model_selector
from dataset.utils import (
    npAnnData,
    create_targets_survival,
    create_targets_binary_classification,
    create_MultiDisease_train_val_test_split,
)
from dataset.RepQueryDataset import RepQueryDataset
from infra.utils import (
    retrieve_config,
    write_sweep_config_yaml,
    _resolve_wandb_id,
)
from dataset.utils import (
    icd_precompute,
    icd_build_representation,
    _concat_features,
    meds_precompute,
    meds_build_representation,
)
import subprocess

from models.integrated_gradients import run_repquery_integrated_gradients

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def initialize_dataloader(
    args: Config,
    view: npAnnData,
    shuffle: bool,
) -> DataLoader:
    match args.model_name:
        case "RepQuery":
            dataset_class = RepQueryDataset
        case _:
            raise ValueError(f"Model {args.model_name} not supported")
    dataset = dataset_class(view, args)

    return DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.mini_batch_size if args.mini_batch_size else args.batch_size,
        pin_memory=True,
        shuffle=shuffle,
        persistent_workers=False if args.hp_sweep or args.num_workers == 0 else True,
        worker_init_fn=lambda worker_id: np.random.seed(args.seed + worker_id),
    )


def setup_callbacks(
    args: Config,
    wandb_logger: Union[WandbLogger, bool],
) -> list[pl.Callback]:
    callbacks = []

    save_checkpoints = args.wandb_project_name != "Test" and args.epochs > 0
    if save_checkpoints:
        if wandb_logger and isinstance(wandb_logger, WandbLogger):
            # Only add name if not already present
            if not args.checkpoint_dir_path.endswith(wandb_logger.experiment.name):
                args.checkpoint_dir_path = join(
                    args.checkpoint_dir_path, wandb_logger.experiment.name
                )
        else:
            raise ValueError("Wandb logger must be initialized if saving checkpoints")
        callbacks.append(
            ModelCheckpoint(
                filename="checkpoint_last_epoch_{epoch:02d}",
                dirpath=args.checkpoint_dir_path,
                save_on_train_epoch_end=True,
                auto_insert_metric_name=False,
                save_weights_only=False,
            )
        )
        if args.model_name == "RepQuery":
            callbacks.append(
                ModelCheckpoint(
                    filename="checkpoint_best_val_epoch_loss_{epoch:02d}",
                    monitor="RepQuery.val.loss",
                    dirpath=args.checkpoint_dir_path,
                    save_on_train_epoch_end=True,
                    auto_insert_metric_name=False,
                    save_weights_only=False,
                )
            )

    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    if args.model_name == "RepQuery":
        callbacks.append(
            EarlyStopping(
                monitor="RepQuery.val.loss",
                patience=args.patience,
                mode="min",
                check_on_train_epoch_end=False,
            )
        )
    return callbacks


tried_configs = set()

os.environ["WANDB__SERVICE_WAIT"] = "600"
os.environ["WANDB_MAX_RETRIES"] = "5"
os.environ["WANDB_RETRY_DELAY"] = "10"


def run(args: Config, pipeline_queue: Optional[mp_queue] = None):
    debug = args.debug

    resume_training = False
    checkpoint_path = None
    checkpoint_vars_to_keep_path = None
    eval_override_vars_to_keep_path = args.vars_to_keep_path
    eval_override_fields = {}
    if args.checkpoint_path:
        epochs = args.epochs
        checkpoint_path = args.checkpoint_path
        checkpoint_map_location = None if torch.cuda.is_available() else torch.device("cpu")
        ckpt = torch.load(
            checkpoint_path,
            weights_only=False,
            map_location=checkpoint_map_location,
        )
        ckpt_args = OmegaConf.create(ckpt["hyper_parameters"])  # type: ignore
        if "args" in ckpt_args:
            ckpt_args = ckpt_args["args"]
        checkpoint_vars_to_keep_path = getattr(ckpt_args, "vars_to_keep_path", None)
        cli_overrides = args
        if epochs > 0:
            # Resume training: CLI overrides win over checkpoint.
            args = OmegaConf.merge(ckpt_args, cli_overrides)
            resume_training = True
        else:
            # Eval-only: checkpoint hyperparameters win over Hydra defaults (e.g. num_heads=0).
            # Do not re-merge cli_overrides afterward — that restores zero sentinels.
            eval_override_fields = {
                "vars_to_keep_path": getattr(cli_overrides, "vars_to_keep_path", None),
                "disable_icd": getattr(cli_overrides, "disable_icd", None),
                "disable_meds": getattr(cli_overrides, "disable_meds", None),
                "checkpoint_dir_path": getattr(cli_overrides, "checkpoint_dir_path", None),
                "wandb_project_name": getattr(cli_overrides, "wandb_project_name", None),
                # Allow eval-time remapping of ICD inputs/targets
                "target_file_name": getattr(cli_overrides, "target_file_name", None),
                "ukbb_field_id_to_value_name": getattr(
                    cli_overrides, "ukbb_field_id_to_value_name", None
                ),
                "icd_embeddings_name": getattr(cli_overrides, "icd_embeddings_name", None),
                "icd_embeddings_codes_to_row_name": getattr(
                    cli_overrides, "icd_embeddings_codes_to_row_name", None
                ),
                "icd_codes_embeddings_name": getattr(
                    cli_overrides, "icd_codes_embeddings_name", None
                ),
                "icd_codes_to_row_name": getattr(cli_overrides, "icd_codes_to_row_name", None),
                "modality_dropout_groups_dir": getattr(cli_overrides, "modality_dropout_groups_dir", None),
                "modality_dropout_protected_path": getattr(cli_overrides, "modality_dropout_protected_path", None),
                "modality_dropout_icd_group_name": getattr(cli_overrides, "modality_dropout_icd_group_name", None),
                "modality_dropout_meds_group_name": getattr(cli_overrides, "modality_dropout_meds_group_name", None),
                "structured_feature_dropout_keep_path": getattr(cli_overrides, "structured_feature_dropout_keep_path", None),
                "include_gp": getattr(cli_overrides, "include_gp", None),
            }
            # Preserve eval-time integrated gradients settings from the CLI.
            # These should not be inherited from the training checkpoint.
            for field_name in cli_overrides.keys():
                field_name = str(field_name)
                if field_name == "run_integrated_gradients" or field_name.startswith("ig_"):
                    eval_override_fields[field_name] = getattr(cli_overrides, field_name, None)
            args = OmegaConf.merge(cli_overrides, ckpt_args)
            args.epochs = 0
            args.checkpoint_path = checkpoint_path
            for field_name, field_value in eval_override_fields.items():
                if field_value is not None:
                    setattr(args, field_name, field_value)

    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = (
            "offline" if args.wandb_mode == "offline" else "online"
        )

    # Don't resume the original wandb run for eval-only runs (epochs=0);
    # create a fresh run so eval results don't overwrite the training run.
    resolved_wandb_id = _resolve_wandb_id(args, checkpoint_path) if args.epochs > 0 else None

    run = wandb.init(
        entity=(args.wandb_entity or os.environ.get("WANDB_ENTITY") or None),
        project=args.wandb_project_name,
        config=OmegaConf.to_container(args, resolve=True),
        resume="allow",
        mode=os.environ["WANDB_MODE"],
        id=resolved_wandb_id,
        settings=wandb.Settings(start_method="thread", _disable_stats=True),
    )

    # Overtake sweep params
    if args.hp_sweep:
        config_str = json.dumps(dict(wandb.config), sort_keys=True)

        if config_str in tried_configs:
            run.finish()
            return

        tried_configs.add(config_str)

        # Update args with sweep config
        for arg in wandb.config.keys():
            setattr(args, arg, wandb.config[arg])
        if args.model_name == "RepQuery":
            logging.warning(
                f"Overriding decoder_hidden_dim to match hidden_dim = {args.hidden_dim} for RepQuery"
            )
            args.decoder_hidden_dim = args.hidden_dim

    if args.model_name == "RepQuery":
        hidden_dim = getattr(args, "hidden_dim", None)
        num_heads = getattr(args, "num_heads", None)
        if hidden_dim == 256 and num_heads == 16:
            logging.warning(
                "RepQuery hidden_dim=256, num_heads=16: watch for CUDA errors "
            )
    print(OmegaConf.to_yaml(args, sort_keys=True))
    pl.seed_everything(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Load hyperparameters from checkpoint
    if args.pretrained_weights_path:
        checkpoint_hp = torch.load(
            args.pretrained_weights_path,
            weights_only=False,
            map_location=lambda storage, loc: storage,
        )["hyper_parameters"]
        if "args" in checkpoint_hp:
            checkpoint_hp = checkpoint_hp["args"]

        # Set model relevant parameters so weight loading works
        args.n_layers = checkpoint_hp.get("n_layers", 1)
        args.hidden_dim = checkpoint_hp.get("hidden_dim", 512)

        model_name = checkpoint_hp.get("model_name", "")
        args.model_name = model_name
        if model_name == "RepQuery":
            defaults = {
                # encoder
                "num_heads": 8,
                "ffn_hidden_dim": 2048,
                "ffn_hidden_dim_multiplier": 4,
                "attention_dropout": 0.0,
                "ffn_dropout": 0.0,
                "use_positional_embedding": False,
                "norm_first": True,
                "use_gate": True,
                "use_cls_token": True,
                "use_cls_token_in_decoder": True,
                "n_cls_tokens": 1,
                "use_projection": False,
                # decoder
                "decoder_n_layers": 1,
                "decoder_hidden_dim": 512,
                "decoder_num_heads": 8,
                "decoder_attention_dropout": 0.0,
                "decoder_ffn_hidden_dim": 2048,
                "decoder_ffn_hidden_dim_multiplier": 4,
                "decoder_ffn_dropout": 0.0,
                # misc
                "torch_transformer": False,
                "reconstruct_all": False,
                "cls_token_only_decoding": False,
                "cls_orthogonality_loss": False,
                "correlation_loss": False,
                "correlation_loss_alpha": 1.0,
                "loss_weighting": _parse_enum_value(MAELossWeighting, "inverse"),
                "checkpoint_dir_path": checkpoint_hp.get("checkpoint_dir_path", ""),
            }

            for key, default in defaults.items():
                setattr(args, key, checkpoint_hp.get(key, default))
        args.shared_num_weights_tokenizer = checkpoint_hp.get(
            "shared_num_weights_tokenizer", False
        )

        # Set overrides
        overrides = HydraConfig.get().overrides.task
        if overrides:
            for item in overrides:
                key, value = item.split("=")
                if hasattr(args, key):
                    # Get the current attribute's type and convert the value
                    current_value = getattr(args, key)
                    if current_value is not None:
                        target_type = type(current_value)
                        # Handle boolean conversion specially
                        if target_type == bool:
                            converted_value = value.lower() in ("true")
                        # Handle enum conversion using proper parsing functions
                        elif hasattr(target_type, "__bases__") and any(
                            issubclass(base, Enum) for base in target_type.__bases__
                        ):
                            converted_value = _parse_enum_value(target_type, value)
                        elif target_type == ListConfig:
                            converted_value = parse_list(value)
                        else:
                            converted_value = target_type(value)
                    else:
                        # If current value is None, try to infer from the string
                        if value.lower() in ("true", "false"):
                            converted_value = value.lower() == "true"
                        elif value.isdigit() or (
                            value.startswith("-") and value[1:].isdigit()
                        ):
                            converted_value = int(value)
                        elif value.replace(".", "", 1).isdigit():
                            converted_value = float(value)
                        else:
                            converted_value = value
                    setattr(args, key, converted_value)

    wandb_logger = WandbLogger(
        experiment=wandb.run, offline=(os.environ["WANDB_MODE"] == "offline")
    )

    args.wandb_run_name = wandb.run.name if wandb.run else "unknown"

    # Load targets from file if necessary
    if isinstance(args.target_outcomes, str) and args.target_outcomes.endswith(".txt"):
        with open(args.target_outcomes, "r") as f:
            args.target_outcomes = f.read().splitlines()

    # Load data
    in_path = Path(args.data_root_path) / args.target_file_name
    adata = npAnnData(ad.read_h5ad(in_path))
    # Preserve a stable mapping from current row to the original dataset row for external stores (e.g., meds CSR)
    adata.obs["global_row"] = np.arange(len(adata), dtype=np.int64)

    # If target is part of features we must remove it
    vars_to_remove = []
    for target in args.target_outcomes:
        if target in adata.var.index:
            adata = adata[:, adata.var.index != target]
            vars_to_remove.append(target)

    # For checkpoint-based runs, rebuild the model in the original checkpoint feature space
    # before applying any eval-time masking (e.g. EHR-only deployment evaluation).
    if checkpoint_path is not None and checkpoint_vars_to_keep_path:
        resolved_checkpoint_vars_to_keep_path = Path(checkpoint_vars_to_keep_path)
        if not resolved_checkpoint_vars_to_keep_path.exists():
            fallback_vars_to_keep_path = (
                Path(eval_override_vars_to_keep_path)
                if eval_override_vars_to_keep_path is not None
                else None
            )
            if fallback_vars_to_keep_path is not None and fallback_vars_to_keep_path.exists():
                logging.warning(
                    "checkpoint vars_to_keep_path %s not found; falling back to eval override %s",
                    checkpoint_vars_to_keep_path,
                    fallback_vars_to_keep_path,
                )
                resolved_checkpoint_vars_to_keep_path = fallback_vars_to_keep_path
            else:
                raise FileNotFoundError(
                    f"Checkpoint vars_to_keep_path does not exist: {checkpoint_vars_to_keep_path}"
                )

        checkpoint_vars_to_keep = set(
            resolved_checkpoint_vars_to_keep_path.read_text().splitlines()
        )
        checkpoint_keep_mask = adata.var.index.isin(checkpoint_vars_to_keep)
        missing = checkpoint_vars_to_keep - set(adata.var.index[checkpoint_keep_mask])
        if missing:
            logging.warning(
                f"checkpoint vars_to_keep_path: {missing} not found in adata.var"
            )
        adata = adata[:, checkpoint_keep_mask]
        logging.info(
            f"checkpoint vars_to_keep_path: restored training feature space with {adata.n_vars} variables"
        )

        # Restore the user's current override for eval-time masking after preserving the
        # checkpoint feature space.
        args.vars_to_keep_path = eval_override_vars_to_keep_path

    # Optionally restrict to a subset of variables (e.g. Delphi vars for fair comparison)
    _vars_to_keep_names = None
    if args.vars_to_keep_path is not None:
        vars_to_keep = set(Path(args.vars_to_keep_path).read_text().splitlines())
        keep_mask = adata.var.index.isin(vars_to_keep)
        missing = vars_to_keep - set(adata.var.index[keep_mask])
        if missing:
            logging.warning(f"vars_to_keep_path: {missing} not found in adata.var")
        if checkpoint_path is not None or args.pretrained_weights_path is not None:
            # Evaluating a pretrained model: defer NaN'ing until after split to avoid removing
            # all-NaN columns in create_MultiDisease_train_val_test_split, which would cause a
            # size mismatch when loading the checkpoint.
            _vars_to_keep_names = vars_to_keep
            logging.info(
                f"vars_to_keep_path: deferring NaN of {(~keep_mask).sum()} vars until after split"
            )
        else:
            # Training from scratch: filter adata to kept vars only (reduces input_size)
            adata = adata[:, keep_mask]
            logging.info(f"vars_to_keep_path: filtered to {adata.n_vars} variables")

    valid_features_mask = ~np.isnan(adata.X)
    args.n_subjects = adata.shape[0]
    args.max_num_features = int(valid_features_mask.sum(axis=1).max())

    val_view_target = None
    if args.model_name != "RepQuery":
        if args.task == Task.SURVIVAL:
            adata = create_targets_survival(adata, args)
        elif args.task == Task.CLASSIFICATION:
            adata = create_targets_binary_classification(adata, args)
        elif args.task == Task.REGRESSION:
            raise ValueError("Regression not supported")

    train_view_target = adata
    train_indices = val_indices = test_indices = None
    use_full_dataset_no_split = (
        checkpoint_path is not None
        and args.epochs == 0
        and getattr(args, "eval_full_dataset_no_split", False)
    )
    if use_full_dataset_no_split:
        logging.info(
            "eval_full_dataset_no_split=True: skipping train/val/test split and using the full dataset for test-only eval export"
        )
        train_view_target = adata
        val_view_target = None
        test_view_target = adata
    else:
        (
            (train_view_target, val_view_target, test_view_target),
            (
                train_indices,
                val_indices,
                test_indices,
            ),
        ) = create_MultiDisease_train_val_test_split(adata, args)

    if _vars_to_keep_names is not None:
        nan_mask = ~train_view_target.var.index.isin(_vars_to_keep_names)
        views_to_mask = [train_view_target, val_view_target, test_view_target]
        if use_full_dataset_no_split:
            views_to_mask = [train_view_target, test_view_target]
        views_to_mask = [view for view in views_to_mask if view is not None]
        for view in views_to_mask:
            view.X = np.array(view.X)  # materialize ArrayView → plain ndarray
            view.X[:, nan_mask] = np.nan
        logging.info(
            f"vars_to_keep_path: NaN'd {nan_mask.sum()} of {len(nan_mask)} vars"
            + (" in full-dataset eval mode" if use_full_dataset_no_split else " in all splits")
        )

    # Lightning models
    if args.model_name == "RepQuery":
        args.input_size = adata.n_vars

        # limit amount of train data for quicker epochs
        if debug:
            train_view_target = train_view_target[: args.batch_size * 2]
            train_indices = train_indices[: args.batch_size * 2]
            val_view_target = train_view_target[: args.batch_size * 2]
            val_indices = train_indices[: args.batch_size * 2]

        train_dataloader = initialize_dataloader(
            args,
            train_view_target,
            shuffle=True,
        )
        val_dataloader = (
            initialize_dataloader(
                args,
                val_view_target,
                shuffle=False,
            )
            if val_view_target
            else None
        )

        callbacks = setup_callbacks(args, wandb_logger)
        # dimensions have to match
        if args.model_name == "RepQuery":
            logging.warning(
                f"Overriding decoder_hidden_dim to match hidden_dim = {args.hidden_dim} for RepQuery"
            )
            args.decoder_hidden_dim = args.hidden_dim
        model = model_selector(args)(args, train_view_target)

        print("batch_size", args.batch_size)
        print("mini_batch_size", args.mini_batch_size)
        print(
            "accumulate_grad_batches",
            (args.batch_size // args.mini_batch_size) if args.mini_batch_size else 1,
        )
        use_gpu = torch.cuda.is_available()

        trainer = Trainer(
            accelerator="gpu" if use_gpu else "cpu",
            callbacks=callbacks,
            logger=wandb_logger,
            max_epochs=args.epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            fast_dev_run=args.fast_dev_run,
            enable_progress_bar=args.enable_progress_bar,
            num_sanity_val_steps=0,
            deterministic=True,
            accumulate_grad_batches=(
                args.batch_size // args.mini_batch_size if args.mini_batch_size else 1
            ),
        )
        trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
            ckpt_path=checkpoint_path if resume_training else None,
        )

        if (
            args.model_name == "RepQuery"
            and (
                getattr(args, "generate_and_save_repquery_logits", False)
                or getattr(args, "run_integrated_gradients", False)
            )
        ):
            # For eval-only runs, use the provided checkpoint directly;
            # otherwise search for the best checkpoint saved during training.
            if checkpoint_path is not None:
                best_checkpoint = checkpoint_path
            else:
                checkpoint_pattern = os.path.join(
                    args.checkpoint_dir_path,
                    "checkpoint_best_val_epoch_loss_*.ckpt",
                )
                checkpoint_files = glob.glob(checkpoint_pattern)
                if checkpoint_files:
                    best_checkpoint = checkpoint_files[0]
                else:
                    raise FileNotFoundError(
                        f"No checkpoint found matching pattern: {checkpoint_pattern}"
                    )

            checkpoint = torch.load(
                best_checkpoint, map_location="cpu", weights_only=False
            )
            logging.info(f"Using best checkpoint: {best_checkpoint}")
            state_dict = checkpoint.get("state_dict")
            if state_dict is None:
                raise KeyError(
                    "Checkpoint does not contain 'state_dict' for RepQuery evaluation"
                )

            if args.model_name == "RepQuery":
                logging.warning(
                    f"Overriding decoder_hidden_dim to match hidden_dim = {args.hidden_dim} for RepQuery"
                )
                args.decoder_hidden_dim = args.hidden_dim
            repquery_model = model_selector(args)(args, train_view_target)
            # Dont need to load these embeddings again.
            state_dict.pop("icd_prognosis_embeddings", None)
            state_dict.pop("meds_embeddings", None)
            state_dict.pop("icd_embeddings", None)
            response = repquery_model.load_state_dict(state_dict, strict=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            repquery_model = repquery_model.to(device)
            repquery_model.eval()

            val_dataloader = None
            if val_view_target is not None:
                val_dataloader = initialize_dataloader(
                    args,
                    val_view_target,
                    shuffle=False,
                )

                if getattr(args, "generate_and_save_repquery_logits", False):
                    # repquery_model.generate_and_save_1year_bin_logits(train_dataloader, "train")
                    repquery_model.generate_and_save_1year_bin_logits(
                        val_dataloader,
                        "val",
                        wandb_logger,
                        save_logits=True,  # getattr(args, "save_logits", False),
                        save_embeddings=True,  # getattr(args, "save_embeddings", True),
                    )

            test_dataloader = initialize_dataloader(
                args,
                test_view_target,
                shuffle=False,
            )

            if getattr(args, "generate_and_save_repquery_logits", False):
                repquery_model.generate_and_save_1year_bin_logits(
                    test_dataloader,
                    "test",
                    wandb_logger,
                    save_logits=True,  # getattr(args, "save_logits", False),
                    save_embeddings=True,  # getattr(args, "save_embeddings", True),
                )

            # -----------------------------
            # Integrated Gradients (optional)
            # -----------------------------
            if getattr(args, "run_integrated_gradients", False):
                ig_split = str(getattr(args, "ig_split", "val")).lower()
                if ig_split == "val":
                    ig_eval_loader = val_dataloader
                elif ig_split == "test":
                    ig_eval_loader = test_dataloader
                else:
                    raise ValueError(
                        f"Unsupported ig_split={ig_split}. Use 'val' or 'test'."
                    )

                if ig_eval_loader is None:
                    raise ValueError(
                        f"Integrated gradients requested for split '{ig_split}', but the corresponding dataloader is None."
                    )

                # create new dataloader without shuffeling
                train_dataloader_export = initialize_dataloader(
                    args,
                    train_view_target,
                    shuffle=False,  # IMPORTANT for export
                )
                ig_path = run_repquery_integrated_gradients(
                    model=repquery_model,
                    train_loader=train_dataloader_export,  # uses TRAIN for baseline_x, not shuffled
                    eval_loader=ig_eval_loader,  # IG computed on selected split
                    args=args,
                )
                logging.info(f"[IG] saved: {ig_path}")
                if wandb_logger and isinstance(wandb_logger, WandbLogger):
                    try:
                        wandb_logger.experiment.save(ig_path)
                    except Exception:
                        pass
        if args.save_outer_oof_predictions and _is_outer_oof_run(args):
            logging.warning(
                "save_outer_oof_predictions is currently implemented for non-Lightning models. "
                f"Skipping automatic OOF export for model {args.model_name}."
            )
        if args.save_test_predictions:
            logging.warning(
                "save_test_predictions is currently implemented for non-Lightning models. "
                f"Skipping automatic test prediction export for model {args.model_name}."
            )

    # Non-Lightning models
    elif args.model_name in ["XGBoost", "TabPFN"]:
        # ------------------------------------------------------------------
        # Optionally build ICD features for non-Lightning models
        # ------------------------------------------------------------------
        use_icd_baseline_onehot = getattr(args, "use_icd_baseline_onehot", False)
        if use_icd_baseline_onehot:
            print("[PRE-ICD] train:", train_view_target.X.shape)
            # Build ICD representation on the FULL adata (before splitting)
            (
                num_icd_codes,
                diag_cols,
                diag_col_to_row,
                assessment_col,
                per_subject_rows,
                per_subject_months,
            ) = icd_precompute(adata.obs, args)

            print(
                f"[ICD] num_icd_codes: {num_icd_codes}, #diag_cols used: {len(diag_cols)}"
            )

            temporal = bool(args.use_temporal_token)

            if temporal:
                icd_full = torch.zeros(
                    (adata.n_obs, num_icd_codes), dtype=torch.float32
                )
            else:
                icd_full = torch.zeros(adata.n_obs, num_icd_codes, dtype=torch.bool)

            for i in range(adata.n_obs):
                icd_full[i] = icd_build_representation(
                    index=i,
                    num_icd_codes=num_icd_codes,
                    per_subject_rows=per_subject_rows,
                    per_subject_months=per_subject_months,
                    use_temporal_token=temporal,
                )

            # IMPORTANT: align via global_row, not train_indices
            train_global_idx = train_view_target.obs["global_row"].to_numpy(
                dtype=np.int64
            )
            val_global_idx = val_view_target.obs["global_row"].to_numpy(dtype=np.int64)
            test_global_idx = test_view_target.obs["global_row"].to_numpy(
                dtype=np.int64
            )

            icd_train = icd_full[train_global_idx]
            icd_val = icd_full[val_global_idx]
            icd_test = icd_full[test_global_idx]

            train_view_target = _concat_features(
                train_view_target,
                icd_train,
                prefix="icd_",
                value_type="Continuous" if temporal else "Categorical single",
            )
            val_view_target = _concat_features(
                val_view_target,
                icd_val,
                prefix="icd_",
                value_type="Continuous" if temporal else "Categorical single",
            )
            test_view_target = _concat_features(
                test_view_target,
                icd_test,
                prefix="icd_",
                value_type="Continuous" if temporal else "Categorical single",
            )

            print(
                f"[ICD] Added ICD features: new train shape {train_view_target.X.shape}"
            )

        # ------------------------------------------------------------------
        # Optional baseline medication one-hot
        # ------------------------------------------------------------------
        use_meds_baseline = getattr(args, "use_meds_baseline_onehot", False)
        if use_meds_baseline:
            (
                num_meds_codes,
                indptr,
                indices,
                names,
                name_to_row,
                global_rows,
            ) = meds_precompute(adata.obs, args)

            print(
                f"[MEDS] num_meds_codes: {num_meds_codes}, "
                f"indptr: {None if indptr is None else indptr.shape}, "
                f"indices: {None if indices is None else indices.shape}"
            )

            meds_full = torch.zeros(adata.n_obs, num_meds_codes, dtype=torch.bool)
            for i in range(adata.n_obs):
                meds_full[i] = meds_build_representation(
                    index=i,
                    num_meds_codes=num_meds_codes,
                    indptr=indptr,
                    indices=indices,
                    names=names,
                    name_to_row=name_to_row,
                    global_rows=global_rows,
                )

            # Again: use global_row mapping to align
            train_global_idx = train_view_target.obs["global_row"].to_numpy(
                dtype=np.int64
            )
            val_global_idx = val_view_target.obs["global_row"].to_numpy(dtype=np.int64)
            test_global_idx = test_view_target.obs["global_row"].to_numpy(
                dtype=np.int64
            )

            meds_train = meds_full[train_global_idx]
            meds_val = meds_full[val_global_idx]
            meds_test = meds_full[test_global_idx]

            # Treat binary meds as Boolean (anything but 'Categorical single') because casting 2505 meds to categorical takes forever
            meds_value_type = "Boolean"  # "Categorical single"
            train_view_target = _concat_features(
                train_view_target, meds_train, prefix="med_", value_type=meds_value_type
            )
            val_view_target = _concat_features(
                val_view_target, meds_val, prefix="med_", value_type=meds_value_type
            )
            test_view_target = _concat_features(
                test_view_target, meds_test, prefix="med_", value_type=meds_value_type
            )

            print(
                f"[MEDS] Added meds features: new train shape {train_view_target.X.shape}"
            )

        model = model_selector(args)(
            args, train_view_target, val_view_target, test_view_target, wandb_logger
        )
        model.train()
        model.evaluate()

        should_save_outer_oof = args.save_outer_oof_predictions and _is_outer_oof_run(args)
        should_save_test_predictions = bool(getattr(args, "save_test_predictions", False))

        if should_save_outer_oof or should_save_test_predictions:
            try:
                preds, prediction_kind, extra_payload = _extract_non_lightning_test_predictions(model, args)

                if should_save_outer_oof:
                    _save_test_predictions(
                        args,
                        test_view_target,
                        preds,
                        prediction_kind,
                        run_id=(wandb.run.id if wandb.run else None),
                        fold=cast(int, args.cross_validation_current_fold),
                        extra_payload=extra_payload,
                    )

                if should_save_test_predictions:
                    _save_test_predictions(
                        args,
                        test_view_target,
                        preds,
                        prediction_kind,
                        run_id=(wandb.run.id if wandb.run else None),
                        fold=None,
                        extra_payload=extra_payload,
                    )
            except NotImplementedError as exc:
                logging.warning(f"Skipping prediction export: {exc}")

    if "train_dataloader" in locals():
        train_dataloader._iterator = None
        train_dataloader = None
    if "val_dataloader" in locals() and val_dataloader:
        val_dataloader._iterator = None
        val_dataloader = None

    vars_to_del = [
        "model",
        "train_view_target",
        "val_view_target",
        "test_view_target",
        "adata",
        "valid_features_mask",
        "train_indices",
        "val_indices",
        "test_indices",
    ]

    for var in vars_to_del:
        if var in locals():
            locals()[var] = None

    gc.collect()
    torch.cuda.empty_cache()

    if pipeline_queue:
        pipeline_queue.put(args)

    # Cleanup wandb
    run.finish()


def _get_allocated_gpu_count() -> int:
    v = os.environ.get("SLURM_GPUS_ON_NODE")
    if v and v.isdigit():
        return int(v)

    v = os.environ.get("SLURM_GPUS_PER_NODE")
    if v:
        # common forms: "2" or "2(x1)"
        head = v.split("(")[0].strip()
        if head.isdigit():
            return int(head)

    try:
        return torch.cuda.device_count()
    except Exception:
        return 1


def _create_sweep_if_needed(args: Config) -> None:
    if args.wandb_sweep_id:
        return

    sweep_config = retrieve_config(args, os.getcwd())
    _id = write_sweep_config_yaml(sweep_config, os.getcwd())
    yaml_path = f"{os.getcwd()}/sweep_config_{_id}.yaml"

    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    sweep_cmd = ["wandb", "sweep", "--project", args.wandb_project_name]
    if entity:
        sweep_cmd += ["--entity", entity]
    sweep_cmd.append(yaml_path)
    out = subprocess.run(
        sweep_cmd,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"Failed to create sweep:\nSTDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr}"
        )

    combined = out.stdout + "\n" + out.stderr
    sweep_id = None
    for line in combined.splitlines():
        if "Creating sweep with ID:" in line:
            sweep_id = line.split()[-1].strip()
            break
    if not sweep_id:
        raise RuntimeError(f"Could not extract sweep id from wandb output:\n{combined}")

    args.wandb_sweep_id = sweep_id
    os.remove(yaml_path)  # keep if you want reproducibility

def _is_outer_oof_run(args: Config) -> bool:
    return (
        args.cross_validation_n_folds > 0
        and getattr(args, "cv_mode", "fixed_test_inner") == "outer_oof"
        and args.cross_validation_current_fold is not None
    )


def _resolve_prediction_output_dir(args: Config, is_outer_oof: bool) -> Path:
    if is_outer_oof:
        out_dir = (
            Path(args.oof_output_dir)
            if args.oof_output_dir
            else Path(args.checkpoint_dir_path) / "outer_oof_predictions"
        )
    else:
        out_dir = (
            Path(args.test_predictions_output_dir)
            if args.test_predictions_output_dir
            else Path(args.checkpoint_dir_path) / "test_predictions"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _extract_non_lightning_test_predictions(
    model: Any, args: Config
) -> tuple[np.ndarray, str, dict[str, np.ndarray] | None]:
    model_name = args.model_name

    if model_name == "XGBoost":
        preds = np.asarray(model.bst.predict(model.dtest))
        if args.task == Task.CLASSIFICATION:
            if preds.ndim == 2:
                preds = preds[:, -1]
            return preds.reshape(-1), "probability", None
        if args.task == Task.SURVIVAL:
            survival_cache = getattr(model, "survival_export_cache", {}).get("test")
            if survival_cache is not None:
                return (
                    np.asarray(survival_cache["risk_scores"]).reshape(-1),
                    "risk_score",
                    {
                        "global_row": np.asarray(survival_cache["global_row"]).reshape(-1),
                        "event": np.asarray(survival_cache["event"], dtype=np.int64).reshape(-1),
                        "time": np.asarray(survival_cache["time"]).reshape(-1),
                        "survival_probs": np.asarray(survival_cache["survival_probs"]),
                        "time_grid": np.asarray(survival_cache["time_grid"]),
                    },
                )
            return preds.reshape(-1), "risk_score", None
        return preds.reshape(-1), "prediction", None

    if model_name == "TabPFN":
        probs = np.asarray(model.model.predict_proba(model.X_test))
        if probs.ndim == 2:
            probs = probs[:, -1]
        return probs.reshape(-1), "probability", None

    raise NotImplementedError(
        f"OOF export is not implemented for model '{model_name}'."
    )


def _save_test_predictions(
    args: Config,
    test_view_target: npAnnData,
    predictions: np.ndarray,
    prediction_kind: str,
    run_id: Optional[str] = None,
    fold: Optional[int] = None,
    extra_payload: dict[str, np.ndarray] | None = None,
) -> None:
    is_outer_oof = fold is not None

    preds = np.asarray(predictions).reshape(-1)
    if preds.shape[0] != test_view_target.n_obs:
        raise ValueError(
            f"Prediction length mismatch: got {preds.shape[0]}, expected {test_view_target.n_obs}"
        )

    if "global_row" in test_view_target.obs.columns:
        global_row = test_view_target.obs["global_row"].to_numpy(dtype=np.int64)
    else:
        global_row = np.arange(test_view_target.n_obs, dtype=np.int64)

    target_key = "__".join([str(t) for t in args.target_outcomes])
    safe_target = _sanitize_name(target_key)
    safe_model = _sanitize_name(args.model_name)

    safe_run = _sanitize_name(run_id) if run_id else "unknown"
    output_dir = _resolve_prediction_output_dir(args, is_outer_oof=is_outer_oof)
    if is_outer_oof:
        out_path = output_dir / f"{safe_model}_{safe_target}_fold{cast(int, fold):02d}.csv"
    else:
        out_path = output_dir / f"{safe_model}_{safe_target}_run_{safe_run}.csv"

    df = pd.DataFrame(
        {
            "global_row": global_row,
            "model_name": args.model_name,
            "task": args.task.value,
            "target": target_key,
            "prediction_kind": prediction_kind,
        }
    )
    if run_id is not None:
        df["run_id"] = str(run_id)
    if fold is not None:
        df["fold"] = int(fold)

    if args.task == Task.CLASSIFICATION:
        if prediction_kind == "probability":
            prob = np.clip(preds.astype(float), 1e-7, 1 - 1e-7)
            logit = np.log(prob / (1 - prob))
            df["probability"] = prob
            df["logit"] = logit
        elif prediction_kind == "logit":
            logit = preds.astype(float)
            prob = 1 / (1 + np.exp(-np.clip(logit, -50, 50)))
            df["logit"] = logit
            df["probability"] = prob
        else:
            df["score"] = preds.astype(float)
    else:
        df["score"] = preds.astype(float)

    if args.task == Task.SURVIVAL and extra_payload is not None:
        global_row_payload = extra_payload.get("global_row")
        if global_row_payload is not None:
            payload_global_row = np.asarray(global_row_payload, dtype=np.int64).reshape(-1)
            if payload_global_row.shape[0] == len(df):
                df["global_row"] = payload_global_row

        event_payload = extra_payload.get("event")
        if event_payload is not None:
            payload_event = np.asarray(event_payload, dtype=np.int64).reshape(-1)
            if payload_event.shape[0] == len(df):
                df["event"] = payload_event

        time_payload = extra_payload.get("time")
        if time_payload is not None:
            payload_time = np.asarray(time_payload, dtype=float).reshape(-1)
            if payload_time.shape[0] == len(df):
                df["time"] = payload_time

        surv_probs = extra_payload.get("survival_probs")
        time_grid = extra_payload.get("time_grid")
        if surv_probs is not None and time_grid is not None:
            surv_probs_arr = np.asarray(surv_probs, dtype=float)
            time_grid_arr = np.asarray(time_grid, dtype=float).reshape(-1)
            if (
                surv_probs_arr.ndim == 2
                and surv_probs_arr.shape[0] == len(df)
                and surv_probs_arr.shape[1] == len(time_grid_arr)
            ):
                for j in range(surv_probs_arr.shape[1]):
                    df[f"surv_prob_{j:03d}"] = surv_probs_arr[:, j]
                    # Repeated intentionally so a single CSV is self-contained.
                    df[f"time_grid_{j:03d}"] = float(time_grid_arr[j])

        cache_path = out_path.with_suffix(".npz")
        np.savez_compressed(
            cache_path,
            global_row=df["global_row"].to_numpy(dtype=np.int64),
            event=df["event"].to_numpy(dtype=np.int64) if "event" in df.columns else np.asarray([], dtype=np.int64),
            time=df["time"].to_numpy(dtype=float) if "time" in df.columns else np.asarray([], dtype=float),
            risk_scores=preds.astype(float),
            survival_probs=np.asarray(surv_probs, dtype=float) if surv_probs is not None else np.asarray([], dtype=float),
            time_grid=np.asarray(time_grid, dtype=float).reshape(-1) if time_grid is not None else np.asarray([], dtype=float),
        )

    if len(args.target_outcomes) == 1 and args.target_outcomes[0] in test_view_target.obs.columns:
        target_col = args.target_outcomes[0]
        y_true = test_view_target.obs[target_col].to_numpy()
        df["y_true"] = y_true

        # Survival metrics require both event and time.
        if args.task == Task.SURVIVAL:
            df["event"] = y_true
            parts = str(target_col).split("_", 1)
            if len(parts) == 2:
                time_col = f"time_{parts[1]}"
                if time_col in test_view_target.obs.columns:
                    df["time"] = test_view_target.obs[time_col].to_numpy()

    should_write_csv = not (args.task == Task.SURVIVAL and extra_payload is not None)
    if should_write_csv:
        df.to_csv(out_path, index=False)
        if is_outer_oof:
            logging.info(f"Saved outer OOF predictions to {out_path}")
        else:
            logging.info(f"Saved test predictions to {out_path}")
    elif args.task == Task.SURVIVAL:
        logging.info(f"Saved survival cache to {cache_path} (CSV export skipped)")


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def control(args: Config):
    os.environ["WANDB_MODE"] = "offline" if args.wandb_mode == "offline" else "online"

    if args.launch_sweep:
        args.epochs = 1000
        # args.wandb_project_name should be passed as command line argument

        _create_sweep_if_needed(args)

        # Reference for `wandb agent`. Prepend the entity only if one is set;
        # otherwise wandb uses the logged-in account's default entity.
        _sweep_entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
        sweep_ref = f"{args.wandb_project_name}/{args.wandb_sweep_id}"
        if _sweep_entity:
            sweep_ref = f"{_sweep_entity}/{sweep_ref}"

        n_gpus = _get_allocated_gpu_count()

        if n_gpus <= 1:
            cmd = [
                "wandb",
                "agent",
                "--count",
                str(args.hp_sweep_n_trials if args.hp_sweep_n_trials else 0),
                sweep_ref,
            ]
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                raise RuntimeError(f"wandb agent failed with code {rc}")
            return

        n_agents = n_gpus
        base_dir = Path(os.getcwd()) / f"wandb_agents_{args.wandb_sweep_id}"
        base_dir.mkdir(parents=True, exist_ok=True)

        total = int(args.hp_sweep_n_trials) if args.hp_sweep_n_trials else 0
        per_agent = (total // n_agents) if total else 0

        orig_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible = [d for d in orig_cvd.split(",") if d]
        if not visible:
            visible = [str(j) for j in range(n_agents)]

        print("CUDA_VISIBLE_DEVICES (parent):", orig_cvd)
        print("Parsed visible devices:", visible)

        if n_agents > len(visible):
            raise RuntimeError(
                f"Requested {n_agents} agents but only {len(visible)} visible CUDA devices: {visible}"
            )

        procs = []
        for i in range(n_agents):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = visible[i]
            print(f"Agent {i} -> CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")

            agent_dir = base_dir / f"agent_{i}"
            agent_dir.mkdir(parents=True, exist_ok=True)
            env["WANDB_DIR"] = str(agent_dir)

            cmd = [
                "wandb",
                "agent",
                "--count",
                str(per_agent if total else 0),
                sweep_ref,
            ]
            procs.append(subprocess.Popen(cmd, env=env))

        rcodes = [p.wait() for p in procs]
        if any(r != 0 for r in rcodes):
            raise RuntimeError(f"Some wandb agents failed: {rcodes}")
    else:
        run(args)


if __name__ == "__main__":
    control()
