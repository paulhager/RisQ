import os
import time
import random
from pathlib import Path
from typing import Optional
import subprocess
import json

from config_types import Config, Task


# Best-effort helper to resolve a W&B run ID from a human-readable run name.
# This allows resuming/logging to an existing run when only the run name is known
def _find_wandb_run_id_by_name(
    project: str,
    run_name: str,
    entity: Optional[str] = None,
) -> Optional[str]:
    try:
        import wandb  # local import to avoid side-effects at module import time

        api = wandb.Api()
        entity_or_default = (
            entity or os.environ.get("WANDB_ENTITY") or api.default_entity
        )
        runs = api.runs(
            f"{entity_or_default}/{project}", filters={"display_name": run_name}
        )
        if not runs:
            # Fallback to client-side filtering if server-side filter is unavailable
            runs = [
                r
                for r in api.runs(f"{entity_or_default}/{project}")
                if r.name == run_name
            ]
        if not runs:
            return None
        if len(runs) > 1:
            runs = sorted(runs, key=lambda r: r.created_at, reverse=True)
            print(
                f"WARNING: Multiple runs found for '{run_name}' in project '{project}'. Taking the most recent one."
            )
        return runs[0].id
    except Exception as e:
        # Non-fatal: if we cannot resolve, just return None and proceed without an ID
        print(f"Warning: failed to resolve W&B run id for '{run_name}': {e}")
        return None


def _resolve_wandb_id(args: Config, checkpoint_path: Optional[str]) -> Optional[str]:
    resolved_wandb_id = args.wandb_id if getattr(args, "wandb_id", None) else None
    if not resolved_wandb_id and checkpoint_path:
        try:
            run_name_from_ckpt = Path(checkpoint_path).parent.name
            if os.environ["WANDB_MODE"] != "offline":
                resolved_wandb_id = _find_wandb_run_id_by_name(
                    project=args.wandb_project_name,
                    run_name=run_name_from_ckpt,
                    entity=getattr(args, "wandb_entity", None) or os.environ.get("WANDB_ENTITY"),
                )
        except Exception as e:
            print(f"Warning: unable to derive W&B run id from checkpoint path: {e}")

    return resolved_wandb_id

def setup_wandb_mode(args: Config, sweep_id=None):
    # Respect requested mode (online/offline)
    os.environ["WANDB_MODE"] = "offline" if args.wandb_mode == "offline" else "online"

    # If multiple agents, still separate local directories to avoid conflicts
    if args.wandb_n_agents > 1:
        sweep_identifier = (
            sweep_id
            if sweep_id
            else f"pending_sweep_{int(time.time()*1000)}_{random.randint(0, 999999):06d}"
        )
        sweep_dir = f"wandb_runs_{sweep_identifier}"
        os.makedirs(sweep_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = sweep_dir
        return sweep_dir

    return None


def sync_wandb_runs(sweep_dir: str):
    """Sync all wandb runs from the specified directory"""
    if not sweep_dir or not os.path.exists(sweep_dir):
        return

    print(f"\nSyncing wandb runs from {sweep_dir}...")

    # Get all run directories
    wandb_dir = Path(sweep_dir) / "wandb"
    if not wandb_dir.exists():
        print("No wandb directory found to sync")
        return

    # Sync with retry logic
    max_retries = 3
    retry_delay = 10

    for run_dir in wandb_dir.glob("run-*"):
        if not (run_dir / "files").exists():
            print(f"Skipping {run_dir.name} as it has no files")
            continue

        for attempt in range(max_retries):
            try:
                print(f"Syncing run {run_dir.name}")
                result = subprocess.run(
                    ["wandb", "sync", str(run_dir)], capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f"Successfully synced {run_dir.name}")
                    break
                else:
                    print(f"Error syncing {run_dir.name}: {result.stderr}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            except Exception as e:
                print(f"Exception while syncing {run_dir.name}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)


def retrieve_config(args: Config, path: str) -> dict:
    # ----- choose metric by task (default, can be overridden below) -----
    if args.task == Task.CLASSIFICATION:
        metric_cfg = {
            "goal": "maximize",
            "name": "supervised.val.auc.max",
        }
    elif args.task == Task.REGRESSION:
        metric_cfg = {
            "goal": "minimize",
            "name": "supervised.val.mae.min",
        }
    elif args.task == Task.SURVIVAL:
        # sensible default for survival
        metric_cfg = {
            "goal": "maximize",
            "name": "supervised.val.c_index_harrell.max",
        }
    elif args.task == Task.PROGNOSIS:
        metric_cfg = {
            "goal": "maximize",
            "name": "val/macro_c_index",
        }
    else:
        raise ValueError(f"Unknown task type: {args.task}")
    
    # Build command list, conditionally including years_to_diag for classification tasks
    command_list = [
        "python",
        "${program}",
        f"paths={args.paths_name}",
        f"model={args.model_name}",
        f"target_outcomes={args.target_outcomes}",
        f"task={args.task}",
        f"include_gp={args.include_gp}",
        "hp_sweep=True",
        f"checkpoint_dir_path={args.checkpoint_dir_path}",
        f"wandb_project_name={args.wandb_project_name}",
    ]
    if args.task == Task.CLASSIFICATION:
        command_list.append(f"years_to_diag={args.years_to_diag}")
    
    sweep_config = {
        "program": f"{path}/train.py",
        "command": command_list,
        "method": args.sweep_method,
        "name": f"{args.model_name}_{args.task}_{args.target_outcomes}_{args.years_to_diag}_{args.target_file_name}" if args.task == Task.CLASSIFICATION else f"{args.model_name}_{args.task}_{args.target_outcomes}_{args.target_file_name}",
        "metric": metric_cfg,
    }
    if args.model_name == "XGBoost":
        sweep_config["parameters"] = {
            "max_depth": {"values": [3, 5, 10]},
            "min_child_weight": {"values": [1, 5, 10]},
            "subsample": {"values": [0.5, 0.75, 1]},
            "eta": {"values": [0.01, 0.1, 0.3]},
            # "colsample_bylevel": {"values": [0.5, 0.75, 1]},
            "colsample_bytree": {"values": [0.5, 0.75, 1]},
            "gamma": {"values": [0, 1, 10]},
            "lambda_l2": {"values": [0, 1, 10]},
            "alpha_l1": {"values": [0, 1, 10]},
        }

    elif args.model_name == "RepQuery":
        # Metric: minimize RepQuery.val.loss
        sweep_config["parameters"] = {
            # core training
            "lr": {"values": [3e-4, 6e-4, 1e-3, 2e-3, 3e-3]}, # 1e-2, 1e-3, 1e-4
            "weight_decay": {"values": [0.0, 1e-4, 1e-3, 1e-2]},
            "batch_size": {"values": [128, 512, 1024]},
            "temporal_sampling_strategy": {"values": ["DIAG_CENTERED"]},

            # FT backbone knobs (RepQuery uses init_ft_transformer)
            "n_layers": {"values": [1]}, # 1, 2, 4
            "hidden_dim": {"values": [64]}, # 8, 16, 32, 
            "num_heads": {"values": [4]}, # 2, 4, 8
            "attention_dropout": {"values": [0]}, # 0.05, 0.1
            "ffn_hidden_dim_multiplier": {"values": [4]}, # 2
            "ffn_dropout": {"values": [0]},
            "n_cls_tokens": {"values": [64]}, # 32, 64, 128; 1, 32, 96 tested

            # decoder knobs
            "decoder_num_heads": {"values": [4]}, #2, 4, 8
            #"decoder_hidden_dim": {"values": [32, 64, 128]},
            "decoder_n_layers": {"values": [1]}, # 4, 8, 16
            "decoder_attention_dropout": {"values": [0]},
            "decoder_ffn_hidden_dim_multiplier": {"values": [4]},
            "decoder_ffn_dropout": {"values": [0]},

            # task-specific
            #"repquery_negative_sample_weight": {"values": [1.0, 0.5, 0.25]},
            "icd_hierarchy_level": {"values": [[3]]},
            "feature_dropout_p": {"values": [0.0, 0.2]},
            "repquery_negative_sample_weight": {"values": [1.0]}, # 0.5, 0.25, 0.1, 0.05
        }


    else:
        raise ValueError(
            f"Model {args.model_name} not supported for hyperparameter sweeps"
        )

    sweep_config["parameters"]["enable_progress_bar"] = {"value": False}
    sweep_config["parameters"]["pretrained_weights_path"] = {
        "value": args.pretrained_weights_path
    }
    sweep_config["parameters"]["target_file_name"] = {"value": args.target_file_name}
    return sweep_config


def write_sweep_config_yaml(sweep_config: dict, path: str) -> int:
    _id = random.randint(0, 1000000)
    with open(f"{path}/sweep_config_{_id}.yaml", "w") as f:
        f.write(json.dumps(sweep_config, indent=4))
    return _id
