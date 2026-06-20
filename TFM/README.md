# RisQ

Training code for **RisQ/RepQuery**, **XGBoost**, and **TabPFN** as used in the paper.

RepQuery encodes all available patient data (lifestyle, blood biomarkers, genetics, medications, clinical history) into a single patient representation that can be queried across any time horizon to simultaneously predict the onset of hundreds of diseases.

---

## Quickstart

```bash
# 1. Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (run from this TFM/ directory)
uv sync

# 3. Create your paths config and fill in the data/output directories
cp configs/paths/example.yaml configs/paths/mymachine.yaml
$EDITOR configs/paths/mymachine.yaml

# 4. Train the default model (RepQuery)
uv run train.py paths=mymachine
```

The sections below cover each step in detail, the available models, and the main training options.

---

## Setup

### Prerequisites

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository and enter the TFM directory**:
   ```bash
   git clone <repository-url>
   cd foundational-ukb/TFM
   ```

3. **Create the virtual environment and install dependencies**:
   ```bash
   uv sync
   source .venv/bin/activate
   ```

4. **Verify**:
   ```bash
   python -c "import torch, numpy, pandas, anndata; print('OK')"
   ```

### Platform notes

The included `uv.lock` was generated on Linux with CUDA 12.1. For other platforms:

```bash
rm uv.lock && uv sync          # fresh resolution
```

If `torch-scatter` fails to load with an undefined-symbol error, reinstall it against your torch build:

```bash
uv pip uninstall torch-scatter
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

---

## Configuration

Runs are controlled by [Hydra](https://hydra.cc/) config files in [`configs/`](configs/). The base config is [`configs/config.yaml`](configs/config.yaml).

Hydra uses two config groups:

| Group | Location | Purpose |
|---|---|---|
| `model` | `configs/model/` | Model type and hyperparameters |
| `paths` | `configs/paths/` | Data and output paths for your machine |

The default model is `RepQuery`. Override group values on the command line:

```bash
uv run train.py model=XGBoost paths=mymachine
```

> Run scripts with `uv run train.py …` (no activation needed), or activate the
> environment once (`source .venv/bin/activate`) and use `python train.py …`.
> The examples below use `python train.py` and assume an activated environment.

### Setting up your paths

Copy the template and fill in the paths for your machine:

```bash
cp configs/paths/example.yaml configs/paths/mymachine.yaml
```

Edit the fields:

```yaml
paths_name: "mymachine"
data_root_path: "/path/to/dataset/complete"   # directory containing the .h5ad file
wandb_dir_path: "/path/to/wandb"               # W&B run metadata
checkpoint_dir_path: "/path/to/checkpoints"    # model checkpoints
num_workers: 8
```

Then launch with `paths=mymachine`.

> **Note:** Hydra changes the working directory at launch by default. Run `train.py` from the `TFM/` directory so the relative modality-dropout paths in your paths config resolve, or pass absolute paths for `modality_dropout_groups_dir` / `modality_dropout_protected_path`.

---

## Models

### RepQuery

The primary model. A transformer encoder builds a patient representation; a cross-attention decoder queries it at a given time horizon, producing survival probabilities for all disease targets jointly.

**Training command:**

```bash
python train.py model=RepQuery paths=mymachine
```

The defaults in [`configs/model/RepQuery.yaml`](configs/model/RepQuery.yaml) reproduce the published configuration, so no extra overrides are needed. Override any of them on the command line to experiment, e.g. `python train.py model=RepQuery paths=mymachine lr=0.001 modality_dropout_p=0.4`.

Key hyperparameters (defaults shown reflect the published configuration; see `config_types.py` for the complete list):

| Parameter | Default | Description |
|---|---|---|
| `n_layers` | 1 | Encoder transformer layers |
| `hidden_dim` | 64 | Encoder hidden dimension |
| `num_heads` | 4 | Encoder attention heads |
| `decoder_n_layers` | 1 | Decoder layers |
| `n_cls_tokens` | 64 | Number of CLS tokens (query embeddings) |
| `lr` | 0.003 | Learning rate |
| `temporal_sampling_strategy` | `DIAG_CENTERED` | Query-time sampling: `UNIFORM`, `GAUSSIAN`, or `DIAG_CENTERED` |
| `feature_dropout_p` | 0.0 | Per-feature dropout probability during training |
| `modality_dropout_p` | 0.6 | Probability of dropping a modality group during training |
| `icd_hierarchy_level` | `[3]` | ICD-10 hierarchy depth for the prognosis pretraining task |
| `years_to_diag` | 15.0 | Maximum prediction horizon (years) |
| `eval_horizons` | `[2,5,10,15]` | Horizons evaluated at validation/test time |
| `epochs` | 300 | Maximum epochs (early stopping active, patience 20) |

**Modality dropout** is controlled by the files in `configs/vars_to_keep/modality_groups/`. Each `*_only.txt` file lists the UKBB field IDs for one modality group. The file given by `modality_dropout_protected_path` (`ehr_only.txt` by default) is never dropped.

### XGBoost

Gradient-boosted tree baseline. Trains **one model per disease target**, so a target field ID must always be provided via `target_outcomes`. XGBoost runs in one of two modes set by `task`:

- **`task=SURVIVAL`** — a survival model (Harrell c-index, IPCW c-index, Antolini c-index, plus horizon-derived AUC from the predicted survival curve). Requires **exactly one** `target_outcome`.
- **`task=CLASSIFICATION`** — binary classification at a **fixed prediction horizon**. The horizon (in years) must be provided via `years_to_diag`.

```bash
# Survival
python train.py model=XGBoost paths=mymachine task=SURVIVAL target_outcomes=[130708]

# Classification at a 2-year horizon
python train.py model=XGBoost paths=mymachine task=CLASSIFICATION target_outcomes=[130708] years_to_diag=2
```

XGBoost-specific parameters are under the `# XGBoost` section of `config_types.py` (e.g. `max_depth`, `eta`, `subsample`).

#### Hyperparameter sweeps

XGBoost (and RepQuery) support W&B Bayesian sweeps. Set `launch_sweep=True`; the run creates a W&B sweep and launches agent(s) automatically. Provide the same target/task/horizon you would for a single run:

```bash
# Sweep XGBoost classification at a 2-year horizon
python train.py model=XGBoost paths=mymachine launch_sweep=True \
  task=CLASSIFICATION target_outcomes=[130708] years_to_diag=2 \
  wandb_project_name=xgb_sweep hp_sweep_n_trials=20
```

The swept grid (`max_depth`, `min_child_weight`, `subsample`, `eta`, `colsample_bytree`, `gamma`, `lambda_l2`, `alpha_l1`) and the optimized metric (Harrell c-index for survival, AUC for classification) are defined in `infra/utils.py` (`retrieve_config`). Control the search with `sweep_method` (default `bayes`) and `hp_sweep_n_trials`.

### TabPFN

In-context-learning baseline using TabPFN.

```bash
python train.py model=TabPFN paths=mymachine
```

`subsample_size` (default 5000) caps the number of training rows passed to TabPFN due to its memory constraints.

> **Note:** This is TabPFN v2. The paper results use **TabPFN v3**, which runs in a
> separate environment and is not yet merged into this branch. The v3 integration
> still needs to be merged in before this matches the paper exactly.

---

## General training options

| Parameter | Default | Description |
|---|---|---|
| `target_outcomes` | `[]` (all targets) | Field IDs to train on; empty = all |
| `task` | `PROGNOSIS` | `PROGNOSIS`, `SURVIVAL`, `CLASSIFICATION`, `REGRESSION` |
| `seed` | 2024 | Global random seed |
| `val_size` / `test_size` | 0.15 / 0.15 | Train/val/test split fractions |
| `cross_validation_n_folds` | 0 | Enable K-fold cross-validation |
| `use_wandb` | `True` | Log to Weights & Biases |
| `wandb_project_name` | `RisQ` | W&B project name |
| `pretrained_weights_path` | `null` | Checkpoint to fine-tune from |

By default, experiments are logged to your logged-in W&B account's default entity. To log to a specific team/user, set `wandb_entity` in your config (or the `WANDB_ENTITY` environment variable).

---

## Evaluation

After training, test-set evaluation runs automatically. To save per-patient predictions:

```bash
python train.py ... save_test_predictions=True test_predictions_output_dir=/path/to/output
```

---

## Integrated Gradients

Feature attribution via integrated gradients can be run post-training by setting `run_integrated_gradients=True`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `ig_targets_mode` | `all` | `all` targets or a specific `list` |
| `ig_horizons_years` | `[2]` | Horizons to explain |
| `ig_num_samples` | 2000 | Max balanced case-control samples per (target, horizon) |
| `ig_out_dir` | `<checkpoint_dir>/integrated_gradients` | Output directory |
