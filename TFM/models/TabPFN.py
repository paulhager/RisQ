import logging
import numpy as np
from tabpfn import TabPFNClassifier
import torch
import anndata as ad
from pytorch_lightning.loggers.wandb import WandbLogger
from scipy import sparse
from sklearn.model_selection import train_test_split

from models.utils import evaluate_metrics, MetricTracker
from config_types import Config, Task


class TabPFN:
    """
    Collection of generalized linear models implemented through sklearn
    """

    def __init__(
        self,
        args: Config,
        train_view: ad.AnnData,
        val_view: ad.AnnData,
        test_view: ad.AnnData,
        wandb_logger: WandbLogger,
    ):
        self.args = args
        self.wandb_logger = wandb_logger

        target = self.args.target_outcomes

        self.X_train = train_view.X
        self.y_train = train_view.obs[target].to_numpy()

        self.X_val = val_view.X
        self.y_val = val_view.obs[target].to_numpy()

        self.X_test = test_view.X
        self.y_test = test_view.obs[target].to_numpy()

        # Sample down to 5k samples if needed due to memory constraints
        if self.X_train.shape[0] > self.args.subsample_size:
            self.X_train, _, self.y_train, _ = train_test_split(
                self.X_train,
                self.y_train,
                train_size=self.args.subsample_size,
                stratify=self.y_train,
            )

        # If after sampling we have no positive cases, add all positive cases to the training set
        if self.y_train.sum() == 0:
            positive_indices = np.where(train_view.obs[target].to_numpy())[0]
            self.X_train = np.concatenate(
                [self.X_train, train_view[positive_indices].X]
            )
            self.y_train = np.concatenate(
                [self.y_train, train_view[positive_indices].obs[target].to_numpy()]
            )

        # cat_indices = np.where(train_view.var["n_categorical_options"] > 2)[0]
        self.model = TabPFNClassifier(
            n_estimators=1,
            fit_mode="low_memory",
            memory_saving_mode=True,
            inference_precision=torch.float32,
            ignore_pretraining_limits=True,
            random_state=self.args.seed,
        )
        self.eval_batch_size = int(getattr(self.args, "tabpfn_eval_batch_size", 256))

        self.logger = wandb_logger
        self.metric_tracker = MetricTracker(self.args.task)

        logging.info(
            "TabPFN settings: n_estimators=1, fit_mode=low_memory, "
            "memory_saving_mode=True, inference_precision=torch.float32, "
            "eval_batch_size=%s",
            self.eval_batch_size,
        )

    def _log_split_diagnostics(self, split: str, x, y) -> None:
        y_np = np.asarray(y).reshape(-1)
        unique_values, counts = np.unique(y_np, return_counts=True)
        label_hist = {
            str(label): int(count) for label, count in zip(unique_values.tolist(), counts.tolist())
        }

        if sparse.issparse(x):
            nan_count = int(np.isnan(x.data).sum())
            inf_count = int(np.isinf(x.data).sum())
            finite = x.data[np.isfinite(x.data)]
            finite_min = float(finite.min()) if finite.size else None
            finite_max = float(finite.max()) if finite.size else None
            all_nan_cols = None
            density = float(x.nnz / (x.shape[0] * x.shape[1]))
        else:
            x_np = np.asarray(x)
            nan_mask = np.isnan(x_np)
            inf_mask = np.isinf(x_np)
            nan_count = int(nan_mask.sum())
            inf_count = int(inf_mask.sum())
            finite = x_np[np.isfinite(x_np)]
            finite_min = float(finite.min()) if finite.size else None
            finite_max = float(finite.max()) if finite.size else None
            all_nan_cols = int(nan_mask.all(axis=0).sum())
            density = None

        logging.info(
            "TabPFN diagnostics [%s]: shape=%s, y_hist=%s, nan_count=%s, "
            "inf_count=%s, all_nan_cols=%s, finite_min=%s, finite_max=%s, density=%s",
            split,
            tuple(x.shape),
            label_hist,
            nan_count,
            inf_count,
            all_nan_cols,
            finite_min,
            finite_max,
            density,
        )

    def train(self):
        self._log_split_diagnostics("train_fit", self.X_train, self.y_train)
        self.model.fit(self.X_train, self.y_train)
        logging.info(
            "TabPFN fit complete: resolved_device=%s",
            getattr(self.model, "device_", "unknown"),
        )

    def evaluate(self):
        # self.eval(self.X_train, self.y_train, "train")
        # self.eval(self.X_val, self.y_val, "val")
        if self.args.test:
            self.eval(self.X_test, self.y_test, "test")

    def predict_proba_batched(self, x: np.ndarray) -> np.ndarray:
        logging.info(
            "TabPFN predict_proba_batched: n_rows=%s, batch_size=%s, n_batches=%s",
            x.shape[0],
            self.eval_batch_size,
            (x.shape[0] + self.eval_batch_size - 1) // self.eval_batch_size,
        )
        if x.shape[0] <= self.eval_batch_size:
            return np.asarray(self.model.predict_proba(x))

        preds = []
        for start in range(0, x.shape[0], self.eval_batch_size):
            stop = start + self.eval_batch_size
            preds.append(np.asarray(self.model.predict_proba(x[start:stop])))

        return np.concatenate(preds, axis=0)

    def eval(self, x: np.ndarray, y: np.ndarray, split: str):
        self._log_split_diagnostics(split, x, y)
        probs = self.predict_proba_batched(x)
        if (
            self.args.task == Task.CLASSIFICATION
            and probs.ndim == 2
            and probs.shape[1] == 2
        ):
            probs = probs[:, 1]

        y_pred = torch.Tensor(probs)
        y_gt = torch.Tensor(y).flatten()
        metrics = evaluate_metrics(self.args, y_pred, y_gt, split, self.metric_tracker)
        self.logger.log_metrics(metrics)
