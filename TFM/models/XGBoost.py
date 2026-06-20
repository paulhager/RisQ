from wandb.integration.xgboost import WandbCallback
import os
import json
import pytorch_lightning as pl
import xgboost as xgb
import pandas as pd
from typing import Tuple, List
import numpy as np
from omegaconf import ListConfig, OmegaConf
import tqdm
import torch
import optuna
from torchmetrics.functional import mean_absolute_error, auroc
from pytorch_lightning.loggers.wandb import WandbLogger
from typing import Dict, Optional

from config_types import Config, Task
from models.utils import evaluate_metrics, MetricTracker
from dataset.utils import npAnnData
from sksurv.metrics import concordance_index_censored


class XGBoost:
    """
    XGBoost model implemented through xgboost
    """

    def __init__(
        self,
        args: Config,
        train_view: npAnnData,
        val_view: npAnnData,
        test_view: npAnnData,
        wandb_logger: WandbLogger,
    ):
        self.args = args

        target = self.args.target_outcomes
        self.feature_names = train_view.var.index.tolist()
        self.target = target
        self.survival_export_cache: Dict[str, Dict[str, np.ndarray]] = {}

        print(f"Initializing XGBoost with task={self.args.task}, target_outcomes={target}")

        if self.args.task == Task.SURVIVAL:
            if len(target) != 1:
                print("XGBoost survival self.args.target_outcomes:", target)
                raise ValueError("XGBoost survival supports exactly one target_outcome")

            event_col = target[0]
            time_suffix = event_col.split("_")[1]
            time_col = f"time_{time_suffix}"

            # TRAIN
            train_time = train_view.obs[time_col].to_numpy().astype(np.float32)
            train_event = train_view.obs[event_col].to_numpy().astype(np.int32)
            train_label = np.where(train_event == 1, train_time, -train_time)

            self.dtrain = self._make_dmatrix(train_view, train_label)

            # For metrics: [event, time]
            self.y_train = torch.stack(
                [
                    torch.tensor(train_event, dtype=torch.float32),
                    torch.tensor(train_time, dtype=torch.float32),
                ],
                dim=1,
            )

            # VAL
            val_time = val_view.obs[time_col].to_numpy().astype(np.float32)
            val_event = val_view.obs[event_col].to_numpy().astype(np.int32)
            val_label = np.where(val_event == 1, val_time, -val_time)

            self.dval = self._make_dmatrix(val_view, val_label)

            self.y_val = torch.stack(
                [
                    torch.tensor(val_event, dtype=torch.float32),
                    torch.tensor(val_time, dtype=torch.float32),
                ],
                dim=1,
            )

            # TEST
            test_time = test_view.obs[time_col].to_numpy().astype(np.float32)
            test_event = test_view.obs[event_col].to_numpy().astype(np.int32)
            test_label = np.where(test_event == 1, test_time, -test_time) # Positive time means event, negative time means censoring

            self.dtest = self._make_dmatrix(test_view, test_label)

            self.y_test = torch.stack(
                [
                    torch.tensor(test_event, dtype=torch.float32),
                    torch.tensor(test_time, dtype=torch.float32),
                ],
                dim=1,
            )

            # ----- store raw arrays for baseline hazard + IBS / IPCW -----
            self.train_time = train_time
            self.train_event = train_event
            self.val_time = val_time
            self.val_event = val_event
            self.test_time = test_time
            self.test_event = test_event

            # structured arrays for sksurv
            def to_structured(ev: np.ndarray, t: np.ndarray) -> np.ndarray:
                return np.array(
                    list(zip(ev.astype(bool), t.astype(float))),
                    dtype=[("event", bool), ("time", float)],
                )

            self.survival_train = to_structured(train_event, train_time)
            self.survival_val = to_structured(val_event, val_time)
            self.survival_test = to_structured(test_event, test_time)

        else:
            # existing regression / classification path
            train_label = train_view.obs[target].values.flatten()
            self.dtrain = self._make_dmatrix(train_view, train_label)

            self.y_train = torch.Tensor(train_view.obs[target].values.flatten())

            val_label = val_view.obs[target].values.flatten()
            self.dval = self._make_dmatrix(val_view, val_label)

            self.y_val = torch.Tensor(val_view.obs[target].values.flatten())

            test_label = test_view.obs[target].values.flatten()
            self.dtest = self._make_dmatrix(test_view, test_label)

            self.y_test = torch.Tensor(test_view.obs[target].values.flatten())
        
        print(f"XGBoost data setup complete. Train samples: {self.dtrain.num_row()}, Val samples: {self.dval.num_row()}, Test samples: {self.dtest.num_row()}")

        self.feature_importance_history = {}

        # Keep references to views for c-index-from-prob computation (classification case)
        self.train_view = train_view
        self.val_view = val_view
        self.test_view = test_view

        if self.args.task == Task.REGRESSION:
            self.xgb_objective = "reg:squarederror"
            self.eval_metric = ["mae", "rmse"]
            self.direction = "minimize"
            self.main_metric = "mae"
        elif self.args.task == Task.CLASSIFICATION:
            self.xgb_objective = "binary:logistic"
            self.eval_metric = ["aucpr", "auc", "logloss"] # Last metric is used for early stopping
            self.direction = "maximize"
            self.main_metric = getattr(self.args, "classification_main_metric", "auc")
        elif self.args.task == Task.SURVIVAL:
            # Cox proportional hazards
            self.xgb_objective = "survival:cox"
            self.eval_metric = ["cox-nloglik"] # Last metric is used for early stopping
            self.direction = "minimize"
            self.main_metric = getattr(self.args, "survival_main_metric", "cox-nloglik")
        else:
            raise ValueError("Task must be either regression or classification or survival")

        self.param = {
            "nthread": -1,
            "max_depth": self.args.max_depth,
            "eta": self.args.eta,
            "objective": self.xgb_objective,
            "eval_metric": self.eval_metric,
            "tree_method": self.args.tree_method,
            "seed": self.args.seed,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "colsample_bylevel": args.colsample_bylevel,
            "lambda": args.lambda_l2,
            "alpha": args.alpha_l1,
            "gamma": args.gamma,
            "min_child_weight": args.min_child_weight,
            "n_jobs": 1,  # needed for deterministic results
        }

        if not getattr(self.args, "xgb_use_imputation_onehot", True):
            tm = self.param.get("tree_method")
            if tm not in ("hist", "gpu_hist"):
                raise ValueError(
                    "xgb_use_imputation_onehot=False (native categorical) requires "
                    "tree_method='hist' or 'gpu_hist'."
                )

        if self.args.task == Task.CLASSIFICATION and self.args.scale_pos_weight:
            pos = train_view.obs[target].sum()
            neg = len(train_view.obs[target]) - pos
            self.param["scale_pos_weight"] = (neg / pos) if pos > 0 else 1.0

        self.evallist = [(self.dtrain, "train"), (self.dval, "val")]

        self.logger = wandb_logger
        self.metric_tracker = MetricTracker(self.args.task)

        if wandb_logger and isinstance(wandb_logger, WandbLogger):
            print(f"Setting up checkpoint directory for XGBoost at: {self.args.checkpoint_dir_path}")
            self.args.checkpoint_dir_path = os.path.join(
                self.args.checkpoint_dir_path, wandb_logger.experiment.name
            )
            os.makedirs(self.args.checkpoint_dir_path, exist_ok=True)
            print(f"Created directory: {self.args.checkpoint_dir_path}")

    def train(self) -> None:
        best_model_callback = GetBestModel(self, self.direction)
        self.bst = xgb.train(
            params=self.param,
            dtrain=self.dtrain,
            num_boost_round=self.args.epochs,
            evals=self.evallist,
            early_stopping_rounds=self.args.patience,
            callbacks=[
                WandbCallback(log_feature_importance=False),
                best_model_callback,
            ],
        )

        self.bst = best_model_callback.best_model

        out_dir = self.args.checkpoint_dir_path
        os.makedirs(out_dir, exist_ok=True)

        if getattr(self.args, "save_xgb_model", False):
            model_path = os.path.join(out_dir, "xgboost_best_model.json")
            self.bst.save_model(model_path)

        best_score = getattr(best_model_callback, "best_score", None)
        best_iteration = getattr(self.bst, "best_iteration", None)
        wandb_experiment = getattr(self.logger, "experiment", None)

        meta_payload = {
            "model_name": "XGBoost",
            "task": str(getattr(self.args, "task", "")),
            "target_outcomes": list(getattr(self.args, "target_outcomes", [])),
            "feature_names": list(self.feature_names),
            "xgb_params": dict(self.param),
            "main_metric": str(getattr(self, "main_metric", "")),
            "eval_metric": list(getattr(self, "eval_metric", [])),
            "best_score": float(best_score) if best_score is not None else None,
            "best_iteration": int(best_iteration) if best_iteration is not None else None,
            "checkpoint_dir_path": str(out_dir),
            "years_to_diag": getattr(self.args, "years_to_diag", None),
            "include_gp": getattr(self.args, "include_gp", None),
            "wandb_project_name": getattr(self.args, "wandb_project_name", None),
            "wandb_run_name": getattr(wandb_experiment, "name", None) if wandb_experiment is not None else None,
            "wandb_run_id": getattr(wandb_experiment, "id", None) if wandb_experiment is not None else None,
        }

        meta_path = os.path.join(out_dir, "xgboost_best_model_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta_payload, f)

        if self.args.calculate_gain:
            importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
            feature_importance_dict = {}

            for importance_type in importance_types:
                try:
                    scores = self.bst.get_score(importance_type=importance_type)
                    # Fill in missing features with 0
                    for feature in self.feature_names:
                        if feature not in scores:
                            scores[feature] = 0
                    feature_importance_dict[importance_type] = scores

                    # Convert to format suitable for wandb Table
                    df = pd.DataFrame.from_dict(
                        scores, orient="index", columns=[importance_type]
                    )
                    df.index.name = "feature"

                    # Log to wandb using log_table
                    self.logger.log_table(
                        key=f"feature_importance_{importance_type}",
                        dataframe=df.reset_index(),
                    )

                    # Store in history
                    if importance_type not in self.feature_importance_history:
                        self.feature_importance_history[importance_type] = []
                    self.feature_importance_history[importance_type].append(scores)

                except Exception as e:
                    print(f"Could not compute {importance_type} importance: {str(e)}")

        if self.args.calculate_permutation_importance and self.args.task != Task.SURVIVAL:
            self.calculate_permutation_importance("val")

    def evaluate(self) -> None:
        if self.args.task == Task.SURVIVAL:
            # Train split: use survival_train for both train & val arguments in IBS/IPCW
            #self._eval_survival_split(
            #    dmatrix=self.dtrain,
            #    gt=self.y_train,
            #    split="train",
            #    survival_val_struct=self.survival_train,
            #)
            # Val split
            self._eval_survival_split(
                dmatrix=self.dval,
                gt=self.y_val, # [event, time]; positive time means event, negative time means censoring
                split="val",
                survival_val_struct=self.survival_val,
            )

            if self.args.test:
                # Before evaluating on test, we have to recompute the time grid on the train+test data (instead of train+val)
                self._eval_survival_split(
                    dmatrix=self.dtest,
                    gt=self.y_test, # [event, time]; positive time means event, negative time means censoring
                    split="test",
                    survival_val_struct=self.survival_test,
                )
        else:
            # existing classification/regression path unchanged
            self.eval(self.dtrain, self.y_train, "train")
            self.eval(self.dval,   self.y_val,   "val")

            # val_subset evaluation only supported for the array-based DMatrix path
            if getattr(self.args, "xgb_use_imputation_onehot", True):
                pl.seed_everything(2025 + 100)
                dval_subset_pos_indices = torch.where(self.y_val == 1)[0].tolist()
                dval_subset_neg_indices = torch.where(self.y_val == 0)[0].tolist()

                if self.args.subsample_final_validation and len(dval_subset_pos_indices) > 0:
                    dval_subset_neg_indices = np.random.choice(
                        dval_subset_neg_indices,
                        size=min(len(dval_subset_pos_indices), len(dval_subset_neg_indices)),
                        replace=False,
                    )

                indices = np.concatenate([dval_subset_pos_indices, dval_subset_neg_indices])

                self.dval_subset = xgb.DMatrix(
                    self.dval.get_data()[indices],
                    label=self.y_val[indices],
                    feature_names=self.feature_names,
                )
                self.eval(self.dval_subset, self.y_val[indices], "val_subset")

            if self.args.test:
                self.eval(self.dtest, self.y_test, "test")

    def eval(self, x: xgb.DMatrix, y: torch.Tensor, split: str) -> None:
        if self.args.task == Task.SURVIVAL:
            # Should not be called; SURVIVAL uses _eval_survival_split
            raise RuntimeError("Use _eval_survival_split for SURVIVAL task.")
        else:
            y_pred = torch.Tensor(self.bst.predict(x))
            metrics = evaluate_metrics(
                self.args,
                y_pred,
                y,
                split,
                self.metric_tracker,
            )
            # Optional: compute a Harrell-style c-index from classification probabilities
            # using the underlying survival times + censoring, if available.
            #extra = self._compute_c_index_from_probs(split, y_pred)
            #metrics.update(extra)
            self.logger.log_metrics(metrics)
    
    def _eval_survival_split(
        self,
        dmatrix: xgb.DMatrix,
        gt: torch.Tensor,          # [N,2] = [event, time], positive time means event, negative time means censoring
        split: str,
        survival_val_struct: np.ndarray,
    ) -> None:
        """
        Compute risk scores + survival probabilities and pass into evaluate_metrics
        to get the c-index using the existing SURVIVAL code in utils.py.
        Additionally derive horizon-based AUC from S(t|x).
        """
        # 1) get risk scores (raw margin): f(x) and set up time grid and baseline cumulative hazard H0(t) for this evaluation split (train/val/test)
        risk_scores = self.bst.predict(dmatrix, output_margin=True)  # [N]
        time_grid, baseline_times, baseline_cum_hazard = self._setup_survival_baseline(survival_val_struct["time"])

        # 2) survival probabilities S(t|x) on the precomputed time_grid
        surv_probs = self._predict_survival_probs(
            risk_scores=risk_scores,
            time_grid=time_grid,
            baseline_times=baseline_times,
            baseline_cum_hazard=baseline_cum_hazard,
        )  # [N, T]

        preds = {
            "risk_scores": torch.tensor(risk_scores, dtype=torch.float32),
            "survival_probs": torch.tensor(surv_probs, dtype=torch.float32),
        }

        # Core survival metrics: c_index_harrell, c_index_ipcw, c_index_antolini
        metrics = evaluate_metrics(
            self.args,
            preds=preds,
            gt=gt,  # [event, time], positive time means event, negative time means censoring
            split=split,
            metric_tracker=self.metric_tracker,
            time_grid=time_grid,
            survival_train=self.survival_train,
            survival_val=survival_val_struct,
        )

        # Horizon-based AUC (2y, 5y, 10y, 15y or from config)
        horizon_metrics = self._compute_horizon_classification_metrics(
            gt=gt,
            surv_probs=surv_probs,
            time_grid=time_grid,
            split=split,
        )
        metrics.update(horizon_metrics)

        # Cache artifacts so train.py can export test survival curves with predictions.
        split_view = {
            "train": self.train_view,
            "val": self.val_view,
            "test": self.test_view,
        }.get(split)
        if split_view is not None and "global_row" in split_view.obs.columns:
            global_row = split_view.obs["global_row"].to_numpy(dtype=np.int64)
        else:
            global_row = np.arange(gt.shape[0], dtype=np.int64)

        self.survival_export_cache[split] = {
            "risk_scores": np.asarray(risk_scores, dtype=float),
            "event": np.asarray(gt[:, 0].detach().cpu().numpy(), dtype=np.int64),
            "time": np.asarray(gt[:, 1].detach().cpu().numpy(), dtype=float),
            "global_row": np.asarray(global_row, dtype=np.int64),
            "survival_probs": np.asarray(surv_probs, dtype=float),
            "time_grid": np.asarray(time_grid, dtype=float),
        }

        self.logger.log_metrics(metrics)
    
    def _setup_survival_baseline(self, evaluation_times) -> None:
        """
            Compute time_grid and baseline cumulative hazard H0(t) from training data.
            The minimum and maximum grid times are derived from all evaluation times (events + censored) in the respective evaluation set (train/val/test).
        """
        # Get event times only from training data (exclude censored)
        train_event_times = self.train_time[self.train_event == 1].astype(float)
        # Get all times (events + censored) from validation data for grid bounds
        val_times = evaluation_times.astype(float)

        # Constrain to evaluation set range
        val_min = val_times.min()
        val_max = val_times.max()
        assert val_min <= val_max, f"Time grid bounds invalid: min={val_min}, max={val_max}."

        # Get candidate times from training events (5th to 95th percentile)
        use_percentiles = False
        if use_percentiles:
            print("Using 5th-95th percentile for time_grid candidate selection.")
            p5_train, p95_train = np.percentile(train_event_times, [5, 95])
            candidate_times = train_event_times[
                (train_event_times >= p5_train) &
                (train_event_times <= p95_train)
            ]
        else:
            candidate_times = train_event_times

        time_grid = candidate_times[
            (candidate_times >= val_min) &
            (candidate_times < val_max)
        ]
        time_grid = np.unique(time_grid)

        # Fallback if not enough times in range
        if len(time_grid) < 10:
            print("Using linear grid fallback for time_grid due to < 10 event times in range.")
            t_min = max(train_event_times.min(), val_min)
            t_max = min(train_event_times.max(), val_max)
            time_grid = np.linspace(t_min, t_max, 100, endpoint=False)

        # ----- Breslow baseline cumulative hazard H0(t) -----
        # linear predictor f(x)
        linpred = self.bst.predict(self.dtrain, output_margin=True)  # shape [N]
        hr = np.exp(linpred)  # hazard ratios

        times = self.train_time.astype(float)
        events = self.train_event.astype(int)

        order = np.argsort(times)
        times_ord = times[order]
        events_ord = events[order]
        hr_ord = hr[order]

        unique_event_times = np.unique(times_ord[events_ord == 1])
        cum_hazard = []
        H = 0.0

        # Loop over event times
        for t in unique_event_times:
            # Risk set: subjects at risk at time t (event time >= t)
            at_risk = times_ord >= t
            # Number of events at time t (sum excludes censored ones, as events_ord is 0 for censored times)
            d_k = events_ord[(times_ord == t)].sum()
            # Total risk score in the risk set
            R_k = hr_ord[at_risk].sum()
            # Breslow increment
            if R_k > 0 and d_k > 0:
                H += d_k / R_k
            cum_hazard.append(H)

        baseline_times = unique_event_times
        baseline_cum_hazard = np.array(cum_hazard, dtype=float)

        return time_grid, baseline_times, baseline_cum_hazard

    def _predict_survival_probs(
        self, risk_scores: np.ndarray, time_grid: np.ndarray, baseline_times: np.ndarray = None, baseline_cum_hazard: np.ndarray = None
    ) -> np.ndarray:
        """
        Given risk scores f(x) and baseline H0(t), compute S(t|x) on time_grid.
        risk_scores: [N] raw scores (output_margin=True)
        returns: [N, len(time_grid)] survival probabilities.
        """
        # If no events, S(t|x)=1 for all t
        if baseline_times is None or len(baseline_times) == 0:
            print("No events in training data; returning S(t|x)=1 for all samples and times.")
            return np.ones((risk_scores.shape[0], time_grid.shape[0]), dtype=float)

        # Interpolate cumulative baseline hazard on the grid
        H0_grid = np.interp(
            time_grid,
            baseline_times,
            baseline_cum_hazard,
            left=0.0,
            right=baseline_cum_hazard[-1],
        )
        hr = np.exp(risk_scores).astype(float)  # [N]
        # S_ij = exp( - H0_grid[j] * hr[i] )
        return np.exp(-np.outer(hr, H0_grid))  # [N, T]
    
    def _compute_horizon_classification_metrics(
        self,
        gt: torch.Tensor,          # [N,2] = [event, time]
        surv_probs: np.ndarray,    # [N, T] S(t|x) on time_grid
        time_grid: np.ndarray,     # [T]
        split: str,
    ) -> Dict[str, torch.Tensor]:
        """
        From survival curves S(t|x), derive horizon-based binary metrics.

        For each horizon H:
          - y_H = 1 if event time <= H (event by H)
          - y_H = 0 otherwise (including censored before/after H)
          - score = 1 - S(H|x) (event probability by horizon)
        """
        events = gt[:, 0].cpu().numpy().astype(int)   # 1=event, 0=censored
        times  = gt[:, 1].cpu().numpy().astype(float)

        # Horizons in years – can be configured in args, falls back to 2/5/10/15
        horizons = getattr(self.args, "eval_horizons", [2.0, 5.0, 10.0, 15.0])

        out: Dict[str, torch.Tensor] = {}

        for H in horizons:
            H = float(H)

            # Skip horizon if outside the time_grid support
            # Commented out to keep comparison consistent across models
            #if H < float(time_grid[0]) or H > float(time_grid[-1]):
            #    continue

            # index on time_grid closest to H
            idx = int(np.argmin(np.abs(time_grid - H)))  # 0 <= idx < T

            # Horizon-specific risk: P(event by H | x) = 1 - S(H|x)
            risk_H = 1.0 - surv_probs[:, idx]  # [N]
            
            # Labels at horizon H
            pos = (events == 1) & (times <= H)
            neg = ((events == 1) & (times > H)) | ((events == 0) & (times > H))

            # Exclude pre-baseline or weird entries
            eligible = (times >= 0.0)
            valid = eligible & (pos | neg)
            '''
            # Labels at horizon H – classification-style:
            #   pos = event by H
            #   neg = everyone else (including censored before H)
            eligible = times >= 0.0  # optional sanity filter
            pos = (events == 1) & (times <= H) & eligible

            valid = eligible  # we use all eligible samples
            '''
            
            if valid.sum() == 0:
                print(f"[Horizon metrics] split={split}, H={H}y: no valid samples for evaluation, skipping.")
                continue

            y_bin = pos[valid].astype(int)
            scores = risk_H[valid].astype(float)

            # Need both classes for AUC
            if y_bin.sum() == 0 or y_bin.sum() == y_bin.size:
                print(f"[Horizon metrics] split={split}, H={H}y: only one class present in valid samples (all {y_bin[0]}), skipping.")
                continue

            y_t = torch.tensor(y_bin, dtype=torch.int64)
            s_t = torch.tensor(scores, dtype=torch.float32)

            # AUC
            auc = auroc(s_t, y_t, task="binary")

            suffix = f"_{int(H)}y"
            out[f"{split}.auc{suffix}"]  = auc

        return out
    
    def _compute_c_index_from_probs(
        self,
        split: str,
        y_pred: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Harrell-style c-index from classification probabilities.

        Uses:
          - predicted risk scores (probabilities) from binary:logistic model
          - underlying time-to-event + censoring information from the views

        Assumes:
          - event/time columns are available in the underlying AnnData
          - event_col for c-index can be configured via args.cindex_event_col
            or defaults to args.target_outcomes[0].
        """
        # FIXME This doesn't work at all, as times are clipped in dataset/utils.py and the whole concept is a bit questionable

        # Map split -> corresponding view
        if split == "train":
            view = self.train_view
        elif split == "val":
            view = self.val_view
        elif split == "test":
            view = self.test_view
        else:
            # e.g. "val_subset" etc. -> skip
            return {}

        if view is None:
            return {}

        # Decide which event/time columns to use
        # By default, use args.cindex_event_col if provided, else target_outcomes[0].
        try:
            event_col = self.target[0]
            time_suffix = event_col.split("_")[1]
            time_col = f"time_{time_suffix}"

            times = view.obs[time_col].to_numpy().astype(float)
            # Interpret event_col as 1=event, 0=censored for c-index
            events = view.obs[event_col].to_numpy().astype(bool)
        except Exception as e:
            print(f"[c-index] split={split}: exception while preparing inputs: {e}")
            return {}

        # Convert predictions to numpy risk scores
        risk_scores = y_pred.detach().cpu().numpy().astype(float)

        # Compute Harrell c-index with censoring
        c_index = concordance_index_censored(
            events,
            times,
            risk_scores,
        )[0]

        return {
            f"{split}.c_index_from_prob": torch.tensor(c_index, dtype=torch.float32)
        }
    
    @staticmethod
    def _xgb_cat_feature_mask(view: npAnnData) -> np.ndarray:
        return (
            (view.var["value_type"] == "Categorical single")
            & (view.var["n_categorical_options"] > 2)
            & (view.var["possible_preprocessing"] != "replace zero with NaN")
        ).to_numpy()

    @staticmethod
    def _to_xgb_dataframe(view: npAnnData, feature_names: List[str]) -> pd.DataFrame:
        print(f"Creating dataframe for XGBoost")
        df = pd.DataFrame(view.X, columns=feature_names)
        print("Done with initial DataFrame creation")

        cat_mask = XGBoost._xgb_cat_feature_mask(view)
        cat_cols = [feature_names[i] for i, is_cat in enumerate(cat_mask) if is_cat]

        # Treat stored numeric codes as categories; keep NaNs as missing values.
        print("Casting categorical columns to 'category' dtype")
        for c in cat_cols:
            # Convert to pandas nullable integer (keeps NA), then to category.
            s = pd.to_numeric(df[c], errors="coerce")
            # If values are float-coded categories (e.g. 1.0, 2.0, NaN), make them integer-safe
            s = s.round()
            df[c] = s.astype("Int32").astype("category")
        print("Done creating dataframe for XGBoost")
        return df

    def _make_dmatrix(self, view: npAnnData, label: np.ndarray) -> xgb.DMatrix:
        if getattr(self.args, "xgb_use_imputation_onehot", True):
            return xgb.DMatrix(
                view.X,
                label=label,
                feature_names=self.feature_names,
            )

        # rawcat mode: pass a DataFrame with category dtypes
        df = XGBoost._to_xgb_dataframe(view, self.feature_names)
        print("Creating DMatrix with enable_categorical=True")
        dmatrix = xgb.DMatrix(
            df,
            label=label,
            feature_names=self.feature_names,
            enable_categorical=True,
        )
        print("Done creating DMatrix")
        return dmatrix

    def calculate_permutation_importance(self, split: str = "val") -> pd.DataFrame:
        """
        Calculate and log permutation importance for the specified data split.

        Args:
            split: Data split to use ('train', 'val', or 'test')

        Returns:
            DataFrame containing the permutation importance results
        """
        # Select appropriate data split
        if split == "train":
            X, y = self.dtrain.get_data(), self.y_train
        elif split == "val":
            X, y = self.dval.get_data(), self.y_val
        elif split == "test":
            X, y = self.dtest.get_data(), self.y_test
        else:
            raise ValueError("Split must be one of: 'train', 'val', 'test'")

        # Calculate permutation importance
        results_df = calculate_permutation_importance(
            self, X, y, self.feature_names, n_repeats=3
        )

        # Log to wandb
        self.logger.log_table(
            key=f"permutation_importance_{split}", dataframe=results_df
        )

        # Store in feature importance history
        if "permutation" not in self.feature_importance_history:
            self.feature_importance_history["permutation"] = []
        self.feature_importance_history["permutation"].append(
            dict(zip(results_df.feature, results_df.importance_mean))
        )

        return results_df


class GetBestModel(xgb.callback.TrainingCallback):
    def __init__(self, model: XGBoost, direction: str):
        self.model = model
        self.direction = direction
        self.best_score = float("inf") if direction == "minimize" else float("-inf")
        self.best_model = None

    def after_iteration(self, model: xgb.Booster, epoch, evals_log: Dict):
        score = evals_log["val"][self.model.main_metric][-1]
        if self.direction == "minimize":
            if score < self.best_score:
                self.best_score = score
                self.best_model = model.copy()
        else:
            if score > self.best_score:
                self.best_score = score
                self.best_model = model.copy()
        return False # Don't stop training


def calculate_permutation_importance(
    model: "XGBoost",
    X: np.ndarray,
    y: torch.Tensor,
    feature_names: List[str],
    n_repeats: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """
    Calculate permutation importance for features in a trained XGBoost model.

    Args:
        model: Trained XGBoost model instance
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        n_repeats: Number of times to repeat the permutation process
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with mean importance scores and their std deviation
        Dictionary containing raw importance scores for each feature
    """
    np.random.seed(random_state)

    # Create DMatrix for prediction
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)

    # Get baseline score
    baseline_pred = torch.Tensor(model.bst.predict(dmatrix))
    if model.args.task == Task.REGRESSION:
        baseline_score = mean_absolute_error(baseline_pred, y)
    else:  # classification
        baseline_score = auroc(baseline_pred, y.to(torch.int), task="binary")

    # Store importance scores for each feature
    importance_scores: Dict[str, List[float]] = {feat: [] for feat in feature_names}

    # For each feature
    for feature in tqdm.tqdm(feature_names):
        # Repeat the permutation process n_repeats times
        for _ in range(n_repeats):
            # Create a copy of the feature matrix
            X_permuted = X.copy()

            # Permute the feature
            perm_idx = np.random.permutation(X.shape[0])
            X_permuted[:, feature_names.index(feature)] = X_permuted[
                perm_idx, feature_names.index(feature)
            ]

            # Create new DMatrix with permuted feature
            dmatrix_permuted = xgb.DMatrix(X_permuted, feature_names=feature_names)

            # Get predictions with permuted feature
            permuted_pred = torch.Tensor(model.bst.predict(dmatrix_permuted))

            # Calculate score with permuted feature
            if model.args.task == Task.REGRESSION:
                permuted_score = mean_absolute_error(permuted_pred, y)
                # For regression, higher difference means more important
                importance = permuted_score - baseline_score
            else:  # classification
                permuted_score = auroc(permuted_pred, y.to(torch.int), task="binary")
                # For classification, lower permuted AUC means more important
                importance = baseline_score - permuted_score

            importance_scores[feature].append(float(importance))

    # Calculate mean and std of importance scores
    mean_imp = {feat: np.mean(scores) for feat, scores in importance_scores.items()}
    std_imp = {feat: np.std(scores) for feat, scores in importance_scores.items()}

    # Create DataFrame with results
    results_df = pd.DataFrame(
        {
            "feature": list(mean_imp.keys()),
            "importance_mean": list(mean_imp.values()),
            "importance_std": list(std_imp.values()),
        }
    )

    for r in range(n_repeats):
        results_df[f"importance_{r}"] = [
            scores[r] for scores in importance_scores.values()
        ]

    # Sort by absolute importance value
    results_df = results_df.reindex(
        results_df.importance_mean.abs().sort_values(ascending=False).index
    )

    return results_df

    
