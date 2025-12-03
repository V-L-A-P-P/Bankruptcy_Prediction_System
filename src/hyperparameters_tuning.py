from abc import ABC, abstractmethod
from config.config import N_CV_SPLITS
from sklearn.model_selection import StratifiedKFold
import optuna
from catboost.utils import get_gpu_device_count
from sklearn.metrics import average_precision_score, balanced_accuracy_score
import catboost

class HyperparameterTuner(ABC):
    def __init__(self):
        self.cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=42)

    @abstractmethod
    def set_to_gpu(self, params):
        pass

    @abstractmethod
    def param_space(self, trial):
        pass

    @abstractmethod
    def cv_score(self, params, X, y, cv, metric='balanced_accuracy'):
        pass

    def objective(self, trial, X, y, metric='balanced_accuracy'):
        params = self.param_space(trial)
        score = self.cv_score(params=params,
                                  X=X,
                                  y=y,
                                  cv=self.cv,
                              metric=metric)
        return score

    def start_tuning(self, X, y, n_trials=10, metric='balanced_accuracy', direction='maximize'):
        study = optuna.create_study(direction=direction)
        study.optimize(lambda trial: self.objective(trial, X, y, metric), n_trials=n_trials)
        return study

class CatboostHyperparameterTuner(HyperparameterTuner):
    def __init__(self):
        super().__init__()

    def set_to_gpu(self, params):
        has_gpu = get_gpu_device_count() > 0
        params["task_type"] = "GPU" if has_gpu else "CPU"
        params["devices"] = "0" if has_gpu else ""
        return params

    def param_space(self, trial):
        params = {
            "iterations": trial.suggest_int("iterations", 800, 1800),
            "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.03, log=True),
            "depth": trial.suggest_int("depth", 3, 6),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 150, 350),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 15.0),
            "border_count": trial.suggest_int("border_count", 32, 128),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.2),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
            "eval_metric": "Logloss",
            "loss_function": "Logloss",
            "verbose": 250
        }
        params = self.set_to_gpu(params)

        return params

    def cv_score(self, params, X, y, cv, metric='balanced_accuracy'):
        scores = []

        for train_idx, val_idx in cv.split(X, y):

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = catboost.CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=100)

            if metric == "average_precision":
                y_val_proba = model.predict_proba(X_val)[:, 1]
                score = average_precision_score(y_val, y_val_proba)
            elif metric == "balanced_accuracy":
                y_val_pred = model.predict(X_val)
                score = balanced_accuracy_score(y_val, y_val_pred)
            else:
                raise ValueError("Unknown metric")
            scores.append(score)
        score = sum(scores) / len(scores)
        return score