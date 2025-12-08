import pandas as pd
import numpy as np
from pathlib import Path

from src.data_preprocessing import DataPreprocessor
from src.train_model import ModelTrainer
from src.hyperparameters_tuning import CatboostHyperparameterTuner
from src.predict_model import Predictor
from src.constants import REQUIRED_COLUMNS, TARGET_COLUMN


def test_full_pipeline(monkeypatch):
    class DummyPreprocessor:
        def preprocess_pipeline(self, X, mode, zero_var_columns=None):
            return X

    class DummyModel:
        def predict(self, X):
            return [1] * len(X)

    monkeypatch.setattr("src.predict_model.Predictor.load_preprocessor", lambda self: DummyPreprocessor())
    monkeypatch.setattr("src.predict_model.Predictor.load_model", lambda self: DummyModel())

    df = pd.DataFrame({
        col: np.random.rand(40) for col in REQUIRED_COLUMNS
    })
    df[TARGET_COLUMN] = np.random.randint(0, 2, 40)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    pre = DataPreprocessor()
    tuner = CatboostHyperparameterTuner()
    trainer = ModelTrainer(pre, tuner)

    model, preprocessor, calibrated_model = trainer.train(
        X,
        y,
        model_name="catboost",
        params={"iterations": 5, "verbose": False}
    )

    file_path_model = Path("models/model.pkl")
    file_path_preprocessor = Path("models/preprocessor.pkl")
    assert file_path_model.exists()
    assert file_path_preprocessor.exists()

    predictor = Predictor()

    df_eval = df.drop(columns=[TARGET_COLUMN]).iloc[:5]
    preds = predictor.predict(df_eval)

    assert len(preds) == 5
