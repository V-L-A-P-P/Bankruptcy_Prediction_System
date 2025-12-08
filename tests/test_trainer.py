import pandas as pd
import numpy as np
from src.train_model import ModelTrainer
from src.data_preprocessing import DataPreprocessor
from src.hyperparameters_tuning import CatboostHyperparameterTuner
from src.constants import REQUIRED_COLUMNS, TARGET_COLUMN
from pathlib import Path

def test_trainer_runs_without_errors(tmp_path):
    # формируем корректный DataFrame с обязательными колонками
    df = pd.DataFrame({
        col: np.random.rand(50) for col in REQUIRED_COLUMNS
    })
    df[TARGET_COLUMN] = np.random.randint(0, 2, 50)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    pre = DataPreprocessor()
    tuner = CatboostHyperparameterTuner()

    trainer = ModelTrainer(pre, tuner)

    # минимальный набор параметров, чтобы модель обучилась
    params = {"iterations": 5, "verbose": False}

    model, preprocessor, calibrator = trainer.train(X, y, model_name="catboost", params=params)

    assert model is not None
    assert hasattr(model, "predict")
    assert preprocessor.zero_var_columns is not None
    file_path_model = Path("models/model.pkl")
    file_path_preprocessor = Path("models/preprocessor.pkl")
    assert file_path_model.exists()
    assert file_path_preprocessor.exists()

