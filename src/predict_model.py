import pandas as pd
from config.config import MODELS_DIR, RAW_DATA_DIR
import pickle
import matplotlib.pyplot as plt

import shap
import pandas as pd

class Predictor:
    def __init__(self):
        self.preprocessor = self.load_preprocessor()
        self.model = self.load_model()
        self.calibrated_model = self.load_calibrated_model()

    def load_preprocessor(self):
        try:
            with open(MODELS_DIR + "/preprocessor.pkl", "rb") as f:
                return pickle.load(f)
        except:
            raise Exception("Train model at first!")

    def load_model(self):
        try:
            with open(MODELS_DIR + "/model.pkl", "rb") as f:
                return pickle.load(f)
        except:
            raise Exception("Train model at first!")

    def load_calibrated_model(self):
        try:
            with open(MODELS_DIR + "/calibrated_model.pkl", "rb") as f:
                return pickle.load(f)
        except:
            raise Exception("Train model at first!")

    def predict(self, X, proba=True):

        X = self.preprocessor.preprocess_pipeline(X, mode='eval')

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(X)
        shap.plots.waterfall(shap_values[0]) # only for first prediction
        plt.tight_layout()
        plt.savefig("waterfall_interpretation.png")

        if proba:
            return self.calibrated_model.predict_proba(X)[:,1]
        else:
            return self.calibrated_model.predict(X)


if __name__ == "__main__":
    import matplotlib

    print(matplotlib.get_backend())

    X = pd.read_csv(RAW_DATA_DIR+'/test.csv')
    model = Predictor()
    print(model.predict(X))