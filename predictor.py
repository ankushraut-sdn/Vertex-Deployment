"""Google Vertex AI Custom Prediction Routine — Predictor class."""
import os, json, logging
import joblib, pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXPECTED_FEATURES = [
    "age", "employment_type", "years_of_service",
    "weekly_hours", "department", "salary_band", "has_dependents",
]


class Predictor:
    def __init__(self):
        self.model = self.preprocessor = self.label_map = None

    def load(self, artifacts_uri: str):
        self.model        = joblib.load(os.path.join(artifacts_uri, "model.pkl"))
        self.preprocessor = joblib.load(os.path.join(artifacts_uri, "preprocessor.joblib"))
        with open(os.path.join(artifacts_uri, "label_map.json")) as f:
            self.label_map = json.load(f)
        logger.info("Model loaded from %s", artifacts_uri)

    def preprocess(self, prediction_input: dict) -> pd.DataFrame:
        instances = prediction_input.get("instances", prediction_input)
        if isinstance(instances, dict):
            instances = [instances]
        df = pd.DataFrame(instances)
        missing = set(EXPECTED_FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return df[EXPECTED_FEATURES]

    def predict(self, instances: pd.DataFrame) -> dict:
        X = self.preprocessor.transform(instances)
        return {
            "raw_predictions": self.model.predict(X).tolist(),
            "probabilities":   self.model.predict_proba(X).tolist(),
        }

    def postprocess(self, prediction_results: dict) -> dict:
        label_order = [self.label_map[k] for k in sorted(self.label_map)]
        labeled     = [self.label_map[str(p)] for p in prediction_results["raw_predictions"]]
        return {"predictions": [
            {"label": lbl, "probability": max(prob),
             "all_proba": dict(zip(label_order, prob))}
            for lbl, prob in zip(labeled, prediction_results["probabilities"])
        ]}
