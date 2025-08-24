# app/config.py
from pathlib import Path


RNG_SEED = 42
BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MLFLOW_TRACKING_URI = "http://mlflow-service:5000" # points to k8s service
EXPERIMENT_NAME = "RF_vs_FNN_vs_LSTM_OOP"


# tuning knobs (small for quick runs)
RF_TRIALS = 6
FNN_TRIALS = 4
LSTM_TRIALS = 4


# docker/k8s-friendly default
MLRUNS_ROOT = "/mlflow/mlruns"