# main.py
import argparse
import os
import numpy as np
from trainer import Trainer
from data import make_tabular_classification, make_sequence_dataset
from utils import set_seeds
from config import MLFLOW_TRACKING_URI

# Use environment variable first; fallback to config
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI is None:
    from app.config import MLFLOW_TRACKING_URI as DEFAULT_URI
    MLFLOW_TRACKING_URI = DEFAULT_URI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['rf', 'fnn', 'lstm'], default='rf', help='Model to train')
    args = parser.parse_args()

    set_seeds(42)

    trainer = Trainer(tracking_uri=MLFLOW_TRACKING_URI)

    if args.model == 'rf':
        X_train, X_val, X_test, y_train, y_val, y_test = make_tabular_classification()
        X = np.concatenate([X_train, X_val, X_test])
        y = np.concatenate([y_train, y_val, y_test])
        trainer.tune_random_forest(X, y)

    elif args.model == 'fnn':
        X_train, X_val, X_test, y_train, y_val, y_test = make_tabular_classification()
        X = np.concatenate([X_train, X_val, X_test])
        y = np.concatenate([y_train, y_val, y_test])
        trainer.tune_fnn(X, y)

    elif args.model == 'lstm':
        X_train, X_val, X_test, y_train, y_val, y_test = make_sequence_dataset()
        X = np.concatenate([X_train, X_val, X_test])
        y = np.concatenate([y_train, y_val, y_test])
        trainer.tune_lstm(X, y)

if __name__ == "__main__":
    main()
