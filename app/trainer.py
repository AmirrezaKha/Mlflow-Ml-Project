# app/trainer.py
import json
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import itertools
import random
import tensorflow as tf

from config import EXPERIMENT_NAME, RF_TRIALS, FNN_TRIALS, LSTM_TRIALS
from models import build_random_forest, build_fnn, build_lstm
from data import make_tabular_classification, make_sequence_dataset
from utils import log_common_artifacts, set_seeds


class Trainer:
    def __init__(self, tracking_uri=None, experiment_name=EXPERIMENT_NAME):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def tune_random_forest(self, X, y):
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )

        param_grid = {
            'n_estimators': [100, 200, 400],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None],
        }

        keys = list(param_grid.keys())
        combos = [dict(zip(keys, vals)) for vals in itertools.islice(itertools.product(*param_grid.values()), RF_TRIALS)]
        random.seed(42)

        with mlflow.start_run(run_name='RF_parent'):
            mlflow.set_tag('model_family', 'RandomForest')
            best = {'f1_macro': -1.0, 'params': None}

            for i, params in enumerate(combos):
                with mlflow.start_run(run_name=f'RF_trial_{i+1}', nested=True):
                    mlflow.log_params(params)
                    clf = build_random_forest({**params, 'seed': 42})
                    clf.fit(X_train, y_train)

                    y_val_pred = clf.predict(X_val)
                    log_common_artifacts(mlflow.active_run(), y_val, y_val_pred, f'rf_trial_{i+1}', ['class_0', 'class_1'])
                    mlflow.sklearn.log_model(clf, artifact_path='model')

                    f1m = f1_score(y_val, y_val_pred, average='macro')
                    if f1m > best['f1_macro']:
                        best.update({'f1_macro': f1m, 'params': params})

            # Retrain best
            mlflow.log_param('best_params', json.dumps(best['params']))
            best_clf = build_random_forest({**best['params'], 'seed': 42})
            best_clf.fit(np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
            y_test_pred = best_clf.predict(X_test)
            log_common_artifacts(mlflow.active_run(), y_test, y_test_pred, 'rf_best_test', ['class_0', 'class_1'])
            mlflow.sklearn.log_model(best_clf, artifact_path='best_model')

    def tune_fnn(self, X, y):
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Flatten param grid into individual combinations
        param_grid = [
            {'hidden_units': [64, 128], 'dropout': [0.2, 0.3], 'batch_size': [32, 64]},
            {'hidden_units': [32, 64], 'dropout': [0.1, 0.2], 'batch_size': [16, 32]},
        ]

        combos = []
        for grid in param_grid:
            keys = list(grid.keys())
            values = list(grid.values())
            for vals in itertools.product(*values):
                combos.append(dict(zip(keys, vals)))
        random.shuffle(combos)

        with mlflow.start_run(run_name='FNN_parent'):
            mlflow.set_tag('model_family', 'FeedForwardNN')
            best = {'f1_macro': -1.0, 'params': None}

            for i, params in enumerate(combos[:FNN_TRIALS]):
                with mlflow.start_run(run_name=f'FNN_trial_{i+1}', nested=True):
                    mlflow.log_params(params)

                    # Build model
                    model = build_fnn(
                        input_dim=X.shape[1],
                        hidden_units=params['hidden_units'],
                        dropout=params['dropout']
                    )

                    model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=30,
                        batch_size=params['batch_size'],
                        verbose=0
                    )

                    y_val_pred = np.argmax(model.predict(X_val), axis=1)
                    f1m = f1_score(y_val, y_val_pred, average='macro')
                    mlflow.log_metric('f1_macro', f1m)
                    log_common_artifacts(mlflow.active_run(), y_val, y_val_pred, f'fnn_trial_{i+1}', ['class_0', 'class_1'])

                    mlflow.tensorflow.log_model(model, artifact_path='model')
                    if f1m > best['f1_macro']:
                        best.update({'f1_macro': f1m, 'params': params})

            # Retrain best model
            mlflow.log_param('best_params', json.dumps(best['params']))
            best_model = build_fnn(
                input_dim=X.shape[1],
                hidden_units=best['params']['hidden_units'],
                dropout=best['params']['dropout']
            )
            best_model.fit(
                np.concatenate([X_train, X_val]),
                np.concatenate([y_train, y_val]),
                epochs=30,
                batch_size=best['params']['batch_size'],
                verbose=0
            )

            y_test_pred = np.argmax(best_model.predict(X_test), axis=1)
            log_common_artifacts(mlflow.active_run(), y_test, y_test_pred, 'fnn_best_test', ['class_0', 'class_1'])
            mlflow.tensorflow.log_model(best_model, artifact_path='best_model')

    def tune_lstm(self, X, y):
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )

        param_grid = [
            {'lstm_units': 32, 'dense_units': 16, 'dropout': 0.2, 'batch_size': 32},
            {'lstm_units': 64, 'dense_units': 32, 'dropout': 0.3, 'batch_size': 64},
        ]

        with mlflow.start_run(run_name='LSTM_parent'):
            mlflow.set_tag('model_family', 'LSTM')
            best = {'f1_macro': -1.0, 'params': None}

            for i, params in enumerate(param_grid[:LSTM_TRIALS]):
                with mlflow.start_run(run_name=f'LSTM_trial_{i+1}', nested=True):
                    mlflow.log_params(params)

                    model = build_lstm(
                        time_steps=X.shape[1],
                        n_features=X.shape[2],
                        lstm_units=params['lstm_units'],
                        dense_units=params['dense_units'],
                        dropout=params['dropout']
                    )

                    model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=30,
                        batch_size=params['batch_size'],
                        verbose=0
                    )

                    y_val_pred = np.argmax(model.predict(X_val), axis=1)
                    f1m = f1_score(y_val, y_val_pred, average='macro')
                    mlflow.log_metric('f1_macro', f1m)
                    log_common_artifacts(mlflow.active_run(), y_val, y_val_pred, f'lstm_trial_{i+1}', ['class_0', 'class_1'])

                    mlflow.tensorflow.log_model(model, artifact_path='model')
                    if f1m > best['f1_macro']:
                        best.update({'f1_macro': f1m, 'params': params})

            # Retrain best
            mlflow.log_param('best_params', json.dumps(best['params']))
            best_model = build_lstm(
                time_steps=X.shape[1],
                n_features=X.shape[2],
                lstm_units=best['params']['lstm_units'],
                dense_units=best['params']['dense_units'],
                dropout=best['params']['dropout']
            )
            best_model.fit(
                np.concatenate([X_train, X_val]),
                np.concatenate([y_train, y_val]),
                epochs=30,
                batch_size=best['params']['batch_size'],
                verbose=0
            )

            y_test_pred = np.argmax(best_model.predict(X_test), axis=1)
            log_common_artifacts(mlflow.active_run(), y_test, y_test_pred, 'lstm_best_test', ['class_0', 'class_1'])
            mlflow.tensorflow.log_model(best_model, artifact_path='best_model')
