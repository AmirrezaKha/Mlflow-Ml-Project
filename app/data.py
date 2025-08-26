# app/data.py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

RNG = 42


def make_tabular_classification(n_samples=2500, n_features=20, n_informative=10, n_classes=2):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        random_state=RNG,
        class_sep=1.2,
        flip_y=0.01,
    )

    # split into train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=RNG)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RNG)

    return X_train.astype('float32'), X_val.astype('float32'), X_test.astype('float32'), \
           y_train.astype('int32'), y_val.astype('int32'), y_test.astype('int32')


def make_sequence_dataset(n_samples=3000, time_steps=30, n_classes=2):
    rng = np.random.default_rng(RNG)
    X = np.zeros((n_samples, time_steps, 1), dtype='float32')
    y = np.zeros((n_samples,), dtype='int32')
    t = np.linspace(0, 2*np.pi, time_steps)
    
    for i in range(n_samples):
        cls = rng.integers(0, n_classes)
        phase = 0.0 if cls == 0 else np.pi/3
        noise_scale = 0.05 if cls == 0 else 0.15
        seq = np.sin(t + phase) + rng.normal(0, noise_scale, size=time_steps)
        X[i, :, 0] = seq.astype('float32')
        y[i] = cls

    # Split: train 64%, val 16%, test 20%
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=RNG, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=RNG, stratify=y_train_full)

    return X_train, X_val, X_test, y_train, y_val, y_test
