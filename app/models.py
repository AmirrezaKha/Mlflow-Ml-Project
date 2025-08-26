# app/models.py
from typing import List, Union
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers


def build_random_forest(params):
    return RandomForestClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', None),
        min_samples_split=params.get('min_samples_split', 2),
        max_features=params.get('max_features', 'sqrt'),
        random_state=params.get('seed', 42),
        n_jobs=-1,
    )


def build_fnn(
    input_dim: int,
    hidden_units: Union[int, List[int]],
    dropout: float = 0.0,
    num_classes: int = 2
):
    # Ensure hidden_units is a list
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for h in hidden_units:
        x = layers.Dense(h, activation='relu')(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_lstm(
    time_steps: int,
    n_features: int,
    lstm_units: int,
    dense_units: int,
    dropout: float,
    num_classes: int = 2
):
    inputs = keras.Input(shape=(time_steps, n_features))
    x = layers.LSTM(lstm_units, return_sequences=False)(inputs)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    if dense_units > 0:
        x = layers.Dense(dense_units, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
