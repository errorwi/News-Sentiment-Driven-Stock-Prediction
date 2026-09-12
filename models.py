from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def train_models(X_train, y_train, X_test, y_test):
    models = {
        "LogReg": LogisticRegression(),
        "RF": RandomForestClassifier(n_estimators=200, max_depth=5)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = (model, acc)

    return results


def create_sequences(df, feature_cols, target_col='return', window_size=5):
    """Create time-series sequences from a DataFrame.

    - df: pandas DataFrame sorted chronologically (older rows first).
    - feature_cols: list of column names to use as input features (e.g. ['sentiment','return','volume']).
    - target_col: column name containing the daily return used to compute next-day direction.
    - window_size: number of past days per sequence.

    Returns: (X, y)
    - X: numpy array shape (n_samples, window_size, n_features)
    - y: numpy array shape (n_samples,) with binary labels (1 if next-day return > 0)
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df = df.copy()
    df = df.sort_index()

    # Target is next-day return direction (1 if next-day return > 0)
    # If df[target_col] yields a DataFrame (e.g., duplicate column names),
    # convert to a Series by taking the first column to avoid assignment errors.
    target_series = df[target_col]
    if isinstance(target_series, pd.DataFrame):
        target_series = target_series.iloc[:, 0]
    target_series = target_series.shift(-1)
    df['__target_next'] = (target_series > 0).astype(int)

    # Drop rows with NaN in features or target
    df = df.dropna(subset=feature_cols + ['__target_next'])

    values = df[feature_cols].values
    targets = df['__target_next'].values

    X, y = [], []
    n_rows = len(df)

    # For window starting at i, inputs are rows [i .. i+window_size-1]
    # The target is next-day relative to the last row in the window, which
    # is stored at index i+window_size-1 in '__target_next' (shifted earlier).
    for i in range(0, n_rows - window_size + 1):
        end = i + window_size
        seq_x = values[i:end]
        seq_y = targets[end - 1]
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)
    return X, y


def build_lstm_model(input_shape, learning_rate=0.001):
    """Builds the LSTM model per requirements.

    Architecture:
    - LSTM(64, return_sequences=True)
    - Dropout(0.2)
    - LSTM(32)
    - Dropout(0.2)
    - Dense(1, activation='sigmoid')
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model


def train_lstm_model(model, X_train, y_train, X_val, y_val,
                     epochs=100, batch_size=32, patience=5):
    """Train the LSTM model without shuffling (chronological).

    Returns: trained model and the Keras history object.
    """
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        shuffle=False
    )
    return model, history


def predict_and_evaluate(model, X, y_true):
    """Predict probabilities and evaluate accuracy + AUC.

    Returns a dict with `accuracy`, `auc`, `y_proba`, and `y_pred`.
    """
    y_proba = model.predict(X).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = None
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = None

    return {
        'accuracy': acc,
        'auc': auc,
        'y_proba': y_proba,
        'y_pred': y_pred
    }
