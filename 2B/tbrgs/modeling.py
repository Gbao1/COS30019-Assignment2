from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import MinMaxScaler
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn is required for Assignment 2B models.") from exc


@dataclass
class SiteModels:
    lstm_model: Any
    gru_model: Any
    rf_model: RandomForestRegressor
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler


def _build_lstm(input_steps: int, output_steps: int, units: int):
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Input, LSTM
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:  # pragma: no cover
        raise ImportError("TensorFlow is required to train LSTM/GRU models.") from exc

    model = Sequential([
        Input(shape=(input_steps, 1)),
        LSTM(units),
        Dense(output_steps),
    ])
    model.compile(optimizer=Adam(), loss="mse")
    return model


def _build_gru(input_steps: int, output_steps: int, units: int):
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, GRU, Input
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:  # pragma: no cover
        raise ImportError("TensorFlow is required to train LSTM/GRU models.") from exc

    model = Sequential([
        Input(shape=(input_steps, 1)),
        GRU(units),
        Dense(output_steps),
    ])
    model.compile(optimizer=Adam(), loss="mse")
    return model


def _prepare_scalers(x_train: np.ndarray, y_train: np.ndarray) -> tuple[MinMaxScaler, MinMaxScaler]:
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x_scaler.fit(x_train)
    y_scaler.fit(y_train)
    return x_scaler, y_scaler


def _scale_xy(x: np.ndarray, y: np.ndarray, x_scaler: MinMaxScaler, y_scaler: MinMaxScaler) -> tuple[np.ndarray, np.ndarray]:
    x_s = x_scaler.transform(x)
    y_s = y_scaler.transform(y)
    return x_s, y_s


def train_site_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    lstm_units: int = 64,
    gru_units: int = 64,
    rf_estimators: int = 300,
    epochs: int = 10,
    batch_size: int = 32,
    random_seed: int = 42,
) -> SiteModels:
    if len(x_train) == 0:
        raise ValueError("x_train is empty. Not enough samples for training.")

    output_steps = y_train.shape[1]
    x_scaler, y_scaler = _prepare_scalers(x_train, y_train)
    x_train_s, y_train_s = _scale_xy(x_train, y_train, x_scaler, y_scaler)
    x_val_s, y_val_s = _scale_xy(x_val, y_val, x_scaler, y_scaler) if len(x_val) else (x_val, y_val)

    x_train_dl = x_train_s.reshape((x_train_s.shape[0], x_train_s.shape[1], 1))
    x_val_dl = x_val_s.reshape((x_val_s.shape[0], x_val_s.shape[1], 1)) if len(x_val_s) else x_val_s

    lstm_model = _build_lstm(input_steps=x_train.shape[1], output_steps=output_steps, units=lstm_units)
    lstm_model.fit(
        x_train_dl,
        y_train_s,
        validation_data=(x_val_dl, y_val_s) if len(x_val_dl) else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    gru_model = _build_gru(input_steps=x_train.shape[1], output_steps=output_steps, units=gru_units)
    gru_model.fit(
        x_train_dl,
        y_train_s,
        validation_data=(x_val_dl, y_val_s) if len(x_val_dl) else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    rf_model = RandomForestRegressor(n_estimators=rf_estimators, random_state=random_seed)
    rf_model.fit(x_train_s, y_train_s.ravel() if output_steps == 1 else y_train_s)

    return SiteModels(
        lstm_model=lstm_model,
        gru_model=gru_model,
        rf_model=rf_model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )


def predict_scaled(models: SiteModels, x: np.ndarray, model_name: str) -> np.ndarray:
    x_s = models.x_scaler.transform(x)
    key = model_name.lower()
    if key == "lstm":
        pred_s = models.lstm_model.predict(x_s.reshape((x_s.shape[0], x_s.shape[1], 1)), verbose=0)
    elif key == "gru":
        pred_s = models.gru_model.predict(x_s.reshape((x_s.shape[0], x_s.shape[1], 1)), verbose=0)
    elif key in {"rf", "random_forest", "randomforest"}:
        pred_s = models.rf_model.predict(x_s)
        if pred_s.ndim == 1:
            pred_s = pred_s.reshape(-1, 1)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return pred_s


def predict_unscaled(models: SiteModels, x: np.ndarray, model_name: str) -> np.ndarray:
    pred_s = predict_scaled(models=models, x=x, model_name=model_name)
    return models.y_scaler.inverse_transform(pred_s)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_f = y_true.ravel()
    y_pred_f = y_pred.ravel()
    mae = float(mean_absolute_error(y_true_f, y_pred_f))
    rmse = float(np.sqrt(mean_squared_error(y_true_f, y_pred_f)))

    non_zero = np.where(np.abs(y_true_f) > 1e-8, np.abs(y_true_f), np.nan)
    mape = float(np.nanmean(np.abs((y_true_f - y_pred_f) / non_zero)) * 100.0)
    if np.isnan(mape):
        mape = 0.0

    return {"mae": mae, "rmse": rmse, "mape": mape}
