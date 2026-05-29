"""
LSTM model for traffic flow prediction.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import time

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from .base_model import BaseTrafficModel

class LSTMTrafficModel(BaseTrafficModel):
    """LSTM-based traffic flow prediction model."""

    def __init__(self, units: int = 64, dropout: float = 0.2, **kwargs):
        """
        Initialize LSTM model.

        Args:
            units: Number of LSTM units
            dropout: Dropout rate
            **kwargs: Additional model parameters
        """
        super().__init__("LSTM", units=units, dropout=dropout, **kwargs)

        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model. Please install: pip install tensorflow")

        self.units = units
        self.dropout = dropout
        self.scaler = MinMaxScaler()
        self.history = None

    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> tf.keras.Model:
        """
        Build LSTM model architecture.

        Args:
            input_shape: Shape of input data (sequence_length, features)

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(self.units // 2, return_sequences=False),
            Dropout(self.dropout),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def _preprocess_data(self, X: np.ndarray, y: np.ndarray = None, fit_scaler: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess data with scaling."""
        # Reshape for scaling if necessary
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])

        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)

        X_scaled = X_scaled.reshape(original_shape)

        if y is not None:
            # For target values, we'll use a simple normalization
            y_scaled = y / np.max(y) if np.max(y) > 0 else y
            return X_scaled, y_scaled

        return X_scaled, None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32, patience: int = 10,
              **kwargs) -> Dict[str, Any]:
        """
        Train the LSTM model.

        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data (optional)
            y_val: Validation target data (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience

        Returns:
            Training history dictionary
        """
        print(f"Training {self.model_name} model...")
        start_time = time.time()

        # Preprocess data
        X_train_scaled, y_train_scaled = self._preprocess_data(X_train, y_train, fit_scaler=True)

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val_scaled = self._preprocess_data(X_val, y_val)
            validation_data = (X_val_scaled, y_val_scaled)

        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)

        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if validation_data else 'loss',
                         patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss',
                            factor=0.5, patience=patience//2, min_lr=1e-7)
        ]

        # Train model
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )

        self.training_time = time.time() - start_time
        self.is_trained = True
        self.history = history.history

        print(f"{self.model_name} training completed in {self.training_time:.2f} seconds")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Input data for prediction

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Preprocess input
        X_scaled, _ = self._preprocess_data(X)

        # Make predictions
        predictions_scaled = self.model.predict(X_scaled, verbose=0)

        # Denormalize predictions (simple approach)
        # Note: In a real scenario, you'd want to store the target scaling parameters
        predictions = predictions_scaled * np.max(X)  # Approximate denormalization

        return predictions

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet"

        import io
        import contextlib

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.model.summary()
        return f.getvalue()

    def save_model(self, filepath: str) -> None:
        """Save the trained model including scaler."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        # Save the Keras model
        model_path = filepath.replace('.pkl', '.h5')
        self.model.save(model_path)

        # Save additional data
        import pickle
        model_data = {
            'model_name': self.model_name,
            'model_params': self.model_params,
            'training_time': self.training_time,
            'is_trained': self.is_trained,
            'scaler': self.scaler,
            'history': self.history
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"LSTM model saved to {model_path} and {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model including scaler."""
        # Load the Keras model
        model_path = filepath.replace('.pkl', '.h5')
        self.model = tf.keras.models.load_model(model_path)

        # Load additional data
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model_name = model_data.get('model_name', self.model_name)
        self.model_params = model_data.get('model_params', self.model_params)
        self.training_time = model_data.get('training_time', 0)
        self.is_trained = model_data.get('is_trained', True)
        self.scaler = model_data.get('scaler', MinMaxScaler())
        self.history = model_data.get('history', None)

        print(f"LSTM model loaded from {model_path} and {filepath}")