"""
Base model interface for traffic flow prediction models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple
import pickle
import time

class BaseTrafficModel(ABC):
    """Abstract base class for all traffic prediction models."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the base model.

        Args:
            model_name: Name of the model
            **kwargs: Model-specific parameters
        """
        self.model_name = model_name
        self.model_params = kwargs
        self.model = None
        self.is_trained = False
        self.training_time = 0
        self.inference_time = 0

    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> Any:
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'model_params': self.model_params,
            'training_time': self.training_time,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.model_name = model_data.get('model_name', self.model_name)
        self.model_params = model_data.get('model_params', self.model_params)
        self.training_time = model_data.get('training_time', 0)
        self.is_trained = model_data.get('is_trained', True)
        print(f"Model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters."""
        return {
            'name': self.model_name,
            'parameters': self.model_params,
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'inference_time': self.inference_time
        }

    def predict_with_timing(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make predictions with timing information."""
        start_time = time.time()
        predictions = self.predict(X)
        inference_time = time.time() - start_time
        self.inference_time = inference_time
        return predictions, inference_time