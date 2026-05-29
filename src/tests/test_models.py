"""
Unit tests for traffic flow prediction models.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lstm_model import LSTMTrafficModel
from models.gru_model import GRUTrafficModel
from models.random_forest_model import RandomForestTrafficModel

class TestModelBase:
    """Base test class with common setup and utilities."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)

        # Create time series data: (samples, sequence_length, features)
        n_samples = 100
        sequence_length = 8
        n_features = 1

        X = np.random.randn(n_samples, sequence_length, n_features) * 100 + 500
        y = np.random.randn(n_samples, 1) * 100 + 600

        # Ensure non-negative values (traffic flow can't be negative)
        X = np.abs(X)
        y = np.abs(y)

        return X, y

    @pytest.fixture
    def train_test_split_data(self, sample_data):
        """Split data into train/test sets."""
        X, y = sample_data
        split_idx = int(0.8 * len(X))

        return {
            'X_train': X[:split_idx],
            'y_train': y[:split_idx],
            'X_test': X[split_idx:],
            'y_test': y[split_idx:]
        }

class TestRandomForestModel(TestModelBase):
    """Test cases for Random Forest model."""

    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = RandomForestTrafficModel()

        assert model.model_name == "RandomForest"
        assert model.n_estimators == 100
        assert model.max_depth == 10
        assert not model.is_trained
        assert model.model is None

    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = RandomForestTrafficModel(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10
        )

        assert model.n_estimators == 200
        assert model.max_depth == 20
        assert model.min_samples_split == 10

    def test_model_training(self, train_test_split_data):
        """Test model training process."""
        model = RandomForestTrafficModel(n_estimators=10)  # Small for fast testing
        data = train_test_split_data

        # Train model
        history = model.train(data['X_train'], data['y_train'])

        assert model.is_trained
        assert model.model is not None
        assert model.training_time > 0
        assert 'train_score' in history
        assert 'feature_importances' in history

    def test_model_prediction(self, train_test_split_data):
        """Test model prediction."""
        model = RandomForestTrafficModel(n_estimators=10)
        data = train_test_split_data

        # Train and predict
        model.train(data['X_train'], data['y_train'])
        predictions = model.predict(data['X_test'])

        assert predictions.shape == data['y_test'].shape
        assert np.all(predictions >= 0)  # Traffic flow should be non-negative

    def test_prediction_before_training(self, sample_data):
        """Test that prediction fails before training."""
        model = RandomForestTrafficModel()
        X, _ = sample_data

        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(X)

    def test_data_preprocessing(self, sample_data):
        """Test data preprocessing functionality."""
        model = RandomForestTrafficModel()
        X, y = sample_data

        # Test fitting scaler
        X_processed, y_processed = model._preprocess_data(X, y, fit_scaler=True)

        assert X_processed.shape[0] == X.shape[0]
        assert X_processed.shape[1] == X.shape[1] * X.shape[2]  # Flattened
        assert y_processed.shape == y.ravel().shape

    def test_feature_importance(self, train_test_split_data):
        """Test feature importance functionality."""
        model = RandomForestTrafficModel(n_estimators=10)
        data = train_test_split_data

        model.train(data['X_train'], data['y_train'])
        importance_dict = model.get_feature_importance()

        assert isinstance(importance_dict, dict)
        assert len(importance_dict) == data['X_train'].shape[1] * data['X_train'].shape[2]

class TestLSTMModel(TestModelBase):
    """Test cases for LSTM model."""

    def test_model_initialization(self):
        """Test LSTM model initialization."""
        try:
            model = LSTMTrafficModel()
            assert model.model_name == "LSTM"
            assert model.units == 64
            assert model.dropout == 0.2
            assert not model.is_trained
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_model_training_short(self, train_test_split_data):
        """Test LSTM model training with minimal epochs."""
        try:
            model = LSTMTrafficModel(units=32)  # Smaller for faster testing
            data = train_test_split_data

            # Train with minimal epochs
            history = model.train(
                data['X_train'], data['y_train'],
                epochs=2, batch_size=16
            )

            assert model.is_trained
            assert model.model is not None
            assert 'loss' in history
            assert model.training_time > 0

        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_lstm_prediction(self, train_test_split_data):
        """Test LSTM prediction."""
        try:
            model = LSTMTrafficModel(units=32)
            data = train_test_split_data

            # Train and predict
            model.train(data['X_train'], data['y_train'], epochs=2, batch_size=16)
            predictions = model.predict(data['X_test'])

            assert predictions.shape == data['y_test'].shape
            assert np.all(predictions >= 0)

        except ImportError:
            pytest.skip("TensorFlow not available")

class TestGRUModel(TestModelBase):
    """Test cases for GRU model."""

    def test_model_initialization(self):
        """Test GRU model initialization."""
        try:
            model = GRUTrafficModel()
            assert model.model_name == "GRU"
            assert model.units == 64
            assert model.dropout == 0.2
            assert not model.is_trained
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_model_training_short(self, train_test_split_data):
        """Test GRU model training with minimal epochs."""
        try:
            model = GRUTrafficModel(units=32)
            data = train_test_split_data

            history = model.train(
                data['X_train'], data['y_train'],
                epochs=2, batch_size=16
            )

            assert model.is_trained
            assert model.model is not None
            assert 'loss' in history
            assert model.training_time > 0

        except ImportError:
            pytest.skip("TensorFlow not available")

class TestModelComparison:
    """Integration tests for model comparison."""

    @pytest.fixture
    def all_models(self):
        """Create instances of all models."""
        models = {}

        # Always include Random Forest
        models['rf'] = RandomForestTrafficModel(n_estimators=10)

        # Include neural models if TensorFlow is available
        try:
            models['lstm'] = LSTMTrafficModel(units=32)
            models['gru'] = GRUTrafficModel(units=32)
        except ImportError:
            pass

        return models

    def test_all_models_training(self, all_models, train_test_split_data):
        """Test that all available models can train successfully."""
        data = train_test_split_data

        for name, model in all_models.items():
            print(f"Testing {name} model...")

            if name in ['lstm', 'gru']:
                model.train(data['X_train'], data['y_train'], epochs=2, batch_size=16)
            else:
                model.train(data['X_train'], data['y_train'])

            assert model.is_trained
            assert model.training_time > 0

    def test_all_models_prediction(self, all_models, train_test_split_data):
        """Test that all models produce valid predictions."""
        data = train_test_split_data

        for name, model in all_models.items():
            # Train model
            if name in ['lstm', 'gru']:
                model.train(data['X_train'], data['y_train'], epochs=2, batch_size=16)
            else:
                model.train(data['X_train'], data['y_train'])

            # Make predictions
            predictions = model.predict(data['X_test'])

            assert predictions.shape == data['y_test'].shape
            assert np.all(np.isfinite(predictions))  # No NaN or inf values
            assert np.all(predictions >= 0)  # Traffic flow should be non-negative

    def test_model_timing(self, all_models, train_test_split_data):
        """Test that timing information is captured correctly."""
        data = train_test_split_data

        for name, model in all_models.items():
            # Train model
            if name in ['lstm', 'gru']:
                model.train(data['X_train'], data['y_train'], epochs=2, batch_size=16)
            else:
                model.train(data['X_train'], data['y_train'])

            # Test prediction timing
            predictions, inference_time = model.predict_with_timing(data['X_test'])

            assert inference_time > 0
            assert model.inference_time > 0
            assert predictions.shape == data['y_test'].shape

    def test_model_info(self, all_models):
        """Test model information retrieval."""
        for name, model in all_models.items():
            info = model.get_model_info()

            assert 'name' in info
            assert 'parameters' in info
            assert 'is_trained' in info
            assert info['name'] == model.model_name

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])