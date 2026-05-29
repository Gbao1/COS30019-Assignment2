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

class TestInputValidation(TestModelBase):
    """Test input validation and edge cases."""

    @pytest.fixture
    def sample_data_with_nans(self):
        """Create sample data with NaN values."""
        np.random.seed(42)
        X = np.random.randn(50, 8, 1) * 100 + 500
        y = np.random.randn(50, 1) * 100 + 600

        # Add some NaN values
        X[0, 0, 0] = np.nan
        y[5, 0] = np.inf

        return X, y

    @pytest.fixture
    def sample_data_with_infs(self):
        """Create sample data with infinity values."""
        np.random.seed(42)
        X = np.random.randn(50, 8, 1) * 100 + 500
        y = np.random.randn(50, 1) * 100 + 600

        # Add some infinity values
        X[10, 5, 0] = np.inf
        y[15, 0] = -np.inf

        return X, y

    def test_nan_validation_random_forest(self, sample_data_with_nans):
        """Test that Random Forest rejects NaN inputs."""
        model = RandomForestTrafficModel(n_estimators=10)
        X, y = sample_data_with_nans

        with pytest.raises(ValueError, match="NaN or infinite"):
            model._preprocess_data(X, y, fit_scaler=True)

    def test_inf_validation_random_forest(self, sample_data_with_infs):
        """Test that Random Forest rejects infinite inputs."""
        model = RandomForestTrafficModel(n_estimators=10)
        X, y = sample_data_with_infs

        with pytest.raises(ValueError, match="NaN or infinite"):
            model._preprocess_data(X, y, fit_scaler=True)

    def test_nan_validation_lstm(self, sample_data_with_nans):
        """Test that LSTM rejects NaN inputs."""
        try:
            model = LSTMTrafficModel(units=32)
            X, y = sample_data_with_nans

            with pytest.raises(ValueError, match="NaN or infinite"):
                model._preprocess_data(X, y, fit_scaler=True)
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_prediction_validation_trained_model(self, train_test_split_data):
        """Test that trained models validate prediction inputs."""
        model = RandomForestTrafficModel(n_estimators=10)
        data = train_test_split_data

        # Train model first
        model.train(data['X_train'], data['y_train'])

        # Create data with NaN
        X_with_nan = data['X_test'].copy()
        X_with_nan[0, 0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN or infinite"):
            model.predict(X_with_nan)

class TestRandomStateHandling(TestModelBase):
    """Test random state parameter handling."""

    def test_random_forest_uses_random_state(self):
        """Test that Random Forest uses the provided random_state."""
        custom_random_state = 123
        model = RandomForestTrafficModel(
            n_estimators=10,
            random_state=custom_random_state
        )

        # Build model to check parameters
        built_model = model.build_model((8, 1))

        assert built_model.random_state == custom_random_state

    def test_random_forest_default_random_state(self):
        """Test that Random Forest uses default random_state when not provided."""
        model = RandomForestTrafficModel(n_estimators=10)
        built_model = model.build_model((8, 1))

        # Should use default (42) when not specified in model_params
        assert built_model.random_state == 42

class TestProperDenormalization(TestModelBase):
    """Test that denormalization produces reasonable results."""

    def test_lstm_denormalization_range(self, train_test_split_data):
        """Test that LSTM predictions are in reasonable range after denormalization."""
        try:
            model = LSTMTrafficModel(units=32)
            data = train_test_split_data

            # Train model
            model.train(data['X_train'], data['y_train'], epochs=2, batch_size=16)

            # Make predictions
            predictions = model.predict(data['X_test'])

            # Check that predictions are in reasonable range
            # Should be roughly in the same range as training data
            y_train_min, y_train_max = data['y_train'].min(), data['y_train'].max()
            pred_min, pred_max = predictions.min(), predictions.max()

            # Predictions shouldn't be orders of magnitude off
            # Allow for some variance but catch completely wrong scaling
            assert pred_min > y_train_min * 0.1, f"Predictions too small: {pred_min} vs train min {y_train_min}"
            assert pred_max < y_train_max * 10, f"Predictions too large: {pred_max} vs train max {y_train_max}"

            # Predictions should be non-negative for traffic flow
            assert np.all(predictions >= 0), "Traffic flow predictions should be non-negative"

        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_gru_denormalization_range(self, train_test_split_data):
        """Test that GRU predictions are in reasonable range after denormalization."""
        try:
            model = GRUTrafficModel(units=32)
            data = train_test_split_data

            # Train model
            model.train(data['X_train'], data['y_train'], epochs=2, batch_size=16)

            # Make predictions
            predictions = model.predict(data['X_test'])

            # Check that predictions are in reasonable range
            y_train_min, y_train_max = data['y_train'].min(), data['y_train'].max()
            pred_min, pred_max = predictions.min(), predictions.max()

            # Predictions shouldn't be orders of magnitude off
            assert pred_min > y_train_min * 0.1, f"Predictions too small: {pred_min} vs train min {y_train_min}"
            assert pred_max < y_train_max * 10, f"Predictions too large: {pred_max} vs train max {y_train_max}"

            # Predictions should be non-negative for traffic flow
            assert np.all(predictions >= 0), "Traffic flow predictions should be non-negative"

        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_target_scaler_consistency(self, train_test_split_data):
        """Test that target scaling and denormalization are consistent."""
        try:
            model = LSTMTrafficModel(units=32)
            data = train_test_split_data

            # Get original target values
            original_targets = data['y_train'][:5].copy()

            # Process and unprocess data
            _, y_scaled = model._preprocess_data(
                data['X_train'][:5], original_targets, fit_scaler=True
            )

            # Manually inverse transform
            y_unscaled = model.target_scaler.inverse_transform(
                y_scaled.reshape(-1, 1)
            ).reshape(original_targets.shape)

            # Should get back original values (within numerical precision)
            np.testing.assert_allclose(
                original_targets, y_unscaled, rtol=1e-10,
                err_msg="Target scaling/unscaling should be consistent"
            )

        except ImportError:
            pytest.skip("TensorFlow not available")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])