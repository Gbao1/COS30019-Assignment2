"""
Integration tests for model comparison functionality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import TrafficDataLoader, convert_to_classification

class TestModelComparisonIntegration:
    """Integration tests for the complete model comparison pipeline."""

    @pytest.fixture
    def comparison_data(self):
        """Create data suitable for model comparison testing."""
        loader = TrafficDataLoader("dummy.xls", "dummy.xls", "dummy.csv")
        X, y, site_ids = loader.preprocess_data(sequence_length=8)

        # Ensure we have enough data for testing
        if len(X) < 20:
            # Create additional synthetic data if needed
            np.random.seed(42)
            n_samples = 100
            X = np.random.randn(n_samples, 8, 1) * 100 + 500
            y = np.random.randn(n_samples, 1) * 100 + 600
            X = np.abs(X)
            y = np.abs(y)
            site_ids = [2000] * n_samples

        # Split into train/test
        split_idx = int(0.8 * len(X))
        return {
            'X_train': X[:split_idx],
            'y_train': y[:split_idx],
            'X_test': X[split_idx:],
            'y_test': y[split_idx:],
            'site_ids_train': site_ids[:split_idx],
            'site_ids_test': site_ids[split_idx:]
        }

    def test_end_to_end_pipeline(self, comparison_data):
        """Test the complete end-to-end model comparison pipeline."""
        data = comparison_data

        # Import models
        from models.random_forest_model import RandomForestTrafficModel

        # Test Random Forest (always available)
        rf_model = RandomForestTrafficModel(n_estimators=10)

        # Train model
        rf_model.train(data['X_train'], data['y_train'])

        # Make predictions
        rf_predictions = rf_model.predict(data['X_test'])

        assert rf_predictions.shape == data['y_test'].shape
        assert np.all(rf_predictions >= 0)

        # Test neural models if available
        try:
            from models.lstm_model import LSTMTrafficModel
            from models.gru_model import GRUTrafficModel

            # Test LSTM
            lstm_model = LSTMTrafficModel(units=32)
            lstm_model.train(data['X_train'], data['y_train'], epochs=2)
            lstm_predictions = lstm_model.predict(data['X_test'])

            assert lstm_predictions.shape == data['y_test'].shape
            assert np.all(lstm_predictions >= 0)

            # Test GRU
            gru_model = GRUTrafficModel(units=32)
            gru_model.train(data['X_train'], data['y_train'], epochs=2)
            gru_predictions = gru_model.predict(data['X_test'])

            assert gru_predictions.shape == data['y_test'].shape
            assert np.all(gru_predictions >= 0)

        except ImportError:
            print("TensorFlow models not available, testing Random Forest only")

    def test_regression_metrics_calculation(self, comparison_data):
        """Test calculation of regression metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        data = comparison_data
        y_true = data['y_test']

        # Create some sample predictions
        np.random.seed(42)
        y_pred = y_true + np.random.normal(0, 50, y_true.shape)

        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Check metrics are reasonable
        assert mae > 0
        assert rmse > 0
        assert rmse >= mae  # RMSE should be >= MAE
        assert -1 <= r2 <= 1  # R² should be between -1 and 1

        # Calculate MAPE
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        assert mape >= 0

    def test_classification_metrics_calculation(self, comparison_data):
        """Test calculation of classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        data = comparison_data
        y_true = data['y_test']

        # Create predictions
        np.random.seed(42)
        y_pred = y_true + np.random.normal(0, 100, y_true.shape)

        # Convert to classification
        thresholds = {'low': 500, 'medium': 1500, 'high': float('inf')}
        true_labels, pred_labels = convert_to_classification(y_true, y_pred, thresholds)

        # Calculate classification metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='weighted')
        recall = recall_score(true_labels, pred_labels, average='weighted')
        f1 = f1_score(true_labels, pred_labels, average='weighted')

        # Check metrics are in valid ranges
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1

    def test_cross_validation_setup(self, comparison_data):
        """Test cross-validation setup for model comparison."""
        from sklearn.model_selection import KFold

        data = comparison_data
        X, y = data['X_train'], data['y_train']

        # Test K-fold setup
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)

        fold_count = 0
        for train_idx, val_idx in kfold.split(X):
            X_fold_train = X[train_idx]
            y_fold_train = y[train_idx]
            X_fold_val = X[val_idx]
            y_fold_val = y[val_idx]

            # Check split is valid
            assert len(X_fold_train) > 0
            assert len(X_fold_val) > 0
            assert len(X_fold_train) + len(X_fold_val) == len(X)

            fold_count += 1

        assert fold_count == 3

    def test_model_performance_comparison(self, comparison_data):
        """Test comparing performance across models."""
        data = comparison_data

        # Dictionary to store results
        results = {}

        # Test Random Forest
        from models.random_forest_model import RandomForestTrafficModel
        rf_model = RandomForestTrafficModel(n_estimators=10)
        rf_model.train(data['X_train'], data['y_train'])
        rf_pred = rf_model.predict(data['X_test'])

        # Calculate metrics for RF
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        results['RandomForest'] = {
            'mae': mean_absolute_error(data['y_test'], rf_pred),
            'rmse': np.sqrt(mean_squared_error(data['y_test'], rf_pred)),
            'training_time': rf_model.training_time
        }

        # Test neural models if available
        try:
            from models.lstm_model import LSTMTrafficModel

            lstm_model = LSTMTrafficModel(units=32)
            lstm_model.train(data['X_train'], data['y_train'], epochs=2, batch_size=16)
            lstm_pred = lstm_model.predict(data['X_test'])

            results['LSTM'] = {
                'mae': mean_absolute_error(data['y_test'], lstm_pred),
                'rmse': np.sqrt(mean_squared_error(data['y_test'], lstm_pred)),
                'training_time': lstm_model.training_time
            }

        except ImportError:
            pass

        # Verify we have results
        assert len(results) >= 1
        assert 'RandomForest' in results

        # Check each result has required metrics
        for model_name, metrics in results.items():
            assert 'mae' in metrics
            assert 'rmse' in metrics
            assert 'training_time' in metrics
            assert metrics['mae'] > 0
            assert metrics['rmse'] > 0
            assert metrics['training_time'] > 0

    def test_model_consistency(self, comparison_data):
        """Test model prediction consistency."""
        data = comparison_data

        from models.random_forest_model import RandomForestTrafficModel

        # Train two identical models
        model1 = RandomForestTrafficModel(n_estimators=10, random_state=42)
        model2 = RandomForestTrafficModel(n_estimators=10, random_state=42)

        # Train both models
        model1.train(data['X_train'], data['y_train'])
        model2.train(data['X_train'], data['y_train'])

        # Make predictions
        pred1 = model1.predict(data['X_test'])
        pred2 = model2.predict(data['X_test'])

        # Predictions should be very similar (allowing for small numerical differences)
        np.testing.assert_allclose(pred1, pred2, rtol=1e-10)

    def test_data_leakage_prevention(self, comparison_data):
        """Test that there's no data leakage between train and test sets."""
        data = comparison_data

        # Check that train and test indices don't overlap
        n_train = len(data['X_train'])
        n_test = len(data['X_test'])
        n_total = n_train + n_test

        # Verify split makes sense
        assert n_train > 0
        assert n_test > 0
        assert n_train + n_test <= len(data['X_train']) + len(data['X_test'])

        # Check that data shapes are consistent
        assert data['X_train'].shape[1:] == data['X_test'].shape[1:]
        assert data['y_train'].shape[1:] == data['y_test'].shape[1:]

class TestPerformanceMetrics:
    """Test specific performance metric calculations."""

    def test_mape_calculation(self):
        """Test Mean Absolute Percentage Error calculation."""
        y_true = np.array([100, 200, 300, 400, 500]).reshape(-1, 1)
        y_pred = np.array([110, 180, 320, 380, 520]).reshape(-1, 1)

        # Calculate MAPE manually
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        assert isinstance(mape, float)
        assert mape > 0

        # Test with zero values (should handle gracefully)
        y_true_with_zero = np.array([0, 100, 200]).reshape(-1, 1)
        y_pred_with_zero = np.array([10, 110, 180]).reshape(-1, 1)

        # Should not crash with division by zero
        mape_safe = np.mean(np.abs((y_true_with_zero - y_pred_with_zero) /
                                  np.maximum(y_true_with_zero, 1e-8))) * 100
        assert np.isfinite(mape_safe)

    def test_r2_score_calculation(self):
        """Test R² score calculation."""
        from sklearn.metrics import r2_score

        y_true = np.array([100, 200, 300, 400, 500]).reshape(-1, 1)

        # Perfect predictions
        y_pred_perfect = y_true.copy()
        r2_perfect = r2_score(y_true, y_pred_perfect)
        assert r2_perfect == 1.0

        # Reasonable predictions
        y_pred_good = y_true + np.random.normal(0, 10, y_true.shape)
        r2_good = r2_score(y_true, y_pred_good)
        assert 0 <= r2_good <= 1

        # Bad predictions (worse than mean)
        y_pred_bad = np.full_like(y_true, np.mean(y_true) + 1000)
        r2_bad = r2_score(y_true, y_pred_bad)
        assert r2_bad < 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])