"""
Random Forest model for traffic flow prediction.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import time

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .base_model import BaseTrafficModel

class RandomForestTrafficModel(BaseTrafficModel):
    """Random Forest-based traffic flow prediction model."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 5, min_samples_leaf: int = 2, **kwargs):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            **kwargs: Additional model parameters
        """
        super().__init__("RandomForest", n_estimators=n_estimators,
                         max_depth=max_depth, min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, **kwargs)

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Random Forest model. Please install: pip install scikit-learn")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.scaler = StandardScaler()
        self.feature_importances_ = None

    def build_model(self, input_shape: Tuple[int, ...], **kwargs) -> RandomForestRegressor:
        """
        Build Random Forest model.

        Args:
            input_shape: Shape of input data (not used for RF, but kept for consistency)

        Returns:
            Random Forest model
        """
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.model_params.get('random_state', 42),
            n_jobs=-1  # Use all available cores
        )

        return model

    def _preprocess_data(self, X: np.ndarray, y: np.ndarray = None, fit_scaler: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess data for Random Forest.

        Random Forest expects 2D input, so we need to flatten the sequence dimension.
        """
        # Input validation
        if not np.all(np.isfinite(X)):
            raise ValueError("Input X contains NaN or infinite values")

        # Flatten the sequence dimension: (samples, sequence_length, features) -> (samples, sequence_length * features)
        if len(X.shape) == 3:
            X_flattened = X.reshape(X.shape[0], -1)
        else:
            X_flattened = X

        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_flattened)
        else:
            X_scaled = self.scaler.transform(X_flattened)

        if y is not None:
            # Input validation for targets
            if not np.all(np.isfinite(y)):
                raise ValueError("Input y contains NaN or infinite values")
            # Random Forest can handle the original target values
            return X_scaled, y.ravel() if len(y.shape) > 1 else y

        return X_scaled, None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              use_grid_search: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Train the Random Forest model.

        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data (optional, not used for RF training)
            y_val: Validation target data (optional, not used for RF training)
            use_grid_search: Whether to use grid search for hyperparameter optimization

        Returns:
            Training information dictionary
        """
        print(f"Training {self.model_name} model...")
        start_time = time.time()

        # Preprocess data
        X_train_scaled, y_train_processed = self._preprocess_data(X_train, y_train, fit_scaler=True)

        if use_grid_search:
            print("Performing grid search for hyperparameter optimization...")
            self.model = self._train_with_grid_search(X_train_scaled, y_train_processed)
        else:
            # Build and train model
            self.model = self.build_model(X_train.shape)
            self.model.fit(X_train_scaled, y_train_processed)

        self.training_time = time.time() - start_time
        self.is_trained = True
        self.feature_importances_ = self.model.feature_importances_

        print(f"{self.model_name} training completed in {self.training_time:.2f} seconds")

        # Calculate training score
        train_score = self.model.score(X_train_scaled, y_train_processed)

        training_info = {
            'training_time': self.training_time,
            'train_score': train_score,
            'feature_importances': self.feature_importances_,
            'n_features': X_train_scaled.shape[1]
        }

        # If validation data is provided, calculate validation score
        if X_val is not None and y_val is not None:
            X_val_scaled, y_val_processed = self._preprocess_data(X_val, y_val)
            val_score = self.model.score(X_val_scaled, y_val_processed)
            training_info['val_score'] = val_score
            print(f"Validation R² score: {val_score:.4f}")

        return training_info

    def _train_with_grid_search(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """Train model with grid search for hyperparameter optimization."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {-grid_search.best_score_:.4f}")

        # Update model parameters
        self.model_params.update(grid_search.best_params_)

        return grid_search.best_estimator_

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

        # Input validation
        if not np.all(np.isfinite(X)):
            raise ValueError("Input contains NaN or infinite values")

        # Preprocess input
        X_scaled, _ = self._preprocess_data(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)

        return predictions.reshape(-1, 1)

    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            feature_names: Names of features (optional)

        Returns:
            Dictionary of feature importances
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importances_))]

        return dict(zip(feature_names, self.feature_importances_))

    def get_model_summary(self) -> str:
        """Get model summary information."""
        if self.model is None:
            return "Model not built yet"

        summary = f"""
Random Forest Model Summary:
- Number of estimators: {self.model.n_estimators}
- Max depth: {self.model.max_depth}
- Min samples split: {self.model.min_samples_split}
- Min samples leaf: {self.model.min_samples_leaf}
- Number of features: {self.model.n_features_in_}
"""

        if self.is_trained:
            summary += f"""
- Training time: {self.training_time:.2f} seconds
- Feature importances available: {self.feature_importances_ is not None}
"""

        return summary

    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance.

        Args:
            top_n: Number of top features to show
            save_path: Path to save the plot (optional)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before plotting feature importance")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting. Please install: pip install matplotlib")
            return

        # Get top features
        feature_importance = self.feature_importances_
        indices = np.argsort(feature_importance)[::-1][:top_n]

        plt.figure(figsize=(12, 8))
        plt.title(f"Top {top_n} Feature Importances - {self.model_name}")
        plt.bar(range(top_n), feature_importance[indices])
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.xticks(range(top_n), indices, rotation=45)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")

        plt.tight_layout()
        plt.show()