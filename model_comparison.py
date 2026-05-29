"""
Comprehensive model comparison script for traffic flow prediction.

This script trains and compares LSTM, GRU, and Random Forest models using both
regression and classification metrics.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import configuration
from config import *

# Import data loading
from src.data.data_loader import TrafficDataLoader, convert_to_classification

# Import models
from src.models.random_forest_model import RandomForestTrafficModel

# Try to import neural models
try:
    from src.models.lstm_model import LSTMTrafficModel
    from src.models.gru_model import GRUTrafficModel
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Only Random Forest will be tested.")

# Import metrics
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import KFold

class ModelComparator:
    """Class to handle comprehensive model comparison."""

    def __init__(self, config_params: Dict = None):
        """
        Initialize the model comparator.

        Args:
            config_params: Configuration parameters (uses defaults if None)
        """
        self.config = config_params or {
            'sequence_length': SEQUENCE_LENGTH,
            'test_size': TEST_SIZE,
            'validation_size': VALIDATION_SIZE,
            'random_state': RANDOM_STATE,
            'cv_folds': CROSS_VALIDATION_FOLDS,
            'model_params': MODEL_PARAMS,
            'traffic_thresholds': TRAFFIC_THRESHOLDS
        }

        self.models = {}
        self.results = {}
        self.data_loader = None

    def load_and_prepare_data(self) -> Dict[str, np.ndarray]:
        """
        Load and prepare data for model training and testing.

        Returns:
            Dictionary containing train/test splits
        """
        print("Loading and preparing data...")

        # Initialize data loader
        self.data_loader = TrafficDataLoader(
            traffic_file=TRAFFIC_DATA_PATH,
            site_file=SITE_DATA_PATH,
            location_file=LOCATION_DATA_PATH
        )

        # Load and preprocess data
        X, y, site_ids = self.data_loader.preprocess_data(
            sequence_length=self.config['sequence_length']
        )

        print(f"Loaded {len(X)} samples with shape {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Number of unique sites: {len(set(site_ids))}")

        # Split data
        n_samples = len(X)
        n_train = int((1 - self.config['test_size']) * n_samples)

        # Random shuffle with fixed seed
        np.random.seed(self.config['random_state'])
        indices = np.random.permutation(n_samples)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

        data_splits = {
            'X_train': X[train_indices],
            'y_train': y[train_indices],
            'X_test': X[test_indices],
            'y_test': y[test_indices],
            'site_ids_train': [site_ids[i] for i in train_indices],
            'site_ids_test': [site_ids[i] for i in test_indices]
        }

        print(f"Training set: {len(data_splits['X_train'])} samples")
        print(f"Test set: {len(data_splits['X_test'])} samples")

        return data_splits

    def initialize_models(self) -> None:
        """Initialize all available models."""
        print("Initializing models...")

        # Random Forest (always available)
        self.models['RandomForest'] = RandomForestTrafficModel(
            **self.config['model_params']['random_forest']
        )

        # Neural models if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self.models['LSTM'] = LSTMTrafficModel(
                **self.config['model_params']['lstm']
            )
            self.models['GRU'] = GRUTrafficModel(
                **self.config['model_params']['gru']
            )

        print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")

    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of regression metrics
        """
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()

        metrics = {
            'mae': float(mean_absolute_error(y_true_flat, y_pred_flat)),
            'rmse': float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
            'r2': float(r2_score(y_true_flat, y_pred_flat))
        }

        # Calculate MAPE (avoiding division by zero)
        non_zero_mask = np.abs(y_true_flat) > 1e-8
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true_flat[non_zero_mask] - y_pred_flat[non_zero_mask]) /
                                y_true_flat[non_zero_mask])) * 100
            metrics['mape'] = float(mape)
        else:
            metrics['mape'] = float('inf')

        return metrics

    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate classification metrics after converting regression to classification.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of classification metrics
        """
        # Convert to classification labels
        true_labels, pred_labels = convert_to_classification(
            y_true, y_pred, self.config['traffic_thresholds']
        )

        metrics = {
            'accuracy': float(accuracy_score(true_labels, pred_labels)),
            'precision': float(precision_score(true_labels, pred_labels, average='weighted', zero_division=0)),
            'recall': float(recall_score(true_labels, pred_labels, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(true_labels, pred_labels, average='weighted', zero_division=0))
        }

        # Add per-class metrics
        for class_idx, class_name in enumerate(['low', 'medium', 'high']):
            if class_idx in true_labels:
                metrics[f'precision_{class_name}'] = float(
                    precision_score(true_labels == class_idx, pred_labels == class_idx, zero_division=0)
                )
                metrics[f'recall_{class_name}'] = float(
                    recall_score(true_labels == class_idx, pred_labels == class_idx, zero_division=0)
                )

        return metrics

    def train_and_evaluate_model(self, model_name: str, model, data_splits: Dict) -> Dict[str, Any]:
        """
        Train and evaluate a single model.

        Args:
            model_name: Name of the model
            model: Model instance
            data_splits: Data splits dictionary

        Returns:
            Dictionary of results
        """
        print(f"\nTraining {model_name} model...")

        start_time = time.time()

        # Prepare training parameters
        train_params = {}
        if model_name in ['LSTM', 'GRU']:
            train_params.update({
                'epochs': self.config['model_params'][model_name.lower()]['epochs'],
                'batch_size': self.config['model_params'][model_name.lower()]['batch_size'],
                'patience': self.config['model_params'][model_name.lower()]['patience']
            })

            # Use validation split for neural models
            n_train = len(data_splits['X_train'])
            n_val = int(self.config['validation_size'] * n_train)

            val_indices = np.random.choice(n_train, n_val, replace=False)
            train_indices = np.setdiff1d(np.arange(n_train), val_indices)

            X_train_subset = data_splits['X_train'][train_indices]
            y_train_subset = data_splits['y_train'][train_indices]
            X_val = data_splits['X_train'][val_indices]
            y_val = data_splits['y_train'][val_indices]

            train_params.update({
                'X_val': X_val,
                'y_val': y_val
            })

            # Train model
            training_history = model.train(X_train_subset, y_train_subset, **train_params)

        else:  # Random Forest
            # Train model
            training_history = model.train(data_splits['X_train'], data_splits['y_train'], **train_params)

        training_time = time.time() - start_time

        # Make predictions
        print(f"Making predictions with {model_name}...")
        pred_start_time = time.time()
        predictions = model.predict(data_splits['X_test'])
        inference_time = time.time() - pred_start_time

        # Calculate metrics
        print(f"Calculating metrics for {model_name}...")
        regression_metrics = self.calculate_regression_metrics(data_splits['y_test'], predictions)
        classification_metrics = self.calculate_classification_metrics(data_splits['y_test'], predictions)

        # Compile results
        results = {
            'model_name': model_name,
            'training_time': training_time,
            'inference_time': inference_time,
            'total_samples': len(data_splits['X_test']),
            'regression_metrics': regression_metrics,
            'classification_metrics': classification_metrics,
            'training_history': training_history if isinstance(training_history, dict) else None
        }

        print(f"{model_name} completed!")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Inference time: {inference_time:.4f}s")
        print(f"  MAE: {regression_metrics['mae']:.2f}")
        print(f"  RMSE: {regression_metrics['rmse']:.2f}")
        print(f"  R²: {regression_metrics['r2']:.4f}")
        print(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {classification_metrics['f1_score']:.4f}")

        return results

    def cross_validate_model(self, model_class, model_params: Dict, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """
        Perform cross-validation for a model.

        Args:
            model_class: Model class to instantiate
            model_params: Model parameters
            X: Input data
            y: Target data

        Returns:
            Dictionary of cross-validation scores
        """
        kfold = KFold(n_splits=self.config['cv_folds'], shuffle=True, random_state=self.config['random_state'])

        cv_scores = {
            'mae': [], 'rmse': [], 'r2': [], 'mape': [],
            'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []
        }

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"  Fold {fold + 1}/{self.config['cv_folds']}")

            # Split data
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            # Initialize and train model
            model = model_class(**model_params)

            if hasattr(model, 'units'):  # Neural model
                model.train(X_fold_train, y_fold_train, epochs=10, batch_size=32, verbose=0)
            else:  # Random Forest
                model.train(X_fold_train, y_fold_train)

            # Make predictions
            predictions = model.predict(X_fold_val)

            # Calculate metrics
            reg_metrics = self.calculate_regression_metrics(y_fold_val, predictions)
            cls_metrics = self.calculate_classification_metrics(y_fold_val, predictions)

            # Store scores
            for metric in ['mae', 'rmse', 'r2', 'mape']:
                cv_scores[metric].append(reg_metrics[metric])

            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                cv_scores[metric].append(cls_metrics[metric])

        return cv_scores

    def run_comparison(self) -> Dict[str, Any]:
        """
        Run the complete model comparison.

        Returns:
            Dictionary containing all results
        """
        print("Starting comprehensive model comparison...")
        print("=" * 80)

        # Load and prepare data
        data_splits = self.load_and_prepare_data()

        # Initialize models
        self.initialize_models()

        # Train and evaluate each model
        model_results = {}
        for model_name, model in self.models.items():
            model_results[model_name] = self.train_and_evaluate_model(
                model_name, model, data_splits
            )

        # Perform cross-validation
        print("\nPerforming cross-validation...")
        cv_results = {}

        # Cross-validation for Random Forest
        if 'RandomForest' in self.models:
            print("Cross-validating Random Forest...")
            cv_results['RandomForest'] = self.cross_validate_model(
                RandomForestTrafficModel,
                self.config['model_params']['random_forest'],
                data_splits['X_train'],
                data_splits['y_train']
            )

        # Cross-validation for neural models
        if TENSORFLOW_AVAILABLE:
            if 'LSTM' in self.models:
                print("Cross-validating LSTM...")
                cv_results['LSTM'] = self.cross_validate_model(
                    LSTMTrafficModel,
                    {'units': self.config['model_params']['lstm']['units'],
                     'dropout': self.config['model_params']['lstm']['dropout']},
                    data_splits['X_train'],
                    data_splits['y_train']
                )

            if 'GRU' in self.models:
                print("Cross-validating GRU...")
                cv_results['GRU'] = self.cross_validate_model(
                    GRUTrafficModel,
                    {'units': self.config['model_params']['gru']['units'],
                     'dropout': self.config['model_params']['gru']['dropout']},
                    data_splits['X_train'],
                    data_splits['y_train']
                )

        # Compile final results
        final_results = {
            'model_results': model_results,
            'cross_validation_results': cv_results,
            'data_info': {
                'total_samples': len(data_splits['X_train']) + len(data_splits['X_test']),
                'train_samples': len(data_splits['X_train']),
                'test_samples': len(data_splits['X_test']),
                'sequence_length': self.config['sequence_length'],
                'unique_sites': len(set(data_splits['site_ids_train'] + data_splits['site_ids_test']))
            },
            'config': self.config
        }

        self.results = final_results
        return final_results

    def create_comparison_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame for easy comparison.

        Returns:
            DataFrame with comparison metrics
        """
        if not self.results:
            raise ValueError("No results available. Run comparison first.")

        summary_data = []

        for model_name, results in self.results['model_results'].items():
            row = {
                'Model': model_name,
                'Training_Time_s': results['training_time'],
                'Inference_Time_s': results['inference_time'],
                **results['regression_metrics'],
                **results['classification_metrics']
            }
            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        # Round numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].round(4)

        return df

    def save_results(self, output_dir: str = "results") -> None:
        """
        Save results to files.

        Args:
            output_dir: Directory to save results
        """
        if not self.results:
            raise ValueError("No results available. Run comparison first.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save complete results as JSON
        results_file = output_path / "complete_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = self._convert_numpy_types(self.results)
            json.dump(json_results, f, indent=2, default=str)

        print(f"Complete results saved to {results_file}")

        # Save summary as CSV
        summary_df = self.create_comparison_summary()
        summary_file = output_path / "model_comparison_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")

        # Save cross-validation results
        if self.results.get('cross_validation_results'):
            cv_summary = self._create_cv_summary()
            cv_file = output_path / "cross_validation_summary.csv"
            cv_summary.to_csv(cv_file, index=False)
            print(f"Cross-validation summary saved to {cv_file}")

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _create_cv_summary(self) -> pd.DataFrame:
        """Create cross-validation summary DataFrame."""
        cv_data = []

        for model_name, cv_results in self.results['cross_validation_results'].items():
            for metric, scores in cv_results.items():
                cv_data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Mean': np.mean(scores),
                    'Std': np.std(scores),
                    'Min': np.min(scores),
                    'Max': np.max(scores)
                })

        return pd.DataFrame(cv_data).round(4)

def main():
    """Main function to run the model comparison."""
    print("Traffic Flow Prediction Model Comparison")
    print("=" * 80)
    print("This script will:")
    print("1. Load and preprocess traffic data")
    print("2. Train LSTM, GRU, and Random Forest models")
    print("3. Evaluate using both regression and classification metrics")
    print("4. Perform cross-validation")
    print("5. Save comprehensive results")
    print("=" * 80)

    # Initialize comparator
    comparator = ModelComparator()

    try:
        # Run comparison
        results = comparator.run_comparison()

        # Display summary
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        summary_df = comparator.create_comparison_summary()
        print(summary_df.to_string(index=False))

        # Save results
        print("\nSaving results...")
        comparator.save_results()

        print("\n" + "=" * 80)
        print("Model comparison completed successfully!")
        print("Check the 'results' directory for detailed output files.")

    except Exception as e:
        print(f"\nError during comparison: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()