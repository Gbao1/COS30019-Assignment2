"""
Configuration settings for traffic flow prediction model comparison.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "src" / "data"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TESTS_DIR = PROJECT_ROOT / "src" / "tests"

# Data settings
TRAFFIC_DATA_PATH = r"C:\Users\Hongl\Downloads\Scats Data October 2006.xls"
SITE_DATA_PATH = r"C:\Users\Hongl\Downloads\SCATSSiteListingSpreadsheet_VicRoads.xls"
LOCATION_DATA_PATH = r"C:\Users\Hongl\Downloads\Traffic_Count_Locations_with_LONG_LAT.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15

# Time series parameters
SEQUENCE_LENGTH = 8  # Number of past time steps to use
PREDICTION_HORIZON = 1  # Number of future time steps to predict
TIME_INTERVAL_MINUTES = 15

# Model-specific parameters
MODEL_PARAMS = {
    'lstm': {
        'units': 64,
        'dropout': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'patience': 10
    },
    'gru': {
        'units': 64,
        'dropout': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'patience': 10
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }
}

# Classification thresholds for converting regression to classification
TRAFFIC_THRESHOLDS = {
    'low': 500,     # Below 500 vehicles per 15 min
    'medium': 1500, # 500-1500 vehicles per 15 min
    'high': float('inf')  # Above 1500 vehicles per 15 min
}

# Evaluation settings
CROSS_VALIDATION_FOLDS = 5
METRICS_TO_CALCULATE = [
    # Regression metrics
    'mae', 'rmse', 'mape', 'r2',
    # Classification metrics (after converting to categories)
    'accuracy', 'precision', 'recall', 'f1_score'
]

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, TESTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)