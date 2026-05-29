"""
Simplified model comparison that works with basic dependencies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not available. Cannot run analysis.")
    exit(1)

class SimpleTrafficDataGenerator:
    """Generate realistic dummy traffic data for demonstration."""

    def __init__(self, n_sites=10, n_days=30):
        self.n_sites = n_sites
        self.n_days = n_days
        np.random.seed(42)

    def generate_data(self):
        """Generate realistic traffic patterns."""
        print("Generating realistic traffic data...")

        # Create time series for 30 days, 15-minute intervals
        intervals_per_day = 24 * 4  # 96 intervals per day
        total_intervals = self.n_days * intervals_per_day

        sites = [2000, 2200, 2820, 2825, 3001, 3002, 3120, 3122, 4030, 4040][:self.n_sites]

        all_data = []

        for site in sites:
            for day in range(self.n_days):
                for interval in range(intervals_per_day):
                    hour = interval // 4
                    minute = (interval % 4) * 15

                    # Generate realistic traffic patterns
                    base_flow = self._generate_realistic_flow(hour, day % 7)

                    # Add noise
                    flow = max(0, base_flow + np.random.normal(0, base_flow * 0.2))

                    all_data.append({
                        'site': site,
                        'day': day,
                        'hour': hour,
                        'minute': minute,
                        'flow': int(flow),
                        'interval': interval
                    })

        df = pd.DataFrame(all_data)
        print(f"Generated {len(df)} traffic flow records for {len(sites)} sites")
        return df

    def _generate_realistic_flow(self, hour, weekday):
        """Generate realistic hourly traffic flow patterns."""
        # Base patterns by hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            base = 800
        elif 10 <= hour <= 16:  # Daytime
            base = 600
        elif 22 <= hour or hour <= 5:  # Night
            base = 200
        else:  # Other times
            base = 400

        # Weekend reduction
        if weekday >= 5:  # Weekend
            base *= 0.7

        return base

def create_sequences(data, sequence_length=8):
    """Create time series sequences from traffic data."""
    print(f"Creating sequences with length {sequence_length}...")

    X_list, y_list, site_list = [], [], []

    for site in data['site'].unique():
        site_data = data[data['site'] == site].sort_values(['day', 'hour', 'minute'])
        flows = site_data['flow'].values

        # Create sequences
        for i in range(len(flows) - sequence_length):
            X_sequence = flows[i:i + sequence_length]
            y_target = flows[i + sequence_length]

            X_list.append(X_sequence)
            y_list.append(y_target)
            site_list.append(site)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Created {len(X)} sequences from {len(data['site'].unique())} sites")
    return X, y, site_list

def convert_to_classification(y_true, y_pred, thresholds):
    """Convert regression values to classification labels."""
    def categorize(values):
        labels = np.zeros(len(values), dtype=int)
        labels[values >= thresholds['low']] = 1    # medium
        labels[values >= thresholds['medium']] = 2  # high
        return labels

    true_labels = categorize(y_true)
    pred_labels = categorize(y_pred)

    return true_labels, pred_labels

def calculate_classification_metrics(true_labels, pred_labels):
    """Calculate classification metrics manually."""
    accuracy = np.mean(true_labels == pred_labels)

    # Calculate per-class precision and recall
    classes = [0, 1, 2]  # low, medium, high
    precisions, recalls = [], []

    for cls in classes:
        tp = np.sum((true_labels == cls) & (pred_labels == cls))
        fp = np.sum((true_labels != cls) & (pred_labels == cls))
        fn = np.sum((true_labels == cls) & (pred_labels != cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    # Weighted averages
    class_counts = [np.sum(true_labels == cls) for cls in classes]
    total_samples = len(true_labels)

    weighted_precision = sum(p * c for p, c in zip(precisions, class_counts)) / total_samples
    weighted_recall = sum(r * c for r, c in zip(recalls, class_counts)) / total_samples
    f1_score = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1_score': f1_score
    }

def run_simple_analysis():
    """Run a simplified model comparison analysis."""
    print("="*80)
    print("SIMPLIFIED TRAFFIC FLOW PREDICTION ANALYSIS")
    print("="*80)
    print("Note: Running with Random Forest only due to missing dependencies")
    print("Install tensorflow and scikit-learn for full analysis")
    print("="*80)

    # Generate data
    generator = SimpleTrafficDataGenerator()
    data = generator.generate_data()

    # Create sequences
    X, y, sites = create_sequences(data, sequence_length=8)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Train Random Forest
    print("\nTraining Random Forest model...")
    start_time = time.time()

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    training_time = time.time() - start_time

    # Make predictions
    print("Making predictions...")
    pred_start = time.time()
    y_pred = rf.predict(X_test_scaled)
    inference_time = time.time() - pred_start

    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calculate MAPE
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

    # Convert to classification and calculate classification metrics
    thresholds = {'low': 500, 'medium': 1500}
    true_labels, pred_labels = convert_to_classification(y_test, y_pred, thresholds)
    class_metrics = calculate_classification_metrics(true_labels, pred_labels)

    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\nModel: Random Forest")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Inference Time: {inference_time:.4f} seconds")
    print(f"Test Samples: {len(y_test)}")

    print(f"\nRegression Metrics:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    print(f"\nClassification Metrics (Traffic Categories):")
    print(f"  Low:    < 500 vehicles/15min")
    print(f"  Medium: 500-1500 vehicles/15min")
    print(f"  High:   > 1500 vehicles/15min")
    print(f"  Accuracy:  {class_metrics['accuracy']:.4f}")
    print(f"  Precision: {class_metrics['precision']:.4f}")
    print(f"  Recall:    {class_metrics['recall']:.4f}")
    print(f"  F1-Score:  {class_metrics['f1_score']:.4f}")

    # Feature importance
    print(f"\nFeature Importance (Historical Time Steps):")
    for i, importance in enumerate(rf.feature_importances_):
        print(f"  Step {i+1}: {importance:.3f}")

    # Save results
    results = {
        'model': 'Random Forest',
        'training_time': training_time,
        'inference_time': inference_time,
        'samples': len(y_test),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        **class_metrics,
        'feature_importance': rf.feature_importances_.tolist()
    }

    # Create results directory and save
    Path("results").mkdir(exist_ok=True)

    results_df = pd.DataFrame([results])
    results_df.to_csv("results/simple_analysis_results.csv", index=False)

    print(f"\nResults saved to: results/simple_analysis_results.csv")
    print(f"\nNote: This is a simplified analysis with synthetic data.")
    print(f"For full analysis with LSTM/GRU models, install all dependencies:")
    print(f"  pip install tensorflow scikit-learn pytest")

    return results

if __name__ == "__main__":
    try:
        results = run_simple_analysis()
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()