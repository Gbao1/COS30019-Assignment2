"""
Data loading and preprocessing for traffic flow prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class TrafficDataLoader:
    """Handles loading and preprocessing of SCATS traffic data."""

    def __init__(self, traffic_file: str, site_file: str, location_file: str):
        """
        Initialize the data loader.

        Args:
            traffic_file: Path to SCATS traffic data Excel file
            site_file: Path to site listing Excel file
            location_file: Path to location CSV file
        """
        self.traffic_file = traffic_file
        self.site_file = site_file
        self.location_file = location_file

    def load_traffic_data(self) -> pd.DataFrame:
        """Load and clean traffic data from Excel file."""
        print(f"Loading traffic data from {self.traffic_file}")

        try:
            # Read the Excel file - usually the first sheet contains the data
            df = pd.read_excel(self.traffic_file, sheet_name=0)
            print(f"Loaded traffic data with shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error loading traffic data: {e}")
            # Return dummy data for testing purposes
            return self._create_dummy_traffic_data()

    def load_site_data(self) -> pd.DataFrame:
        """Load site information."""
        print(f"Loading site data from {self.site_file}")

        try:
            df = pd.read_excel(self.site_file, sheet_name=0)
            print(f"Loaded site data with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading site data: {e}")
            return self._create_dummy_site_data()

    def load_location_data(self) -> pd.DataFrame:
        """Load location data."""
        print(f"Loading location data from {self.location_file}")

        try:
            df = pd.read_csv(self.location_file)
            print(f"Loaded location data with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading location data: {e}")
            return self._create_dummy_location_data()

    def _create_dummy_traffic_data(self) -> pd.DataFrame:
        """Create dummy traffic data for testing when real data is unavailable."""
        print("Creating dummy traffic data for testing...")

        # Create 30 days worth of 15-minute intervals
        date_range = pd.date_range(start='2006-10-01', end='2006-10-31', freq='15min')

        # Create data for 10 different sites
        sites = [2000, 2200, 2820, 2825, 3001, 3002, 3120, 3122, 4030, 4040]

        data = []
        np.random.seed(42)

        for site in sites:
            for timestamp in date_range:
                # Generate realistic traffic patterns
                hour = timestamp.hour
                weekday = timestamp.weekday()

                # Base traffic with daily patterns
                if 7 <= hour <= 9 or 17 <= hour <= 19:  # Peak hours
                    base_flow = 800 + np.random.normal(200, 100)
                elif 10 <= hour <= 16:  # Daytime
                    base_flow = 600 + np.random.normal(150, 80)
                elif 22 <= hour <= 5:  # Night
                    base_flow = 200 + np.random.normal(50, 30)
                else:  # Other times
                    base_flow = 400 + np.random.normal(100, 50)

                # Weekend reduction
                if weekday >= 5:  # Weekend
                    base_flow *= 0.7

                # Ensure non-negative
                flow = max(0, base_flow)

                data.append({
                    'NB_SCATS_SITE': site,
                    'QT_INTERVAL_COUNT': int(flow),
                    'CD_DAY': timestamp.strftime('%d/%m/%Y'),
                    'CD_MELWAY_COORDINATE': f'{site}A1',
                    'DT_REPORT': timestamp.strftime('%d/%m/%Y %H:%M:%S')
                })

        df = pd.DataFrame(data)
        print(f"Created dummy traffic data with {len(df)} records for {len(sites)} sites")
        return df

    def _create_dummy_site_data(self) -> pd.DataFrame:
        """Create dummy site data."""
        sites = [2000, 2200, 2820, 2825, 3001, 3002, 3120, 3122, 4030, 4040]
        data = []

        for i, site in enumerate(sites):
            data.append({
                'NB_SCATS_SITE': site,
                'NM_REGION': f'Region_{i%3}',
                'Location': f'Test Location {site}'
            })

        return pd.DataFrame(data)

    def _create_dummy_location_data(self) -> pd.DataFrame:
        """Create dummy location data."""
        sites = [2000, 2200, 2820, 2825, 3001, 3002, 3120, 3122, 4030, 4040]
        data = []

        # Melbourne coordinates range
        for i, site in enumerate(sites):
            data.append({
                'SITE': site,
                'LATITUDE': -37.8 + np.random.uniform(-0.2, 0.2),
                'LONGITUDE': 144.9 + np.random.uniform(-0.3, 0.3)
            })

        return pd.DataFrame(data)

    def preprocess_data(self, sequence_length: int = 8) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Preprocess the traffic data for time series prediction.

        Args:
            sequence_length: Number of past time steps to use for prediction

        Returns:
            Tuple of (X, y, site_ids) where:
            - X: Input sequences of shape (n_samples, sequence_length, 1)
            - y: Target values of shape (n_samples, 1)
            - site_ids: List of site IDs corresponding to each sample
        """
        # Load all data
        traffic_df = self.load_traffic_data()

        # Prepare the data
        X_list, y_list, site_list = [], [], []

        # Get unique sites
        if 'NB_SCATS_SITE' in traffic_df.columns:
            sites = traffic_df['NB_SCATS_SITE'].unique()
        else:
            print("Warning: NB_SCATS_SITE column not found, using dummy sites")
            sites = [2000, 2200, 2820, 2825, 3001, 3002, 3120, 3122, 4030, 4040]

        print(f"Processing data for {len(sites)} sites...")

        for site in sites:
            if 'NB_SCATS_SITE' in traffic_df.columns:
                site_data = traffic_df[traffic_df['NB_SCATS_SITE'] == site].copy()
            else:
                # Create site-specific dummy data
                site_data = traffic_df[traffic_df.index % len(sites) == list(sites).index(site)].copy()

            if len(site_data) < sequence_length + 1:
                continue

            # Sort by time if time column exists
            if 'DT_REPORT' in site_data.columns:
                site_data['datetime'] = pd.to_datetime(site_data['DT_REPORT'], errors='coerce')
                site_data = site_data.sort_values('datetime').dropna(subset=['datetime'])

            # Get traffic flow values
            if 'QT_INTERVAL_COUNT' in site_data.columns:
                flows = site_data['QT_INTERVAL_COUNT'].values
            else:
                # Use dummy flow data
                flows = np.random.normal(500, 200, len(site_data))
                flows = np.maximum(flows, 0)  # Ensure non-negative

            # Create sequences
            for i in range(len(flows) - sequence_length):
                X_sequence = flows[i:i + sequence_length].reshape(-1, 1)
                y_target = flows[i + sequence_length]

                X_list.append(X_sequence)
                y_list.append(y_target)
                site_list.append(site)

        X = np.array(X_list)
        y = np.array(y_list).reshape(-1, 1)

        print(f"Created {len(X)} sequences with shape {X.shape}")
        print(f"Target shape: {y.shape}")

        return X, y, site_list

def convert_to_classification(y_true: np.ndarray, y_pred: np.ndarray,
                            thresholds: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert continuous traffic flow values to categorical labels.

    Args:
        y_true: True traffic flow values
        y_pred: Predicted traffic flow values
        thresholds: Dictionary with 'low', 'medium', 'high' thresholds

    Returns:
        Tuple of (true_labels, pred_labels) as integer arrays
    """
    def categorize(values, thresholds):
        labels = np.zeros(len(values), dtype=int)
        labels[values >= thresholds['low']] = 1  # medium
        labels[values >= thresholds['medium']] = 2  # high
        return labels

    true_labels = categorize(y_true.flatten(), thresholds)
    pred_labels = categorize(y_pred.flatten(), thresholds)

    return true_labels, pred_labels