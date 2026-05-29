"""
Test cases for data loading and preprocessing functionality.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import TrafficDataLoader, convert_to_classification

class TestTrafficDataLoader:
    """Test cases for TrafficDataLoader class."""

    @pytest.fixture
    def data_loader(self):
        """Create a data loader instance with dummy file paths."""
        return TrafficDataLoader(
            traffic_file="dummy_traffic.xls",
            site_file="dummy_sites.xls",
            location_file="dummy_locations.csv"
        )

    def test_initialization(self, data_loader):
        """Test data loader initialization."""
        assert data_loader.traffic_file == "dummy_traffic.xls"
        assert data_loader.site_file == "dummy_sites.xls"
        assert data_loader.location_file == "dummy_locations.csv"

    def test_dummy_traffic_data_creation(self, data_loader):
        """Test creation of dummy traffic data."""
        df = data_loader._create_dummy_traffic_data()

        # Check basic structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'NB_SCATS_SITE' in df.columns
        assert 'QT_INTERVAL_COUNT' in df.columns
        assert 'CD_DAY' in df.columns
        assert 'DT_REPORT' in df.columns

        # Check data quality
        assert df['QT_INTERVAL_COUNT'].min() >= 0  # Traffic flow should be non-negative
        assert len(df['NB_SCATS_SITE'].unique()) > 1  # Multiple sites

    def test_dummy_site_data_creation(self, data_loader):
        """Test creation of dummy site data."""
        df = data_loader._create_dummy_site_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'NB_SCATS_SITE' in df.columns
        assert 'NM_REGION' in df.columns

    def test_dummy_location_data_creation(self, data_loader):
        """Test creation of dummy location data."""
        df = data_loader._create_dummy_location_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'SITE' in df.columns
        assert 'LATITUDE' in df.columns
        assert 'LONGITUDE' in df.columns

        # Check coordinate ranges (Melbourne area)
        assert df['LATITUDE'].between(-38.2, -37.6).all()
        assert df['LONGITUDE'].between(144.6, 145.2).all()

    def test_load_traffic_data_fallback(self, data_loader):
        """Test that load_traffic_data falls back to dummy data when file not found."""
        df = data_loader.load_traffic_data()

        # Should create dummy data when file doesn't exist
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_preprocess_data(self, data_loader):
        """Test data preprocessing for time series."""
        sequence_length = 8
        X, y, site_ids = data_loader.preprocess_data(sequence_length=sequence_length)

        # Check output shapes
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(site_ids, list)

        if len(X) > 0:  # If data was successfully processed
            assert X.shape[1] == sequence_length  # Sequence length dimension
            assert X.shape[2] == 1  # Single feature (traffic flow)
            assert y.shape[1] == 1  # Single target
            assert len(site_ids) == len(X)  # One site ID per sample

            # Check data quality
            assert np.all(X >= 0)  # Traffic flow should be non-negative
            assert np.all(y >= 0)  # Target should be non-negative

    def test_preprocess_data_different_sequence_lengths(self, data_loader):
        """Test preprocessing with different sequence lengths."""
        for seq_len in [4, 8, 16]:
            X, y, site_ids = data_loader.preprocess_data(sequence_length=seq_len)

            if len(X) > 0:
                assert X.shape[1] == seq_len

    def test_preprocess_data_consistency(self, data_loader):
        """Test that preprocessing produces consistent results with same random seed."""
        # First run
        X1, y1, sites1 = data_loader.preprocess_data(sequence_length=8)

        # Second run with new instance (should use same random seed in dummy data)
        data_loader2 = TrafficDataLoader("dummy1.xls", "dummy2.xls", "dummy3.csv")
        X2, y2, sites2 = data_loader2.preprocess_data(sequence_length=8)

        if len(X1) > 0 and len(X2) > 0:
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)

class TestClassificationConversion:
    """Test cases for converting regression to classification."""

    @pytest.fixture
    def sample_traffic_data(self):
        """Create sample traffic flow data."""
        np.random.seed(42)

        # Create data with different traffic levels
        low_traffic = np.random.uniform(0, 400, 50)      # Low traffic
        medium_traffic = np.random.uniform(600, 1200, 50)  # Medium traffic
        high_traffic = np.random.uniform(1600, 2500, 50)   # High traffic

        y_true = np.concatenate([low_traffic, medium_traffic, high_traffic])

        # Add some prediction noise
        noise = np.random.normal(0, 50, len(y_true))
        y_pred = y_true + noise

        return y_true.reshape(-1, 1), y_pred.reshape(-1, 1)

    def test_convert_to_classification(self, sample_traffic_data):
        """Test conversion from regression to classification."""
        y_true, y_pred = sample_traffic_data

        thresholds = {
            'low': 500,
            'medium': 1500,
            'high': float('inf')
        }

        true_labels, pred_labels = convert_to_classification(y_true, y_pred, thresholds)

        # Check output types and shapes
        assert isinstance(true_labels, np.ndarray)
        assert isinstance(pred_labels, np.ndarray)
        assert true_labels.shape == (len(y_true),)
        assert pred_labels.shape == (len(y_pred),)

        # Check label ranges
        assert np.all(true_labels >= 0)
        assert np.all(true_labels <= 2)
        assert np.all(pred_labels >= 0)
        assert np.all(pred_labels <= 2)

    def test_classification_thresholds(self):
        """Test classification with specific threshold values."""
        # Create data with known values
        y_values = np.array([100, 300, 700, 1200, 1800, 2500]).reshape(-1, 1)

        thresholds = {
            'low': 500,
            'medium': 1500,
            'high': float('inf')
        }

        true_labels, _ = convert_to_classification(y_values, y_values, thresholds)

        # Expected labels: [0, 0, 1, 1, 2, 2]
        expected = np.array([0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(true_labels, expected)

    def test_classification_edge_cases(self):
        """Test classification with edge case values."""
        # Test exactly on thresholds
        y_values = np.array([500, 1500]).reshape(-1, 1)

        thresholds = {
            'low': 500,
            'medium': 1500,
            'high': float('inf')
        }

        true_labels, _ = convert_to_classification(y_values, y_values, thresholds)

        # Values >= threshold should be in higher category
        assert true_labels[0] == 1  # 500 >= 500, so medium
        assert true_labels[1] == 2  # 1500 >= 1500, so high

    def test_classification_different_thresholds(self):
        """Test classification with different threshold values."""
        y_values = np.array([200, 800, 1200]).reshape(-1, 1)

        thresholds1 = {'low': 300, 'medium': 1000, 'high': float('inf')}
        thresholds2 = {'low': 600, 'medium': 1500, 'high': float('inf')}

        labels1, _ = convert_to_classification(y_values, y_values, thresholds1)
        labels2, _ = convert_to_classification(y_values, y_values, thresholds2)

        # Should produce different classifications
        assert not np.array_equal(labels1, labels2)

class TestDataQuality:
    """Test data quality and edge cases."""

    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        loader = TrafficDataLoader("dummy.xls", "dummy.xls", "dummy.csv")

        # Override to return empty DataFrame
        original_method = loader.load_traffic_data
        loader.load_traffic_data = lambda: pd.DataFrame()

        X, y, sites = loader.preprocess_data()

        # Should handle empty data gracefully
        assert len(X) == 0
        assert len(y) == 0
        assert len(sites) == 0

    def test_insufficient_data_handling(self):
        """Test handling when there's insufficient data for sequences."""
        loader = TrafficDataLoader("dummy.xls", "dummy.xls", "dummy.csv")

        # Create very small dataset
        small_df = pd.DataFrame({
            'NB_SCATS_SITE': [2000, 2000, 2000],  # Only 3 records for one site
            'QT_INTERVAL_COUNT': [100, 200, 150],
            'DT_REPORT': ['01/01/2006 00:00', '01/01/2006 00:15', '01/01/2006 00:30']
        })

        loader.load_traffic_data = lambda: small_df

        X, y, sites = loader.preprocess_data(sequence_length=8)

        # Should produce no sequences since we don't have enough data
        assert len(X) == 0

    def test_data_consistency_multiple_sites(self):
        """Test data consistency across multiple sites."""
        loader = TrafficDataLoader("dummy.xls", "dummy.xls", "dummy.csv")

        X, y, site_ids = loader.preprocess_data(sequence_length=8)

        if len(X) > 0:
            # Check that we have data from multiple sites
            unique_sites = set(site_ids)
            assert len(unique_sites) > 1

            # Check that each site contributes reasonable amount of data
            site_counts = {}
            for site in site_ids:
                site_counts[site] = site_counts.get(site, 0) + 1

            # Each site should contribute some data
            for count in site_counts.values():
                assert count > 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])