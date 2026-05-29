# Traffic Flow Prediction Model Comparison

This project implements and compares three machine learning models for traffic flow prediction: LSTM, GRU, and Random Forest. The comparison includes both regression and classification metrics as requested.

## 🚀 Quick Start

### Option 1: Run Everything Automatically
```bash
python run_analysis.py
```

This script will:
- Check dependencies
- Run tests (optional)
- Execute model comparison
- Generate visualizations
- Create comprehensive results

### Option 2: Step-by-Step Execution

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run tests (optional but recommended):**
```bash
python -m pytest src/tests/ -v
```

3. **Run model comparison:**
```bash
python model_comparison.py
```

4. **Generate visualizations:**
```bash
python -c "from src.utils.visualization import main; main()"
```

## 📊 What Gets Generated

After running the analysis, you'll get:

### Results Files
- `results/complete_results.json` - Detailed results with all metrics
- `results/model_comparison_summary.csv` - Summary table for easy comparison
- `results/cross_validation_summary.csv` - Cross-validation results

### Visualizations
- `results/plots/metric_comparison.png` - Bar charts comparing all metrics
- `results/plots/time_comparison.png` - Training and inference time comparison
- `results/plots/performance_radar.png` - Radar chart of normalized performance
- `results/plots/cross_validation.png` - Box plots of cross-validation results

### Report
- `REPORT.md` - 80-90% complete academic report ready for submission

## 🏗️ Project Structure

```
📁 src/
├── 📁 data/
│   ├── 📄 data_loader.py          # Data loading and preprocessing
│   └── 📄 __init__.py
├── 📁 models/
│   ├── 📄 base_model.py           # Abstract base class
│   ├── 📄 lstm_model.py           # LSTM implementation
│   ├── 📄 gru_model.py            # GRU implementation
│   ├── 📄 random_forest_model.py  # Random Forest implementation
│   └── 📄 __init__.py
├── 📁 tests/
│   ├── 📄 test_models.py          # Model unit tests
│   ├── 📄 test_data_loader.py     # Data loading tests
│   ├── 📄 test_comparison.py      # Integration tests
│   └── 📄 __init__.py
└── 📁 utils/
    ├── 📄 visualization.py       # Plotting utilities
    └── 📄 __init__.py

📄 model_comparison.py             # Main comparison script
📄 config.py                      # Configuration parameters
📄 run_analysis.py                # Quick setup script
📄 REPORT.md                      # Draft report (80-90% complete)
```

## 🎯 Models Implemented

### 1. LSTM (Long Short-Term Memory)
- **Architecture**: 2-layer LSTM with dropout
- **Parameters**: 64 → 32 units, 0.2 dropout
- **Best for**: Capturing long-term dependencies in traffic patterns

### 2. GRU (Gated Recurrent Unit) 
- **Architecture**: 2-layer GRU with dropout
- **Parameters**: 64 → 32 units, 0.2 dropout
- **Best for**: Faster training while maintaining temporal modeling

### 3. Random Forest
- **Architecture**: Ensemble of 100 decision trees
- **Parameters**: Max depth 10, min samples split 5
- **Best for**: Non-linear patterns without requiring sequential modeling

## 📈 Metrics Evaluated

### Regression Metrics
- **MAE** (Mean Absolute Error) - Average prediction error
- **RMSE** (Root Mean Squared Error) - Penalizes large errors more
- **R²** (R-squared) - Explained variance (higher = better)
- **MAPE** (Mean Absolute Percentage Error) - Relative error percentage

### Classification Metrics
Traffic flow converted to categories:
- **Low**: < 500 vehicles/15min
- **Medium**: 500-1500 vehicles/15min  
- **High**: > 1500 vehicles/15min

Metrics:
- **Accuracy** - Overall correct classification rate
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1-Score** - Harmonic mean of precision and recall

## 🔧 Configuration

Key settings in `config.py`:
- **Sequence Length**: 8 time steps (2 hours of 15-min intervals)
- **Train/Test Split**: 80%/20%
- **Cross-Validation**: 5-fold
- **Neural Network Epochs**: 50 (with early stopping)

## 📝 Data Requirements

The system expects these files in your Downloads folder:
- `Scats Data October 2006.xls` - Traffic flow data
- `SCATSSiteListingSpreadsheet_VicRoads.xls` - Site information
- `Traffic_Count_Locations_with_LONG_LAT.csv` - Location coordinates

**Note**: If these files are not available, the system will automatically generate realistic dummy data for demonstration purposes.

## 🧪 Testing

Comprehensive test suite covers:
- **Unit Tests**: Individual model functionality
- **Integration Tests**: End-to-end pipeline
- **Data Tests**: Data loading and preprocessing
- **Metric Tests**: Calculation accuracy

Run tests with:
```bash
python -m pytest src/tests/ -v
```

## 📊 Expected Results

Based on typical traffic prediction studies:

| Model | Speed | Accuracy | Best Use Case |
|-------|--------|----------|---------------|
| Random Forest | ⭐⭐⭐ | ⭐⭐⭐ | Real-time applications |
| GRU | ⭐⭐ | ⭐⭐⭐⭐ | Balance of speed/accuracy |
| LSTM | ⭐ | ⭐⭐⭐⭐⭐ | Maximum accuracy needed |

## 🎓 For Your Assignment

### What's Already Done (80-90%)
- ✅ Complete model implementations
- ✅ Comprehensive test suite
- ✅ Both regression and classification metrics
- ✅ Cross-validation
- ✅ Visualization generation
- ✅ Report structure with methodology and literature review

### What You Need to Add (10-20%)
1. **Run the analysis** and populate result tables in REPORT.md
2. **Add your interpretation** of the results in the Discussion section
3. **Write conclusions** based on your specific findings
4. **Customize** any parameters or add additional analysis as needed

### For Submission
1. Run `python run_analysis.py`
2. Update REPORT.md with your actual results
3. Include the `results/` folder with your submission
4. Add any additional analysis or insights

## 🆘 Troubleshooting

### TensorFlow Issues
If TensorFlow isn't available:
- The system will only run Random Forest
- Install with: `pip install tensorflow`
- For GPU support on Windows: check TensorFlow DirectML documentation

### Memory Issues
If you run out of memory:
- Reduce batch size in `config.py`
- Reduce number of estimators for Random Forest
- Use fewer epochs for neural networks

### Data Issues
If real data files aren't available:
- System automatically generates dummy data
- Results will be based on synthetic traffic patterns
- This is perfectly fine for demonstrating the methodology

## 📞 Support

If you encounter issues:
1. Check that all dependencies are installed: `python run_analysis.py` (it checks for you)
2. Run tests to verify setup: `python -m pytest src/tests/ -v`
3. Check the error messages - they're designed to be helpful
4. Review the configuration in `config.py`

Good luck with your assignment! 🚀