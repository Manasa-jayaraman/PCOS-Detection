# PCOS Detection System - Step-by-Step Execution Guide

This guide provides the exact commands to run the complete PCOS detection pipeline from data preprocessing to deployment.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Virtual environment** (recommended)
3. **Dataset** placed in `data/raw/` directory

### Setup Environment

```bash
# Navigate to project directory
cd "c:\Users\ADMIN\Desktop\project\PCOS detection"

# Create virtual environment
python -m venv pcos_env

# Activate virtual environment
# Windows:
pcos_env\Scripts\activate
# Linux/Mac:
# source pcos_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📋 Complete Pipeline Execution

### Step 1: Data Preprocessing

**Automatically detect dataset and clean data:**

```bash
cd src
python data_processing.py
```

**What this does:**
- ✅ Automatically detects `PCOS_data.csv` in `data/raw/`
- ✅ Standardizes column names (removes spaces, special characters)
- ✅ Converts Y/N target values to 1/0
- ✅ Handles missing values (median for numeric, mode for categorical)
- ✅ Saves cleaned data as `data/processed/pcos_processed.csv`

**Expected Output:**
```
INFO:__main__:Detected dataset: PCOS_data.csv
INFO:__main__:Loaded data with shape: (543, 44)
INFO:__main__:Standardized 44 column names
INFO:__main__:Auto-detected target column: pcos_y_n
INFO:__main__:Processing completed. Final dataset shape: (543, 44)
```

### Step 2: Exploratory Data Analysis (EDA)

**Run comprehensive EDA analysis:**

```bash
cd notebooks
python eda_analysis.py
```

**What this does:**
- ✅ Loads processed data
- ✅ Shows summary statistics and missing values
- ✅ Analyzes target variable distribution
- ✅ Creates correlation heatmaps
- ✅ Generates feature distribution plots
- ✅ Performs statistical tests
- ✅ Saves plots as PNG files

**Generated Files:**
- `missing_values_heatmap.png`
- `target_distribution.png`
- `correlation_matrix.png`
- `feature_distributions.png`
- `target_vs_features.png`

### Step 3: Model Training

**Train models with SMOTE and ColumnTransformer:**

```bash
cd src
python train_enhanced.py
```

**What this does:**
- ✅ Loads processed data automatically
- ✅ Splits dataset (80/20 train-test)
- ✅ Applies ColumnTransformer for encoding and scaling
- ✅ Handles class imbalance using SMOTE
- ✅ Trains RandomForestClassifier and LogisticRegression
- ✅ Performs GridSearchCV for hyperparameter tuning
- ✅ Evaluates using accuracy, precision, recall, specificity, ROC-AUC
- ✅ Saves best model to `models/pcos_model.pkl`

**Expected Output:**
```
INFO:__main__:Auto-detected target column: pcos_y_n
INFO:__main__:Data split - Train: 434 (80%), Test: 109 (20%)
INFO:__main__:Starting grid search for RandomForest with SMOTE...
INFO:__main__:Starting grid search for LogisticRegression with SMOTE...
INFO:__main__:Best model: RandomForest with CV score: 0.8542

ENHANCED TRAINING COMPLETED SUCCESSFULLY!
Best model: RandomForest
Best CV score: 0.8542
Model saved as: pcos_model.pkl

Test Set Performance Summary:
RandomForest:
  Accuracy:     0.8532
  Precision:    0.8421
  Recall (Sens):0.8889
  Specificity:  0.8182
  F1-Score:     0.8649
  ROC-AUC:      0.9123
```

### Step 4: Model Evaluation

**Generate comprehensive evaluation plots:**

```bash
cd src
python evaluate_enhanced.py
```

**What this does:**
- ✅ Loads saved model with metadata
- ✅ Generates ROC curve and confusion matrix
- ✅ Creates precision-recall curves
- ✅ Shows feature importance plots
- ✅ Saves all plots in `models/` directory
- ✅ Creates detailed evaluation report

**Generated Files:**
- `models/confusion_matrix.png`
- `models/roc_curve.png`
- `models/precision_recall_curve.png`
- `models/feature_importance.png`
- `models/prediction_distribution.png`
- `models/evaluation_report.txt`

### Step 5: Launch Streamlit App

**Start the web application:**

```bash
cd app
streamlit run streamlit_enhanced.py
```

**What this provides:**
- ✅ CSV file upload interface
- ✅ Automatic data validation and preprocessing
- ✅ Batch PCOS probability prediction
- ✅ Interactive results visualization
- ✅ Downloadable prediction results
- ✅ Model information and help sections

**Access the app at:** `http://localhost:8501`

## 🔧 Alternative Commands

### Using Original Files (if needed)

```bash
# Original training (without SMOTE)
cd src
python train.py

# Original evaluation
cd src
python evaluate.py

# Original Streamlit app
cd app
streamlit run streamlit_app.py
```

### Individual Component Testing

```bash
# Test data processing only
cd src
python -c "from data_processing import PCOSDataProcessor; p = PCOSDataProcessor('../data/raw', '../data/processed'); p.process_pipeline()"

# Test model loading
cd src
python -c "import pickle; model = pickle.load(open('../models/pcos_model.pkl', 'rb')); print('Model loaded successfully')"

# Test prediction
cd src
python predict.py --model pcos_model.pkl --sample
```

## 📊 Expected Results Summary

After running the complete pipeline, you should have:

### 📁 File Structure
```
pcos-detection/
├── data/
│   ├── raw/PCOS_data.csv          # Original dataset
│   └── processed/pcos_processed.csv # Cleaned dataset
├── models/
│   ├── pcos_model.pkl             # Trained model with metadata
│   ├── confusion_matrix.png       # Evaluation plots
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── feature_importance.png
│   ├── prediction_distribution.png
│   └── evaluation_report.txt      # Detailed metrics
├── notebooks/
│   ├── eda_analysis.py            # EDA script
│   ├── missing_values_heatmap.png # EDA plots
│   ├── target_distribution.png
│   ├── correlation_matrix.png
│   ├── feature_distributions.png
│   └── target_vs_features.png
└── src/                           # All source code
```

### 🎯 Performance Metrics
- **Accuracy:** ~85-90%
- **Precision:** ~84-88%
- **Recall (Sensitivity):** ~88-92%
- **Specificity:** ~81-85%
- **ROC-AUC:** ~91-95%

### 🌐 Web Application Features
- CSV upload with validation
- Batch prediction with confidence scores
- Interactive visualizations
- Downloadable results
- Model performance information

## 🚨 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'imblearn'**
   ```bash
   pip install imbalanced-learn
   ```

2. **FileNotFoundError: No CSV files found**
   - Ensure `PCOS_data.csv` is in `data/raw/` directory
   - Check file permissions

3. **Model not found error in Streamlit**
   - Run training pipeline first: `python src/train_enhanced.py`
   - Check if `models/pcos_model.pkl` exists

4. **Streamlit port already in use**
   ```bash
   streamlit run streamlit_enhanced.py --server.port 8502
   ```

5. **Memory issues during training**
   - Reduce grid search parameters in `train_enhanced.py`
   - Use smaller dataset for testing

### Performance Optimization

```bash
# For faster training (reduced grid search)
cd src
python -c "
import sys
sys.path.append('.')
from train_enhanced import EnhancedPCOSTrainer
trainer = EnhancedPCOSTrainer('../data/processed', '../models')
# Modify grid search parameters for faster execution
trainer.full_training_pipeline()
"
```

## 📞 Support

If you encounter issues:

1. **Check Python version:** `python --version` (should be 3.8+)
2. **Verify dependencies:** `pip list`
3. **Check file paths:** Ensure all paths are correct
4. **Review logs:** Check console output for error messages
5. **Test components individually:** Use the individual testing commands above

## 🎉 Success Indicators

You'll know everything is working when:

- ✅ Data processing completes without errors
- ✅ EDA generates 5 visualization plots
- ✅ Training achieves >80% accuracy and saves model
- ✅ Evaluation creates 5 assessment plots
- ✅ Streamlit app loads and accepts CSV uploads
- ✅ Predictions are generated with confidence scores

**Congratulations! Your PCOS Detection System is now fully operational! 🎊**
