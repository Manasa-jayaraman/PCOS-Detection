# PCOS Detection System

A comprehensive machine learning system for detecting Polycystic Ovary Syndrome (PCOS) using patient medical data. This project includes data processing, model training, evaluation, and a user-friendly web interface.

## 🏥 About PCOS

Polycystic Ovary Syndrome (PCOS) is a hormonal disorder affecting women of reproductive age. Early detection and proper management are crucial for preventing long-term health complications.

## 🚀 Features

- **Data Processing Pipeline**: Automated data cleaning, preprocessing, and feature engineering
- **Multiple ML Models**: Support for Random Forest, Gradient Boosting, Logistic Regression, and SVM
- **Hyperparameter Tuning**: Grid search with cross-validation for optimal model performance
- **Comprehensive Evaluation**: ROC curves, confusion matrices, feature importance analysis
- **Web Interface**: User-friendly Streamlit app for predictions
- **CLI Tools**: Command-line interface for batch predictions
- **Jupyter Notebooks**: Interactive exploratory data analysis

## 📁 Project Structure

```
pcos-detection/
├── data/
│   ├── raw/                    # Original dataset (keep read-only)
│   └── processed/              # Cleaned CSVs used for modeling
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory Data Analysis
├── src/
│   ├── data_processing.py      # Data loading, cleaning, and preprocessing
│   ├── features.py            # Feature engineering utilities
│   ├── train.py               # Model training and hyperparameter tuning
│   ├── evaluate.py            # Model evaluation and visualization
│   └── predict.py             # CLI prediction interface
├── models/
│   └── pcos_model.pkl         # Trained model (generated after training)
├── app/
│   └── pcos_app.py       # Web application interface
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                # Git ignore rules
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pcos-detection
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv pcos_env
   source pcos_env/bin/activate  # On Windows: pcos_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### 1. Data Preparation

Place your PCOS dataset in the `data/raw/` directory. The dataset should include features such as:
- Patient demographics (age, weight, height, BMI)
- Medical history (menstrual cycle, pregnancy history)
- Hormone levels (FSH, LH, TSH, AMH, etc.)
- Symptoms (hair growth, skin darkening, weight gain)
- Ultrasound parameters (follicle counts, sizes)

### 2. Data Processing

Process and clean your raw data:

```bash
cd src
python data_processing.py
```

This will:
- Load raw data from `data/raw/`
- Clean and preprocess the data
- Handle missing values and outliers
- Save processed data to `data/processed/`

### 3. Exploratory Data Analysis

Open and run the Jupyter notebook for EDA:

```bash
jupyter notebook notebooks/01_EDA.ipynb
```

### 4. Model Training

Train multiple models with hyperparameter tuning:

```bash
cd src
python train.py
```

This will:
- Load processed data
- Train multiple ML models (Random Forest, Gradient Boosting, etc.)
- Perform grid search for hyperparameter optimization
- Save the best model to `models/pcos_model.pkl`

### 5. Model Evaluation

Evaluate the trained model:

```bash
cd src
python evaluate.py
```

This generates:
- Performance metrics (accuracy, precision, recall, F1-score, AUC)
- Visualization plots (ROC curves, confusion matrices)
- Feature importance analysis

### 6. Making Predictions

#### CLI Interface

**Single prediction with sample data**:
```bash
cd src
python predict.py --model pcos_model.pkl --sample
```

**Interactive single prediction**:
```bash
cd src
python predict.py --model pcos_model.pkl --single
```

**Batch predictions**:
```bash
cd src
python predict.py --model pcos_model.pkl --input ../data/new_patients.csv --output predictions.csv
```

**Show required features**:
```bash
cd src
python predict.py --model pcos_model.pkl --features
```

### Web Interface

Launch the Streamlit web application:

```bash
cd app
streamlit run pcos_app.py
```

The web interface provides:
- Interactive form for patient data input
- Real-time PCOS prediction with confidence scores
- Batch file upload for multiple predictions
- Educational information about PCOS
- Model information and disclaimers

## 🔧 Configuration

### Feature Engineering

Modify the feature engineering configuration in `src/features.py`:

```python
config = {
    'bmi': {
        'create': True,
        'weight_col': 'Weight',
        'height_col': 'Height'
    },
    'hormone_ratios': {
        'create': True,
        'hormone_cols': ['LH', 'FSH', 'TSH', 'Testosterone']
    },
    # ... other configurations
}
```

### Model Parameters

Adjust model hyperparameters in `src/train.py`:

```python
'RandomForest': {
    'model': RandomForestClassifier(random_state=42),
    'params': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        # ... other parameters
    }
}
```

## 📈 Model Performance

The system evaluates models using multiple metrics:

- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Precision-Recall AUC**: Area under the PR curve

## 🔬 Feature Engineering

The system includes advanced feature engineering:

- **BMI Categories**: Underweight, Normal, Overweight, Obese
- **Hormone Ratios**: LH/FSH ratio, TSH/T3 ratio, etc.
- **Age Groups**: Categorical age ranges
- **Interaction Features**: Cross-feature interactions
- **Statistical Features**: Group-based aggregations
- **Polynomial Features**: Non-linear feature combinations

## 📋 Data Requirements

### Required Features

The model expects the following features (adjust based on your dataset):

- `Age`: Patient age in years
- `Weight`: Weight in kg
- `Height`: Height in cm
- `BMI`: Body Mass Index
- `Blood_Group`: Encoded blood group
- `Pulse_rate`: Heart rate in bpm
- `RR`: Respiratory rate
- `Hb`: Hemoglobin level
- `Cycle`: Menstrual cycle regularity (0/1)
- `Cycle_length`: Cycle length in days
- `Marriage_Status`: Marital status (0/1)
- `Pregnant`: Pregnancy status (0/1)
- `No_of_abortions`: Number of abortions
- Hormone levels: `FSH`, `LH`, `TSH`, `AMH`, `PRL`, `Vit_D3`
- Blood parameters: `PRG`, `RBS`
- Symptoms: `Weight_gain`, `hair_growth`, `Skin_darkening`, etc.
- Lifestyle: `Fast_food`, `Reg_Exercise`
- Vital signs: `BP_Systolic`, `BP_Diastolic`
- Ultrasound: `Follicle_No_L`, `Follicle_No_R`, `Avg_F_size_L`, etc.

### Target Variable

- `PCOS`: Binary target (0 = No PCOS, 1 = PCOS)

## ⚠️ Important Disclaimers

**Medical Disclaimer**: This tool is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

**Model Limitations**: 
- Predictions may not be 100% accurate
- False positives and negatives are possible
- Model performance depends on training data quality
- Regular model retraining may be necessary

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request


---

**Remember**: This is a research tool. Always consult healthcare professionals for medical advice and diagnosis.
