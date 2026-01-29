"""
Script to evaluate an existing PCOS detection model.
This will load the model and display its performance metrics.
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def load_model_and_data(model_path, data_path=None):
    """Load the trained model and optionally the dataset. Returns (model, feature_names, df)
    The model pickle may contain a dict with keys 'model' and 'feature_names' or be a raw model object.
    """
    print(f"Loading model from: {model_path}")

    # Load the model file
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Support both dict-containing metadata and raw model objects
    if isinstance(model_data, dict):
        model = model_data.get('model')
        feature_names = model_data.get('feature_names')
    else:
        model = model_data
        feature_names = None

    print(f"Model loaded successfully. Type: {type(model).__name__}")

    df = None
    if data_path:
        data_path = Path(data_path)
        if data_path.exists():
            print(f"\nLoading dataset from: {data_path}")
            df = pd.read_csv(data_path)
            print(f"Dataset shape: {df.shape}")
        else:
            print(f"Data file not found: {data_path}")

    return model, feature_names, df

def evaluate_model(model, X_test, y_test, X_train=None, y_train=None):
    """Evaluate the model and print performance metrics"""
    # Make predictions on test set
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Calculate test metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test, zero_division=0),
        'recall': recall_score(y_test, y_pred_test, zero_division=0),
        'f1': f1_score(y_test, y_pred_test, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_test),
    }
    
    # Calculate test specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    test_metrics['specificity'] = tn / (tn + fp)
    
    # If training data is provided, calculate training metrics
    train_metrics = None
    if X_train is not None and y_train is not None:
        y_pred_train = model.predict(X_train)
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, zero_division=0),
            'recall': recall_score(y_train, y_pred_train, zero_division=0),
            'f1': f1_score(y_train, y_pred_train, zero_division=0),
            'roc_auc': roc_auc_score(y_train, y_pred_proba_train),
        }
        
        # Calculate training specificity
        tn_train, fp_train, _, _ = confusion_matrix(y_train, y_pred_train).ravel()
        train_metrics['specificity'] = tn_train / (tn_train + fp_train)
    
    return {
        'test_metrics': test_metrics,
        'train_metrics': train_metrics,
        'confusion_matrix': confusion_matrix(y_test, y_pred_test),
        'classification_report': classification_report(y_test, y_pred_test)
    }

def print_metrics(metrics, dataset_name):
    """Print the evaluation metrics in a formatted way"""
    print(f"\n{dataset_name.upper()} PERFORMANCE:")
    print("-" * 50)
    
    if metrics is None:
        print("No metrics available")
        return
    
    # Print main metrics
    print(f"{'Accuracy:':<15} {metrics['accuracy']:.4f}")
    print(f"{'Precision:':<15} {metrics['precision']:.4f}")
    print(f"{'Recall:':<15} {metrics['recall']:.4f}")
    print(f"{'F1 Score:':<15} {metrics['f1']:.4f}")
    print(f"{'Specificity:':<15} {metrics['specificity']:.4f}")
    print(f"{'ROC-AUC:':<15} {metrics['roc_auc']:.4f}")

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Evaluate saved PCOS model and print/save metrics')
    parser.add_argument('--model', '-m', default=str(project_root / 'models' / 'pcos_model.pkl'), help='Path to saved model pickle')
    parser.add_argument('--data', '-d', default=str(project_root / 'data' / 'processed' / 'pcos_processed.csv'), help='Path to processed CSV to use')
    parser.add_argument('--no-split', action='store_true', help='Do not split dataset; evaluate on provided data as test set (no train metrics)')
    parser.add_argument('--output', '-o', help='Optional path to write JSON metrics report')
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    # Load model and full dataset
    model, feature_names, df = load_model_and_data(model_path, data_path)
    if df is None:
        print("No dataset available to evaluate. Place a processed CSV at:", data_path)
        return

    # Auto-detect target column if present
    target_candidates = [c for c in df.columns if c.lower().startswith('pcos') or c.lower().endswith('pcos')]
    if 'pcos_y_n' in df.columns:
        target_col = 'pcos_y_n'
    elif target_candidates:
        target_col = target_candidates[0]
    else:
        # Try common names
        for cand in ['PCOS', 'pcos', 'target']:
            if cand in df.columns:
                target_col = cand
                break
        else:
            print("Could not detect target column in dataset. Available columns:", df.columns.tolist())
            return

    print(f"Using target column: {target_col}")

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])

    # Determine feature set
    if feature_names is not None:
        features = feature_names
    else:
        features = [c for c in df.columns if c != target_col]

    # Ensure features exist in dataframe
    missing = [f for f in features if f not in df.columns]
    if missing:
        print("Warning: some feature names from the model are missing in the dataset. They will be ignored:", missing)
        features = [f for f in features if f in df.columns]

    X = df[features]
    y = df[target_col].astype(int)

    # If --no-split: evaluate on provided dataset as test set only
    from sklearn.model_selection import train_test_split
    results = None
    if args.no_split:
        print("\nEvaluating model on provided dataset as test set (no train/test split)")
        results = evaluate_model(model, X, y, X_train=None, y_train=None)
        X_train = None
        X_test = X
        y_train = None
        y_test = y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"\nData split for evaluation: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")
        # Evaluate the model using our evaluate_model helper
        results = evaluate_model(model, X_test, y_test, X_train=X_train, y_train=y_train)
    
    # Print results
    print("\n" + "="*60)
    print("PCOS DETECTION MODEL EVALUATION")
    print("="*60)
    
    # Print test metrics
    print_metrics(results['test_metrics'], 'Test Set')
    
    # Print confusion matrix
    print("\nCONFUSION MATRIX (Test Set):")
    print("-" * 30)
    print("           Predicted")
    print("           No PCOS  PCOS")
    print(f"Actual No  {results['confusion_matrix'][0][0]:<8} {results['confusion_matrix'][0][1]}")
    print(f"      Yes  {results['confusion_matrix'][1][0]:<8} {results['confusion_matrix'][1][1]}")
    
    # Print classification report
    print("\nCLASSIFICATION REPORT (Test Set):")
    print("-" * 30)
    print(results['classification_report'])
    
    # Print training metrics if available
    if results.get('train_metrics'):
        print_metrics(results['train_metrics'], 'Training Set')
        # Training confusion matrix
        try:
            y_train_pred = model.predict(X_train)
            cm_train = confusion_matrix(y_train, y_train_pred)
            print("\nCONFUSION MATRIX (Training Set):")
            print("-" * 30)
            print("           Predicted")
            print("           No PCOS  PCOS")
            print(f"Actual No  {cm_train[0][0]:<8} {cm_train[0][1]}")
            print(f"      Yes  {cm_train[1][0]:<8} {cm_train[1][1]}")
            print("\nCLASSIFICATION REPORT (Training Set):")
            print("-" * 30)
            print(classification_report(y_train, y_train_pred))
        except Exception as e:
            print(f"Could not print training confusion matrix/report: {e}")
    
    print("\nEvaluation complete!")

    # Optionally save JSON report
    if args.output:
        try:
            report = {
                'model': str(model_path),
                'data': str(data_path),
                'target_column': target_col,
                'test_metrics': results.get('test_metrics') if results else None,
                'train_metrics': results.get('train_metrics') if results else None,
            }
            # Convert any numpy arrays to lists
            if report['test_metrics'] and 'confusion_matrix' in report['test_metrics']:
                pass
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
            print(f"\nSaved metrics report to: {out_path}")
        except Exception as e:
            print(f"Failed to write report to {args.output}: {e}")

if __name__ == "__main__":
    main()
