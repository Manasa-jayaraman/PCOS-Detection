"""
Evaluation Module for PCOS Detection

This module handles:
- Model evaluation metrics
- Visualization of results
- Performance analysis
- ROC curves, confusion matrices, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import learning_curve, validation_curve
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette('husl')


class PCOSModelEvaluator:
    """Model evaluation and visualization utilities"""
    
    def __init__(self, model_path: str, results_path: str = None):
        self.model_path = Path(model_path)
        self.results_path = Path(results_path) if results_path else Path("../results")
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.model = None
    
    def load_model(self, model_filename: str):
        """Load trained model from pickle file"""
        try:
            filepath = self.model_path / model_filename
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from: {filepath}")
            return self.model
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def evaluate_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = auc(*roc_curve(y_true, y_pred_proba)[:2])
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        logger.info("Classification Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
        
        return metrics, class_report
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names or ['No PCOS', 'PCOS'],
                   yticklabels=class_names or ['No PCOS', 'PCOS'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(self.results_path / save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(self.results_path / save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fpr, tpr, roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(self.results_path / save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return precision, recall, avg_precision
    
    def plot_feature_importance(self, model, feature_names, top_n=20, save_path=None):
        """Plot feature importance"""
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not support feature importance")
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.results_path / save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def plot_learning_curves(self, model, X, y, cv=5, save_path=None):
        """Plot learning curves to analyze bias/variance"""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('ROC AUC Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(self.results_path / save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return train_sizes, train_scores, val_scores
    
    def plot_validation_curve(self, model, X, y, param_name, param_range, cv=5, save_path=None):
        """Plot validation curve for hyperparameter analysis"""
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring='roc_auc', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel(param_name)
        plt.ylabel('ROC AUC Score')
        plt.title(f'Validation Curve - {param_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(self.results_path / save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return train_scores, val_scores
    
    def create_evaluation_report(self, X_test, y_test, model=None, save_report=True):
        """Create comprehensive evaluation report"""
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model provided or loaded")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics, class_report = self.evaluate_classification_metrics(y_test, y_pred, y_pred_proba)
        
        # Create visualizations
        logger.info("Creating evaluation plots...")
        
        # Confusion Matrix
        cm = self.plot_confusion_matrix(y_test, y_pred, save_path='confusion_matrix.png')
        
        # ROC Curve
        if y_pred_proba is not None:
            fpr, tpr, roc_auc = self.plot_roc_curve(y_test, y_pred_proba, save_path='roc_curve.png')
            
            # Precision-Recall Curve
            precision, recall, avg_precision = self.plot_precision_recall_curve(
                y_test, y_pred_proba, save_path='precision_recall_curve.png'
            )
        
        # Feature Importance
        feature_importance = self.plot_feature_importance(
            model, X_test.columns, save_path='feature_importance.png'
        )
        
        # Compile report
        report = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance.to_dict() if feature_importance is not None else None
        }
        
        if y_pred_proba is not None:
            report.update({
                'roc_auc': roc_auc,
                'average_precision': avg_precision
            })
        
        # Save report
        if save_report:
            report_path = self.results_path / 'evaluation_report.txt'
            with open(report_path, 'w') as f:
                f.write("PCOS Detection Model - Evaluation Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Classification Metrics:\n")
                for metric, value in metrics.items():
                    f.write(f"{metric.upper()}: {value:.4f}\n")
                
                f.write(f"\nDetailed Classification Report:\n")
                f.write(classification_report(y_test, y_pred))
                
            logger.info(f"Evaluation report saved to: {report_path}")
        
        return report
    
    def compare_models(self, models_dict, X_test, y_test):
        """Compare multiple models performance"""
        comparison_results = {}
        
        plt.figure(figsize=(12, 8))
        
        for i, (model_name, model) in enumerate(models_dict.items()):
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            metrics, _ = self.evaluate_classification_metrics(y_test, y_pred, y_pred_proba)
            comparison_results[model_name] = metrics
            
            # ROC Curve
            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {metrics["roc_auc"]:.3f})')
        
        # Plot ROC curves
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_path / 'model_comparison_roc.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results).T
        logger.info("Model Comparison Results:")
        logger.info(comparison_df)
        
        return comparison_df


def main():
    """Example usage of evaluation module"""
    # Configuration
    model_path = "../models"
    data_path = "../data/processed"
    model_filename = "pcos_model.pkl"
    data_filename = "pcos_processed.csv"
    target_column = "PCOS"
    
    # Initialize evaluator
    evaluator = PCOSModelEvaluator(model_path)
    
    try:
        # Load model
        model = evaluator.load_model(model_filename)
        
        # Load test data (you would typically have this from train/test split)
        # For demo purposes, we'll load the processed data
        df = pd.read_csv(Path(data_path) / data_filename)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Create evaluation report
        report = evaluator.create_evaluation_report(X, y, model)
        
        print("Evaluation completed successfully!")
        print("Check the results directory for generated plots and reports.")
        
    except Exception as e:
        print(f"Error in evaluation: {e}")


if __name__ == "__main__":
    main()
