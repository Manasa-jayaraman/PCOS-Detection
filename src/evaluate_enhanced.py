"""
Enhanced Evaluation Module for PCOS Detection

This module handles:
- Loading saved models with metadata
- Comprehensive model evaluation
- ROC curve and confusion matrix generation
- Performance visualization and plots
- Saving plots to project root or models directory
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
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12


class EnhancedPCOSEvaluator:
    """Enhanced model evaluation and visualization utilities"""
    
    def __init__(self, model_path: str, data_path: str, results_path: str = None):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.results_path = Path(results_path) if results_path else Path("../models")
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.model_data = None
        self.model = None
        self.X_test = None
        self.y_test = None
    
    def load_model_with_metadata(self, model_filename: str = "pcos_model.pkl"):
        """Load trained model with metadata from pickle file"""
        try:
            filepath = self.model_path / model_filename
            with open(filepath, 'rb') as f:
                self.model_data = pickle.load(f)
            
            if isinstance(self.model_data, dict):
                self.model = self.model_data['model']
                logger.info(f"Model loaded: {self.model_data.get('model_name', 'Unknown')}")
                logger.info(f"CV Score: {self.model_data.get('cv_score', 'Unknown')}")
                logger.info(f"Features: {len(self.model_data.get('feature_names', []))}")
            else:
                self.model = self.model_data
                logger.info("Model loaded (no metadata available)")
            
            return True
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def load_test_data(self, data_filename: str = "pcos_processed.csv"):
        """Load test data for evaluation"""
        try:
            filepath = self.data_path / data_filename
            df = pd.read_csv(filepath)
            
            # Get target column from model metadata or auto-detect
            if self.model_data and isinstance(self.model_data, dict):
                target_col = self.model_data.get('target_column')
            else:
                target_candidates = [col for col in df.columns if 'pcos' in col.lower()]
                target_col = target_candidates[0] if target_candidates else None
            
            if not target_col:
                raise ValueError("Could not determine target column")
            
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # For demonstration, we'll use the entire dataset as "test" data
            # In practice, you'd load the actual test set from train/test split
            self.X_test = X
            self.y_test = y
            
            logger.info(f"Loaded test data: {X.shape[0]} samples, {X.shape[1]} features")
            return True
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return False
    
    def evaluate_comprehensive_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        if self.model is None or self.X_test is None:
            raise ValueError("Model and test data must be loaded first")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate all metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)  # Sensitivity
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': recall,
            'specificity': specificity,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        # Print metrics
        print("="*60)
        print("COMPREHENSIVE EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Precision:          {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"Specificity:        {specificity:.4f}")
        print(f"F1-Score:           {f1:.4f}")
        print(f"ROC-AUC:            {roc_auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"True Negatives:  {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives:  {tp}")
        
        return metrics, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_pred, save_path: str = "confusion_matrix.png"):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No PCOS', 'PCOS'],
                   yticklabels=['No PCOS', 'PCOS'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm[i, j]/total:.1%})', 
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        save_filepath = self.results_path / save_path
        plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_filepath}")
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_pred_proba, save_path: str = "roc_curve.png"):
        """Plot and save ROC curve"""
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = _[optimal_idx]
        plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8, 
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
        plt.legend(loc="lower right", fontsize=12)
        
        plt.tight_layout()
        save_filepath = self.results_path / save_path
        plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to: {save_filepath}")
        plt.show()
        
        return fpr, tpr, roc_auc
    
    def plot_precision_recall_curve(self, y_pred_proba, save_path: str = "precision_recall_curve.png"):
        """Plot and save Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=3,
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.tight_layout()
        save_filepath = self.results_path / save_path
        plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to: {save_filepath}")
        plt.show()
        
        return precision, recall, avg_precision
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = "feature_importance.png"):
        """Plot feature importance if available"""
        if not self.model or not hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            logger.warning("Model does not support feature importance visualization")
            return None
        
        # Get feature importance
        classifier = self.model.named_steps['classifier']
        importances = classifier.feature_importances_
        
        # Get feature names from model metadata
        if self.model_data and isinstance(self.model_data, dict):
            feature_names = self.model_data.get('feature_names', [f'Feature_{i}' for i in range(len(importances))])
        else:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Handle feature names after preprocessing
        if len(feature_names) != len(importances):
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.tight_layout()
        
        save_filepath = self.results_path / save_path
        plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to: {save_filepath}")
        plt.show()
        
        return importance_df
    
    def plot_prediction_distribution(self, y_pred_proba, save_path: str = "prediction_distribution.png"):
        """Plot distribution of prediction probabilities"""
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Histogram of probabilities by true class
        plt.subplot(1, 2, 1)
        pcos_probs = y_pred_proba[self.y_test == 1]
        no_pcos_probs = y_pred_proba[self.y_test == 0]
        
        plt.hist(no_pcos_probs, bins=30, alpha=0.7, label='No PCOS', color='skyblue', density=True)
        plt.hist(pcos_probs, bins=30, alpha=0.7, label='PCOS', color='salmon', density=True)
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Prediction Probability Distribution', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Box plot
        plt.subplot(1, 2, 2)
        data_for_box = [no_pcos_probs, pcos_probs]
        plt.boxplot(data_for_box, labels=['No PCOS', 'PCOS'])
        plt.ylabel('Predicted Probability', fontsize=12)
        plt.title('Probability Distribution by True Class', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_filepath = self.results_path / save_path
        plt.savefig(save_filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction distribution plot saved to: {save_filepath}")
        plt.show()
    
    def generate_evaluation_report(self, save_report: bool = True):
        """Generate comprehensive evaluation report"""
        if not self.model or not self.X_test:
            raise ValueError("Model and test data must be loaded first")
        
        # Get comprehensive metrics
        metrics, y_pred, y_pred_proba = self.evaluate_comprehensive_metrics()
        
        # Generate all plots
        logger.info("Generating evaluation plots...")
        
        # Confusion Matrix
        cm = self.plot_confusion_matrix(y_pred)
        
        # ROC Curve
        fpr, tpr, roc_auc = self.plot_roc_curve(y_pred_proba)
        
        # Precision-Recall Curve
        precision, recall, avg_precision = self.plot_precision_recall_curve(y_pred_proba)
        
        # Feature Importance (if available)
        feature_importance = self.plot_feature_importance()
        
        # Prediction Distribution
        self.plot_prediction_distribution(y_pred_proba)
        
        # Save comprehensive report
        if save_report:
            report_path = self.results_path / 'evaluation_report.txt'
            with open(report_path, 'w') as f:
                f.write("PCOS Detection Model - Enhanced Evaluation Report\n")
                f.write("=" * 60 + "\n\n")
                
                # Model info
                if self.model_data and isinstance(self.model_data, dict):
                    f.write(f"Model: {self.model_data.get('model_name', 'Unknown')}\n")
                    f.write(f"CV Score: {self.model_data.get('cv_score', 'Unknown'):.4f}\n")
                    f.write(f"Features: {len(self.model_data.get('feature_names', []))}\n\n")
                
                # Metrics
                f.write("Performance Metrics:\n")
                f.write("-" * 30 + "\n")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"{metric.upper()}: {value:.4f}\n")
                    else:
                        f.write(f"{metric.upper()}: {value}\n")
                
                f.write(f"\nDetailed Classification Report:\n")
                f.write(classification_report(self.y_test, y_pred))
                
                f.write(f"\nGenerated Plots:\n")
                f.write("- confusion_matrix.png\n")
                f.write("- roc_curve.png\n")
                f.write("- precision_recall_curve.png\n")
                f.write("- feature_importance.png\n")
                f.write("- prediction_distribution.png\n")
            
            logger.info(f"Evaluation report saved to: {report_path}")
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'feature_importance': feature_importance
        }
    
    def run_complete_evaluation(self, model_filename: str = "pcos_model.pkl", 
                              data_filename: str = "pcos_processed.csv"):
        """Run complete evaluation pipeline"""
        logger.info("Starting comprehensive model evaluation...")
        
        # Load model
        if not self.load_model_with_metadata(model_filename):
            raise ValueError("Failed to load model")
        
        # Load test data
        if not self.load_test_data(data_filename):
            raise ValueError("Failed to load test data")
        
        # Generate evaluation report
        results = self.generate_evaluation_report()
        
        logger.info("\n" + "="*60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("Generated files:")
        logger.info("- confusion_matrix.png")
        logger.info("- roc_curve.png")
        logger.info("- precision_recall_curve.png")
        logger.info("- feature_importance.png")
        logger.info("- prediction_distribution.png")
        logger.info("- evaluation_report.txt")
        
        return results


def main():
    """Main function to run evaluation"""
    # Set up paths
    model_path = "../models"
    data_path = "../data/processed"
    results_path = "../models"  # Save plots in models directory
    
    # Initialize evaluator
    evaluator = EnhancedPCOSEvaluator(model_path, data_path, results_path)
    
    # Run complete evaluation
    try:
        results = evaluator.run_complete_evaluation()
        print("\nEvaluation completed successfully!")
        print("Check the models/ directory for generated plots and reports.")
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
