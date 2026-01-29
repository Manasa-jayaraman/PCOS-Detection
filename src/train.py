"""
Training Module for PCOS Detection

This module handles:
- Model training with multiple algorithms
- Hyperparameter tuning using GridSearch
- Cross-validation
- Model persistence
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCOSModelTrainer:
    """Main class for PCOS model training and evaluation"""
    
    def __init__(self, data_path: str, model_save_path: str):
        self.data_path = Path(data_path)
        self.model_save_path = Path(model_save_path)
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_processed_data(self, filename: str, target_col: str):
        """Load processed data for training"""
        try:
            filepath = self.data_path / filename
            df = pd.read_csv(filepath)
            
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            logger.info(f"Loaded data with {X.shape[0]} samples and {X.shape[1]} features")
            return X, y
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split - Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_model_configs(self):
        """Get model configurations for grid search"""
        return {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
    
    def train_model_with_gridsearch(self, model_name: str, cv_folds: int = 5):
        """Train a single model with grid search"""
        model_configs = self.get_model_configs()
        
        if model_name not in model_configs:
            raise ValueError(f"Model {model_name} not supported")
        
        config = model_configs[model_name]
        model = config['model']
        param_grid = config['params']
        
        logger.info(f"Starting grid search for {model_name}...")
        
        # Stratified K-Fold for better cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Store the best model
        self.models[model_name] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"{model_name} - Best CV Score: {grid_search.best_score_:.4f}")
        logger.info(f"{model_name} - Best Params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_all_models(self, cv_folds: int = 5):
        """Train all models with grid search"""
        model_configs = self.get_model_configs()
        
        for model_name in model_configs.keys():
            try:
                self.train_model_with_gridsearch(model_name, cv_folds)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Find the best model
        self._select_best_model()
    
    def _select_best_model(self):
        """Select the best performing model"""
        best_score = 0
        best_model_name = None
        
        for model_name, model_info in self.models.items():
            if model_info['best_score'] > best_score:
                best_score = model_info['best_score']
                best_model_name = model_name
        
        if best_model_name:
            self.best_model = self.models[best_model_name]['model']
            self.best_score = best_score
            logger.info(f"Best model: {best_model_name} with CV score: {best_score:.4f}")
        else:
            logger.warning("No best model found")
    
    def evaluate_model(self, model, model_name: str):
        """Evaluate a trained model on test set"""
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        logger.info(f"\n{model_name} Test Set Evaluation:")
        logger.info(f"AUC Score: {auc_score:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(self.y_test, y_pred)}")
        logger.info(f"\nConfusion Matrix:\n{confusion_matrix(self.y_test, y_pred)}")
        
        return {
            'auc_score': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        results = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            results[model_name] = self.evaluate_model(model, model_name)
        
        return results
    
    def save_model(self, model, filename: str):
        """Save trained model to pickle file"""
        try:
            self.model_save_path.mkdir(parents=True, exist_ok=True)
            filepath = self.model_save_path / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Model saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def save_best_model(self, filename: str = "pcos_model.pkl"):
        """Save the best performing model"""
        if self.best_model is not None:
            self.save_model(self.best_model, filename)
        else:
            logger.warning("No best model to save")
    
    def cross_validate_model(self, model, cv_folds: int = 5):
        """Perform cross-validation on a model"""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='roc_auc')
        
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def get_feature_importance(self, model, feature_names):
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 most important features:")
            logger.info(importance_df.head(10))
            
            return importance_df
        else:
            logger.warning("Model does not support feature importance")
            return None
    
    def full_training_pipeline(self, data_filename: str, target_col: str, 
                             model_filename: str = "pcos_model.pkl"):
        """Complete training pipeline"""
        logger.info("Starting full training pipeline...")
        
        # Load data
        X, y = self.load_processed_data(data_filename, target_col)
        
        # Split data
        self.split_data(X, y)
        
        # Train all models
        self.train_all_models()
        
        # Evaluate all models
        results = self.evaluate_all_models()
        
        # Save best model
        self.save_best_model(model_filename)
        
        # Get feature importance for best model
        if self.best_model:
            feature_importance = self.get_feature_importance(self.best_model, X.columns)
        
        logger.info("Training pipeline completed successfully!")
        # If a best model was selected, compute and print train/test metrics
        if self.best_model is not None and self.X_train is not None and self.X_test is not None:
            try:
                # Ensure y arrays are in integer form
                y_train = self.y_train.astype(int)
                y_test = self.y_test.astype(int)

                # Predictions
                y_train_pred = self.best_model.predict(self.X_train)
                y_test_pred = self.best_model.predict(self.X_test)

                # Training metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                train_precision = precision_score(y_train, y_train_pred, zero_division=0)
                train_recall = recall_score(y_train, y_train_pred, zero_division=0)
                train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

                # Testing metrics
                test_accuracy = accuracy_score(y_test, y_test_pred)
                test_precision = precision_score(y_test, y_test_pred, zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

                # Print results in formatted output to terminal
                print("\n========== MODEL PERFORMANCE ==========")
                print("Training Performance:")
                print(f"Accuracy : {train_accuracy:.4f}")
                print(f"Precision: {train_precision:.4f}")
                print(f"Recall   : {train_recall:.4f}")
                print(f"F1 Score : {train_f1:.4f}")

                print("\nTesting Performance:")
                print(f"Accuracy : {test_accuracy:.4f}")
                print(f"Precision: {test_precision:.4f}")
                print(f"Recall   : {test_recall:.4f}")
                print(f"F1 Score : {test_f1:.4f}")
                print("=======================================")

                # Also log the values
                logger.info("Training metrics: Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f",
                            train_accuracy, train_precision, train_recall, train_f1)
                logger.info("Testing metrics: Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f",
                            test_accuracy, test_precision, test_recall, test_f1)
            except Exception as e:
                logger.error(f"Error computing train/test metrics: {e}")

        return {
            'models': self.models,
            'best_model': self.best_model,
            'evaluation_results': results,
            'feature_importance': feature_importance if 'feature_importance' in locals() else None
        }


def main():
    """Main function for standalone execution"""
    # Configuration
    data_path = "../data/processed"
    model_save_path = "../models"
    data_filename = "pcos_processed.csv"
    target_column = "PCOS"  # Update with your actual target column name
    
    # Initialize trainer
    trainer = PCOSModelTrainer(data_path, model_save_path)
    
    # Run full training pipeline
    try:
        results = trainer.full_training_pipeline(data_filename, target_column)
        print("Training completed successfully!")
        print(f"Best model saved with CV score: {trainer.best_score:.4f}")
    except Exception as e:
        print(f"Error in training: {e}")


if __name__ == "__main__":
    main()
