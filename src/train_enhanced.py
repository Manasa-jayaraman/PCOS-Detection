"""
Enhanced Training Module for PCOS Detection

This module handles:
- Automatic data loading and preprocessing
- Column transformation with encoding and scaling using ColumnTransformer
- Class imbalance handling using SMOTE
- Model training with RandomForest and LogisticRegression
- 80/20 train-test split
- Comprehensive evaluation metrics (accuracy, precision, recall, specificity, ROC-AUC)
- Model persistence with metadata
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPCOSTrainer:
    """Enhanced PCOS model trainer with SMOTE and ColumnTransformer"""
    
    def __init__(self, data_path: str, model_save_path: str):
        self.data_path = Path(data_path)
        self.model_save_path = Path(model_save_path)
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_col = None
        self.feature_cols = None
        self.preprocessor = None
        self.numerical_cols = []
        self.categorical_cols = []
    
    def load_processed_data(self, filename: str = "pcos_processed.csv", target_col: str = None):
        """Load processed data for training"""
        try:
            filepath = self.data_path / filename
            df = pd.read_csv(filepath)
            
            # Auto-detect target column if not provided
            if target_col is None:
                target_candidates = [col for col in df.columns if 'pcos' in col.lower()]
                if target_candidates:
                    target_col = target_candidates[0]
                    logger.info(f"Auto-detected target column: {target_col}")
                else:
                    raise ValueError("Could not auto-detect target column. Please specify target_col parameter.")
            
            self.target_col = target_col
            
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            self.feature_cols = X.columns.tolist()
            
            # Identify column types
            self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            
            logger.info(f"Loaded data with {X.shape[0]} samples and {X.shape[1]} features")
            logger.info(f"Numerical features: {len(self.numerical_cols)}")
            logger.info(f"Categorical features: {len(self.categorical_cols)}")
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            return X, y
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets (80/20)"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split - Train: {self.X_train.shape[0]} ({100*(1-test_size):.0f}%), Test: {self.X_test.shape[0]} ({100*test_size:.0f}%)")
        logger.info(f"Train target distribution: {self.y_train.value_counts().to_dict()}")
        logger.info(f"Test target distribution: {self.y_test.value_counts().to_dict()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_preprocessor(self):
        """Create preprocessing pipeline with ColumnTransformer"""
        transformers = []
        
        # Add numerical transformer
        if self.numerical_cols:
            transformers.append(('num', StandardScaler(), self.numerical_cols))
        
        # Add categorical transformer
        if self.categorical_cols:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False), self.categorical_cols))
        
        # Create ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        self.preprocessor = preprocessor
        logger.info("Created preprocessing pipeline with ColumnTransformer")
        return preprocessor
    
    def create_pipeline_with_smote(self, model):
        """Create pipeline with preprocessing and SMOTE"""
        pipeline = ImbPipeline([
            ('preprocessor', self.preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        return pipeline
    
    def get_model_configs(self):
        """Get model configurations for RandomForest and LogisticRegression"""
        return {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__max_features': ['sqrt', 'log2']
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'classifier__C': [0.1, 1, 10, 100],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear', 'saga']
                }
            }
        }
    
    def train_model_with_gridsearch(self, model_name: str, cv_folds: int = 5):
        """Train a single model with grid search, preprocessing, and SMOTE"""
        model_configs = self.get_model_configs()
        
        if model_name not in model_configs:
            raise ValueError(f"Model {model_name} not supported")
        
        config = model_configs[model_name]
        model = config['model']
        param_grid = config['params']
        
        logger.info(f"Starting grid search for {model_name} with SMOTE...")
        
        # Create pipeline with SMOTE
        pipeline = self.create_pipeline_with_smote(model)
        
        # Stratified K-Fold for better cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Store the best model
        self.models[model_name] = {
            'pipeline': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        logger.info(f"{model_name} - Best CV Score: {grid_search.best_score_:.4f}")
        logger.info(f"{model_name} - Best Params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_all_models(self, cv_folds: int = 5):
        """Train all models with grid search"""
        # Create preprocessor first
        self.create_preprocessor()
        
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
            self.best_model = self.models[best_model_name]['pipeline']
            self.best_score = best_score
            self.best_model_name = best_model_name
            logger.info(f"Best model: {best_model_name} with CV score: {best_score:.4f}")
        else:
            logger.warning("No best model found")
    
    def evaluate_model(self, model, model_name: str):
        """Evaluate a trained model on both training and test sets with comprehensive metrics"""
        # Training set predictions
        y_train_pred = model.predict(self.X_train)
        y_train_pred_proba = model.predict_proba(self.X_train)[:, 1]
        
        # Test set predictions
        y_test_pred = model.predict(self.X_test)
        y_test_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate training metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        train_precision = precision_score(self.y_train, y_train_pred)
        train_recall = recall_score(self.y_train, y_train_pred)
        train_f1 = f1_score(self.y_train, y_train_pred)
        train_auc = roc_auc_score(self.y_train, y_train_pred_proba)
        
        # Calculate test metrics
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_precision = precision_score(self.y_test, y_test_pred)
        test_recall = recall_score(self.y_test, y_test_pred)
        test_f1 = f1_score(self.y_test, y_test_pred)
        test_auc = roc_auc_score(self.y_test, y_test_pred_proba)
        
        # Calculate specificity for both sets
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(self.y_train, y_train_pred).ravel()
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(self.y_test, y_test_pred).ravel()
        
        train_specificity = tn_train / (tn_train + fp_train)
        test_specificity = tn_test / (tn_test + fp_test)
        
        # Print formatted output
        print("\n" + "="*42)
        print(f"{model_name.upper()} - MODEL PERFORMANCE")
        print("="*42)
        
        print("\nTRAINING PERFORMANCE:")
        print(f"{'Accuracy:':<12} {train_accuracy:.4f}")
        print(f"{'Precision:':<12} {train_precision:.4f}")
        print(f"{'Recall:':<12} {train_recall:.4f}")
        print(f"{'F1 Score:':<12} {train_f1:.4f}")
        print(f"{'Specificity:':<12} {train_specificity:.4f}")
        print(f"{'ROC-AUC:':<12} {train_auc:.4f}")
        
        print("\nTEST PERFORMANCE:")
        print(f"{'Accuracy:':<12} {test_accuracy:.4f}")
        print(f"{'Precision:':<12} {test_precision:.4f}")
        print(f"{'Recall:':<12} {test_recall:.4f}")
        print(f"{'F1 Score:':<12} {test_f1:.4f}")
        print(f"{'Specificity:':<12} {test_specificity:.4f}")
        print(f"{'ROC-AUC:':<12} {test_auc:.4f}")
        print("="*42 + "\n")
        
        # Also log detailed metrics for debugging
        logger.info(f"\n{model_name} Detailed Evaluation:")
        logger.info(f"\nTraining Set - Classification Report:\n{classification_report(self.y_train, y_train_pred)}")
        logger.info(f"\nTest Set - Classification Report:\n{classification_report(self.y_test, y_test_pred)}")
        
        return {
            'train_metrics': {
                'accuracy': train_accuracy,
                'precision': train_precision,
                'recall': train_recall,
                'f1_score': train_f1,
                'specificity': train_specificity,
                'roc_auc': train_auc,
                'predictions': y_train_pred,
                'probabilities': y_train_pred_proba,
                'classification_report': classification_report(self.y_train, y_train_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(self.y_train, y_train_pred).tolist()
            },
            'test_metrics': {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1_score': test_f1,
                'specificity': test_specificity,
                'roc_auc': test_auc,
                'predictions': y_test_pred,
                'probabilities': y_test_pred_proba,
                'classification_report': classification_report(self.y_test, y_test_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(self.y_test, y_test_pred).tolist()
            }
        }
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        results = {}
        
        for model_name, model_info in self.models.items():
            pipeline = model_info['pipeline']
            results[model_name] = self.evaluate_model(pipeline, model_name)
        
        return results
    
    def save_model(self, model, filename: str, include_metadata: bool = True):
        """Save trained model to pickle file with metadata"""
        try:
            self.model_save_path.mkdir(parents=True, exist_ok=True)
            filepath = self.model_save_path / filename
            
            if include_metadata:
                model_data = {
                    'model': model,
                    'feature_names': self.feature_cols,
                    'target_column': self.target_col,
                    'model_name': self.best_model_name,
                    'cv_score': self.best_score,
                    'numerical_cols': self.numerical_cols,
                    'categorical_cols': self.categorical_cols
                }
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
            else:
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
    
    def get_feature_importance(self):
        """Get feature importance from the best model if available"""
        if not self.best_model:
            return None
        
        classifier = self.best_model.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            # For RandomForest
            # Get feature names after preprocessing
            preprocessor = self.best_model.named_steps['preprocessor']
            feature_names = []
            
            # Add numerical feature names
            if self.numerical_cols:
                feature_names.extend(self.numerical_cols)
            
            # Add categorical feature names (after one-hot encoding)
            if self.categorical_cols:
                cat_transformer = preprocessor.named_transformers_['cat']
                if hasattr(cat_transformer, 'get_feature_names_out'):
                    cat_features = cat_transformer.get_feature_names_out(self.categorical_cols)
                    feature_names.extend(cat_features)
                else:
                    feature_names.extend(self.categorical_cols)
            
            importances = classifier.feature_importances_
            
            # Ensure we have the right number of features
            if len(feature_names) != len(importances):
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        
        elif hasattr(classifier, 'coef_'):
            # For LogisticRegression
            coefficients = np.abs(classifier.coef_[0])
            
            # Get feature names after preprocessing
            preprocessor = self.best_model.named_steps['preprocessor']
            feature_names = []
            
            if self.numerical_cols:
                feature_names.extend(self.numerical_cols)
            
            if self.categorical_cols:
                cat_transformer = preprocessor.named_transformers_['cat']
                if hasattr(cat_transformer, 'get_feature_names_out'):
                    cat_features = cat_transformer.get_feature_names_out(self.categorical_cols)
                    feature_names.extend(cat_features)
                else:
                    feature_names.extend(self.categorical_cols)
            
            if len(feature_names) != len(coefficients):
                feature_names = [f'feature_{i}' for i in range(len(coefficients))]
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': coefficients
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        
        return None
    
    def full_training_pipeline(self, data_filename: str = "pcos_processed.csv", target_col: str = None, 
                             model_filename: str = "pcos_model.pkl"):
        """Complete training pipeline with 80/20 split, SMOTE, and comprehensive evaluation"""
        logger.info("Starting enhanced training pipeline...")
        
        # Load data
        X, y = self.load_processed_data(data_filename, target_col)
        
        # Split data (80/20)
        self.split_data(X, y, test_size=0.2)
        
        # Train models (RandomForest and LogisticRegression)
        self.train_all_models()
        
        # Evaluate all models
        results = self.evaluate_all_models()
        
        # Save best model with metadata
        self.save_best_model(model_filename)
        
        # Get feature importance for best model
        feature_importance = self.get_feature_importance()
        if feature_importance is not None:
            logger.info("Top 10 most important features:")
            logger.info(feature_importance.head(10))
        
        logger.info("Enhanced training pipeline completed successfully!")
        
        return {
            'models': self.models,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'evaluation_results': results,
            'feature_importance': feature_importance
        }


def main():
    """Main function for standalone execution"""
    # Configuration
    data_path = "../data/processed"
    model_save_path = "../models"
    data_filename = "pcos_processed.csv"
    
    # Initialize enhanced trainer
    trainer = EnhancedPCOSTrainer(data_path, model_save_path)
    
    # Run full training pipeline (auto-detect target column)
    try:
        results = trainer.full_training_pipeline(data_filename)
        print("\n" + "="*60)
        print("ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best model: {trainer.best_model_name}")
        print(f"Best CV score: {trainer.best_score:.4f}")
        print(f"Model saved as: pcos_model.pkl")
        
        # Print evaluation summary
        if results['evaluation_results']:
            print("\nTest Set Performance Summary:")
            print("-" * 50)
            for model_name, metrics in results['evaluation_results'].items():
                print(f"\n{model_name}:")
                print(f"  Accuracy:     {metrics['accuracy']:.4f}")
                print(f"  Precision:    {metrics['precision']:.4f}")
                print(f"  Recall (Sens):{metrics['recall']:.4f}")
                print(f"  Specificity:  {metrics['specificity']:.4f}")
                print(f"  F1-Score:     {metrics['f1_score']:.4f}")
                print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
        
        print("\nFeatures used:")
        print(f"- Numerical: {len(trainer.numerical_cols)}")
        print(f"- Categorical: {len(trainer.categorical_cols)}")
        print(f"- Total: {len(trainer.feature_cols)}")
        
        print("\nPipeline components:")
        print("✓ ColumnTransformer for preprocessing")
        print("✓ SMOTE for class imbalance handling")
        print("✓ GridSearchCV for hyperparameter tuning")
        print("✓ Comprehensive evaluation metrics")
        
    except Exception as e:
        print(f"Error in training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
