"""
Feature Engineering Module for PCOS Detection

This module provides helper functions for:
- Feature creation and transformation
- Feature selection
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import logging

logger = logging.getLogger(__name__)


class PCOSFeatureEngineer:
    """Feature engineering utilities for PCOS detection"""
    
    def __init__(self):
        self.feature_selector = None
        self.poly_features = None
        self.selected_features = None
    
    def create_bmi_categories(self, df: pd.DataFrame, weight_col: str, height_col: str) -> pd.DataFrame:
        """Create BMI and BMI categories"""
        df_new = df.copy()
        
        # Calculate BMI (assuming height in cm, weight in kg)
        df_new['BMI'] = df_new[weight_col] / ((df_new[height_col] / 100) ** 2)
        
        # Create BMI categories
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return 0  # Underweight
            elif 18.5 <= bmi < 25:
                return 1  # Normal
            elif 25 <= bmi < 30:
                return 2  # Overweight
            else:
                return 3  # Obese
        
        df_new['BMI_Category'] = df_new['BMI'].apply(categorize_bmi)
        logger.info("Created BMI and BMI categories")
        
        return df_new
    
    def create_hormone_ratios(self, df: pd.DataFrame, hormone_cols: list) -> pd.DataFrame:
        """Create hormone ratio features"""
        df_new = df.copy()
        
        # Example hormone ratios (adjust based on your dataset)
        if len(hormone_cols) >= 2:
            for i in range(len(hormone_cols)):
                for j in range(i + 1, len(hormone_cols)):
                    col1, col2 = hormone_cols[i], hormone_cols[j]
                    ratio_name = f"{col1}_{col2}_ratio"
                    
                    # Avoid division by zero
                    df_new[ratio_name] = df_new[col1] / (df_new[col2] + 1e-8)
                    logger.info(f"Created hormone ratio: {ratio_name}")
        
        return df_new
    
    def create_age_groups(self, df: pd.DataFrame, age_col: str) -> pd.DataFrame:
        """Create age group categories"""
        df_new = df.copy()
        
        def categorize_age(age):
            if age < 20:
                return 0  # Teens
            elif 20 <= age < 30:
                return 1  # Young adults
            elif 30 <= age < 40:
                return 2  # Adults
            else:
                return 3  # Mature adults
        
        df_new['Age_Group'] = df_new[age_col].apply(categorize_age)
        logger.info("Created age group categories")
        
        return df_new
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: list) -> pd.DataFrame:
        """Create interaction features between specified pairs"""
        df_new = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 in df.columns and col2 in df.columns:
                interaction_name = f"{col1}_x_{col2}"
                df_new[interaction_name] = df_new[col1] * df_new[col2]
                logger.info(f"Created interaction feature: {interaction_name}")
        
        return df_new
    
    def create_polynomial_features(self, df: pd.DataFrame, feature_cols: list, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features"""
        df_new = df.copy()
        
        if self.poly_features is None:
            self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        
        # Apply polynomial transformation to selected features
        poly_array = self.poly_features.fit_transform(df[feature_cols])
        poly_feature_names = self.poly_features.get_feature_names_out(feature_cols)
        
        # Add polynomial features to dataframe
        for i, name in enumerate(poly_feature_names):
            if name not in feature_cols:  # Skip original features
                df_new[f"poly_{name}"] = poly_array[:, i]
        
        logger.info(f"Created {len(poly_feature_names) - len(feature_cols)} polynomial features")
        return df_new
    
    def select_features_univariate(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """Select top k features using univariate statistical tests"""
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        self.selected_features = X.columns[selected_indices].tolist()
        
        logger.info(f"Selected {k} features using univariate selection: {self.selected_features}")
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series, n_features: int = 10) -> pd.DataFrame:
        """Select features using Recursive Feature Elimination"""
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        self.selected_features = X.columns[selected_indices].tolist()
        
        logger.info(f"Selected {n_features} features using RFE: {self.selected_features}")
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Get feature importance using Random Forest"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Calculated feature importance using Random Forest")
        return importance_df
    
    def create_statistical_features(self, df: pd.DataFrame, group_cols: list, agg_cols: list) -> pd.DataFrame:
        """Create statistical aggregation features"""
        df_new = df.copy()
        
        for group_col in group_cols:
            for agg_col in agg_cols:
                if group_col in df.columns and agg_col in df.columns:
                    # Mean by group
                    group_mean = df.groupby(group_col)[agg_col].transform('mean')
                    df_new[f"{agg_col}_mean_by_{group_col}"] = group_mean
                    
                    # Standard deviation by group
                    group_std = df.groupby(group_col)[agg_col].transform('std')
                    df_new[f"{agg_col}_std_by_{group_col}"] = group_std
                    
                    logger.info(f"Created statistical features for {agg_col} by {group_col}")
        
        return df_new
    
    def engineer_all_features(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Apply all feature engineering steps based on configuration"""
        df_engineered = df.copy()
        
        # BMI features
        if 'bmi' in config and config['bmi']['create']:
            df_engineered = self.create_bmi_categories(
                df_engineered, 
                config['bmi']['weight_col'], 
                config['bmi']['height_col']
            )
        
        # Age groups
        if 'age_groups' in config and config['age_groups']['create']:
            df_engineered = self.create_age_groups(
                df_engineered, 
                config['age_groups']['age_col']
            )
        
        # Hormone ratios
        if 'hormone_ratios' in config and config['hormone_ratios']['create']:
            df_engineered = self.create_hormone_ratios(
                df_engineered, 
                config['hormone_ratios']['hormone_cols']
            )
        
        # Interaction features
        if 'interactions' in config and config['interactions']['create']:
            df_engineered = self.create_interaction_features(
                df_engineered, 
                config['interactions']['feature_pairs']
            )
        
        # Polynomial features
        if 'polynomial' in config and config['polynomial']['create']:
            df_engineered = self.create_polynomial_features(
                df_engineered, 
                config['polynomial']['feature_cols'],
                config['polynomial']['degree']
            )
        
        logger.info("Completed all feature engineering steps")
        return df_engineered


def get_default_feature_config():
    """Get default feature engineering configuration"""
    return {
        'bmi': {
            'create': True,
            'weight_col': 'Weight',
            'height_col': 'Height'
        },
        'age_groups': {
            'create': True,
            'age_col': 'Age'
        },
        'hormone_ratios': {
            'create': True,
            'hormone_cols': ['LH', 'FSH', 'TSH', 'Testosterone']
        },
        'interactions': {
            'create': True,
            'feature_pairs': [('BMI', 'Age'), ('Weight', 'Height')]
        },
        'polynomial': {
            'create': False,  # Can be computationally expensive
            'feature_cols': ['BMI', 'Age'],
            'degree': 2
        }
    }


def main():
    """Example usage of feature engineering"""
    # This would typically be called from the training pipeline
    print("Feature engineering module loaded successfully!")
    print("Use PCOSFeatureEngineer class for feature engineering operations.")


if __name__ == "__main__":
    main()
