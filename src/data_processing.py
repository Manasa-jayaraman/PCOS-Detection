"""
Data Processing Module for PCOS Detection

This module handles:
- Automatic dataset detection
- Column name standardization
- Data cleaning and preprocessing
- Target variable conversion
- Saving processed data for modeling
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCOSDataProcessor:
    """Main class for PCOS data processing pipeline"""
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.dataset_file = None
        
    def detect_dataset(self) -> str:
        """Automatically detect CSV dataset in raw data directory"""
        csv_files = list(self.raw_data_path.glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in data/raw/ directory")
        
        # Filter out .gitkeep and other non-data files
        data_files = [f for f in csv_files if not f.name.startswith('.')]
        
        if not data_files:
            raise FileNotFoundError("No data CSV files found in data/raw/ directory")
        
        if len(data_files) > 1:
            logger.warning(f"Multiple CSV files found: {[f.name for f in data_files]}. Using the first one.")
        
        self.dataset_file = data_files[0].name
        logger.info(f"Detected dataset: {self.dataset_file}")
        return self.dataset_file
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names by removing spaces, parentheses, special characters"""
        df_clean = df.copy()
        
        # Create mapping of old to new column names
        column_mapping = {}
        for col in df.columns:
            # Remove spaces, parentheses, special characters
            new_col = re.sub(r'[^a-zA-Z0-9]', '_', col)
            # Remove multiple underscores
            new_col = re.sub(r'_+', '_', new_col)
            # Remove leading/trailing underscores
            new_col = new_col.strip('_')
            # Convert to lowercase
            new_col = new_col.lower()
            column_mapping[col] = new_col
        
        df_clean.rename(columns=column_mapping, inplace=True)
        logger.info(f"Standardized {len(column_mapping)} column names")
        
        return df_clean, column_mapping
    
    def convert_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Y/N target variables to 1/0"""
        df_clean = df.copy()
        
        # Find potential target columns (PCOS related)
        target_candidates = [col for col in df_clean.columns if 'pcos' in col.lower()]
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                unique_vals = df_clean[col].unique()
                # Check if column has Y/N or Yes/No values
                if set(str(v).upper() for v in unique_vals if pd.notna(v)).issubset({'Y', 'N', 'YES', 'NO'}):
                    df_clean[col] = df_clean[col].map({
                        'Y': 1, 'N': 0, 'YES': 1, 'NO': 0,
                        'y': 1, 'n': 0, 'yes': 1, 'no': 0
                    })
                    logger.info(f"Converted {col} from Y/N to 1/0")
        
        return df_clean
    
    def load_raw_data(self, filename: str = None) -> pd.DataFrame:
        """Load raw dataset from CSV file"""
        try:
            if filename is None:
                filename = self.detect_dataset()
            
            filepath = self.raw_data_path / filename
            df = pd.read_csv(filepath)
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Standardize column names
            df, column_mapping = self.standardize_column_names(df)
            
            # Convert Y/N target variables
            df = self.convert_target_variable(df)
            
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataset"""
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove outliers
        df_clean = self._remove_outliers(df_clean)
        
        # Encode categorical variables
        df_clean = self._encode_categorical(df_clean)
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For numerical columns, fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median_val}")
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled {col} missing values with mode: {mode_val}")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, method='iqr') -> pd.DataFrame:
        """Remove outliers using IQR method"""
        df_clean = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
            if len(outliers) > 0:
                logger.info(f"Removed {len(outliers)} outliers from {col}")
        
        return df_clean
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
            logger.info(f"Encoded categorical column: {col}")
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Scale numerical features"""
        df_scaled = df.copy()
        feature_cols = [col for col in df.columns if col != target_col]
        
        df_scaled[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        logger.info("Applied feature scaling")
        
        return df_scaled
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to CSV"""
        try:
            self.processed_data_path.mkdir(parents=True, exist_ok=True)
            filepath = self.processed_data_path / filename
            df.to_csv(filepath, index=False)
            logger.info(f"Saved processed data to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def process_pipeline(self, input_filename: str = None, output_filename: str = "pcos_processed.csv", target_col: str = None):
        """Complete data processing pipeline"""
        logger.info("Starting data processing pipeline...")
        
        # Load raw data (auto-detect if no filename provided)
        df = self.load_raw_data(input_filename)
        
        # Auto-detect target column if not provided
        if target_col is None:
            target_candidates = [col for col in df.columns if 'pcos' in col.lower()]
            if target_candidates:
                target_col = target_candidates[0]
                logger.info(f"Auto-detected target column: {target_col}")
            else:
                raise ValueError("Could not auto-detect target column. Please specify target_col parameter.")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Don't scale features for now - will be done in training pipeline
        # df_processed = self.scale_features(df_clean, target_col)
        
        # Save processed data
        self.save_processed_data(df_clean, output_filename)
        
        logger.info("Data processing pipeline completed successfully!")
        return df_clean


def main():
    """Main function for standalone execution"""
    # Configuration
    raw_data_path = "../data/raw"
    processed_data_path = "../data/processed"
    output_filename = "pcos_processed.csv"
    
    # Initialize processor
    processor = PCOSDataProcessor(raw_data_path, processed_data_path)
    
    # Run processing pipeline (auto-detect dataset and target)
    try:
        processed_df = processor.process_pipeline(output_filename=output_filename)
        print(f"Processing completed. Final dataset shape: {processed_df.shape}")
        print(f"Columns: {list(processed_df.columns)}")
        print(f"Target column detected and processed successfully")
    except Exception as e:
        print(f"Error in processing: {e}")


if __name__ == "__main__":
    main()
