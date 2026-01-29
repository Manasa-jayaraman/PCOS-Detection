"""
Prediction Module for PCOS Detection

This module provides:
- CLI interface for making predictions
- Batch prediction capabilities
- Single sample prediction
- Model loading and inference utilities
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import json
from pathlib import Path
from typing import Union, List, Dict
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCOSPredictor:
    """PCOS prediction utility class"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self.scaler = None
        
    def load_model(self, model_filename: str):
        """Load trained model from pickle file"""
        try:
            filepath = self.model_path / model_filename
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle different model saving formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_names = model_data.get('feature_names')
                self.scaler = model_data.get('scaler')
            else:
                self.model = model_data
            
            logger.info(f"Model loaded successfully from: {filepath}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {filepath}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def preprocess_input(self, input_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Ensure all required features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Fill missing features with default values (0 or median)
                for feature in missing_features:
                    df[feature] = 0
            
            # Reorder columns to match training data
            df = df[self.feature_names]
        
        # Apply scaling if scaler is available
        if self.scaler:
            df_scaled = pd.DataFrame(
                self.scaler.transform(df),
                columns=df.columns,
                index=df.index
            )
            return df_scaled
        
        return df
    
    def predict_single(self, input_data: Dict) -> Dict:
        """Make prediction for a single sample"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess input
        processed_data = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(processed_data)[0]
        
        # Get prediction probability if available
        probability = None
        if hasattr(self.model, 'predict_proba'):
            prob_array = self.model.predict_proba(processed_data)[0]
            probability = {
                'no_pcos': float(prob_array[0]),
                'pcos': float(prob_array[1])
            }
        
        # Interpret result
        result = {
            'prediction': int(prediction),
            'prediction_label': 'PCOS' if prediction == 1 else 'No PCOS',
            'probability': probability,
            'confidence': float(max(prob_array)) if probability else None
        }
        
        logger.info(f"Prediction: {result['prediction_label']} "
                   f"(Confidence: {result['confidence']:.3f})" if result['confidence'] else "")
        
        return result
    
    def predict_batch(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """Make predictions for batch of samples from CSV file"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Load input data
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} samples for prediction")
            
            # Preprocess data
            processed_data = self.preprocess_input(df)
            
            # Make predictions
            predictions = self.model.predict(processed_data)
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(processed_data)
                df['pcos_probability'] = probabilities[:, 1]
                df['no_pcos_probability'] = probabilities[:, 0]
            
            # Add predictions to dataframe
            df['prediction'] = predictions
            df['prediction_label'] = df['prediction'].map({0: 'No PCOS', 1: 'PCOS'})
            
            # Save results if output file specified
            if output_file:
                df.to_csv(output_file, index=False)
                logger.info(f"Predictions saved to: {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise
    
    def get_feature_requirements(self) -> List[str]:
        """Get list of required features for prediction"""
        if self.feature_names:
            return self.feature_names
        else:
            logger.warning("Feature names not available. Model may have been saved without feature info.")
            return []
    
    def validate_input(self, input_data: Dict) -> Dict:
        """Validate input data and provide feedback"""
        validation_result = {
            'is_valid': True,
            'missing_features': [],
            'invalid_values': [],
            'warnings': []
        }
        
        required_features = self.get_feature_requirements()
        
        if required_features:
            # Check for missing features
            missing = set(required_features) - set(input_data.keys())
            if missing:
                validation_result['missing_features'] = list(missing)
                validation_result['warnings'].append(f"Missing features will be filled with default values: {missing}")
        
        # Check for invalid values (negative where not expected, etc.)
        for feature, value in input_data.items():
            if pd.isna(value):
                validation_result['invalid_values'].append(f"{feature}: NaN value")
            elif isinstance(value, (int, float)) and value < 0:
                # Most medical measurements shouldn't be negative
                if feature.lower() not in ['temperature_diff', 'change', 'delta']:
                    validation_result['warnings'].append(f"{feature}: Negative value ({value}) may be unusual")
        
        if validation_result['missing_features'] or validation_result['invalid_values']:
            validation_result['is_valid'] = False
        
        return validation_result


def create_sample_input():
    """Create a sample input for demonstration"""
    return {
        'Age': 25,
        'Weight': 65.0,
        'Height': 165.0,
        'BMI': 23.9,
        'Blood_Group': 1,  # Encoded
        'Pulse_rate': 72,
        'RR': 16,
        'Hb': 12.5,
        'Cycle': 1,  # Regular
        'Cycle_length': 28,
        'Marriage_Status': 0,  # Single
        'Pregnant': 0,  # No
        'No_of_abortions': 0,
        'FSH': 5.2,
        'LH': 4.8,
        'TSH': 2.1,
        'AMH': 3.5,
        'PRL': 15.2,
        'Vit_D3': 25.0,
        'PRG': 0.8,
        'RBS': 95.0,
        'Weight_gain': 0,
        'hair_growth': 0,
        'Skin_darkening': 0,
        'Hair_loss': 0,
        'Pimples': 0,
        'Fast_food': 1,
        'Reg_Exercise': 1,
        'BP_Systolic': 120,
        'BP_Diastolic': 80,
        'Follicle_No_L': 8,
        'Follicle_No_R': 7,
        'Avg_F_size_L': 12.5,
        'Avg_F_size_R': 11.8,
        'Endometrium': 8.2
    }


def main():
    """CLI interface for PCOS prediction"""
    parser = argparse.ArgumentParser(description='PCOS Detection Prediction Tool')
    parser.add_argument('--model', '-m', required=True, help='Path to trained model file')
    parser.add_argument('--input', '-i', help='Input CSV file for batch prediction')
    parser.add_argument('--output', '-o', help='Output CSV file for batch prediction results')
    parser.add_argument('--single', '-s', action='store_true', help='Interactive single prediction mode')
    parser.add_argument('--sample', action='store_true', help='Use sample input for demonstration')
    parser.add_argument('--features', '-f', action='store_true', help='Show required features')
    parser.add_argument('--eval', help='Path to labeled test CSV to evaluate model performance (must include target column)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PCOSPredictor("../models")
    
    # Load model
    if not predictor.load_model(args.model):
        print("Failed to load model. Exiting.")
        return
    
    # Show required features
    if args.features:
        features = predictor.get_feature_requirements()
        print("Required features:")
        for i, feature in enumerate(features, 1):
            print(f"{i:2d}. {feature}")
        return
    
    # Batch prediction
    if args.input:
        try:
            results = predictor.predict_batch(args.input, args.output)
            print(f"Batch prediction completed for {len(results)} samples")
            
            # Show summary
            pcos_count = (results['prediction'] == 1).sum()
            print(f"Predictions: {pcos_count} PCOS, {len(results) - pcos_count} No PCOS")
            
        except Exception as e:
            print(f"Error in batch prediction: {e}")
        return

    # Evaluation on labeled test set (runs once in CLI)
    if args.eval:
        try:
            df_test = pd.read_csv(args.eval)
            # Determine target column
            target_col = 'pcos_y_n' if 'pcos_y_n' in df_test.columns else None
            if target_col is None:
                print("Could not find target column 'pcos_y_n' in the provided CSV.")
                return
            y_test = df_test[target_col].astype(int)
            X_test = df_test.drop(columns=[target_col])

            # Preprocess features to match model
            X_proc = predictor.preprocess_input(X_test)
            # Predict
            y_pred = predictor.model.predict(X_proc)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print("\n===== Model Performance Metrics =====")
            print(f"Accuracy : {acc * 100:.2f}%")
            print(f"Precision: {prec * 100:.2f}%")
            print(f"Recall   : {rec * 100:.2f}%")
            print(f"F1 Score : {f1 * 100:.2f}%")
            print("=====================================\n")
        except Exception as e:
            print(f"Error during evaluation: {e}")
        return
    
    # Single prediction with sample data
    if args.sample:
        sample_data = create_sample_input()
        print("Using sample input data:")
        print(json.dumps(sample_data, indent=2))
        
        try:
            result = predictor.predict_single(sample_data)
            print(f"\nPrediction Result:")
            print(f"Prediction: {result['prediction_label']}")
            if result['probability']:
                print(f"Probabilities:")
                print(f"  No PCOS: {result['probability']['no_pcos']:.3f}")
                print(f"  PCOS: {result['probability']['pcos']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            
        except Exception as e:
            print(f"Error in prediction: {e}")
        return
    
    # Interactive single prediction
    if args.single:
        print("Interactive PCOS Prediction")
        print("Enter patient information (press Enter to skip optional fields):")
        
        # Get required features
        features = predictor.get_feature_requirements()
        input_data = {}
        
        # Basic patient info
        basic_features = ['Age', 'Weight', 'Height', 'BMI']
        for feature in basic_features:
            if feature in features:
                while True:
                    try:
                        value = input(f"{feature}: ")
                        if value.strip():
                            input_data[feature] = float(value)
                        break
                    except ValueError:
                        print("Please enter a valid number")
        
        # For demonstration, fill remaining features with defaults
        sample_data = create_sample_input()
        for feature in features:
            if feature not in input_data:
                input_data[feature] = sample_data.get(feature, 0)
        
        try:
            # Validate input
            validation = predictor.validate_input(input_data)
            if validation['warnings']:
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")
            
            # Make prediction
            result = predictor.predict_single(input_data)
            print(f"\nPrediction Result:")
            print(f"Prediction: {result['prediction_label']}")
            if result['probability']:
                print(f"Probabilities:")
                print(f"  No PCOS: {result['probability']['no_pcos']:.3f}")
                print(f"  PCOS: {result['probability']['pcos']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
            
        except Exception as e:
            print(f"Error in prediction: {e}")
        return
    
    # Show help if no action specified
    parser.print_help()


if __name__ == "__main__":
    main()
