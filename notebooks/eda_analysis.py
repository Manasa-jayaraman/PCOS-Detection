"""
Exploratory Data Analysis Script for PCOS Detection

This script performs comprehensive EDA on the PCOS dataset including:
- Data overview and summary statistics
- Missing values analysis
- Target variable distribution
- Feature correlations
- Distribution plots
- Statistical tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_processing import PCOSDataProcessor

# Set plotting style
plt.style.use('default')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class PCOSEDAAnalyzer:
    """Comprehensive EDA analyzer for PCOS dataset"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.target_col = None
        self.numerical_cols = []
        self.categorical_cols = []
        
    def load_processed_data(self):
        """Load processed data for EDA"""
        try:
            # Try to load processed data first
            processed_file = self.data_path.parent / 'processed' / 'pcos_processed.csv'
            if processed_file.exists():
                self.df = pd.read_csv(processed_file)
                print(f"Loaded processed data: {self.df.shape}")
            else:
                # If no processed data, process raw data
                print("No processed data found. Processing raw data...")
                processor = PCOSDataProcessor(
                    str(self.data_path), 
                    str(self.data_path.parent / 'processed')
                )
                self.df = processor.process_pipeline()
                
            # Identify target column
            target_candidates = [col for col in self.df.columns if 'pcos' in col.lower()]
            if target_candidates:
                self.target_col = target_candidates[0]
                print(f"Target column: {self.target_col}")
            
            # Identify column types
            self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            
            # Remove target from features if it's numerical
            if self.target_col in self.numerical_cols:
                self.numerical_cols.remove(self.target_col)
                
            print(f"Numerical columns: {len(self.numerical_cols)}")
            print(f"Categorical columns: {len(self.categorical_cols)}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def data_overview(self):
        """Display basic data information"""
        print("="*60)
        print("DATA OVERVIEW")
        print("="*60)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nData types:")
        print(self.df.dtypes.value_counts())
        
        print("\nBasic statistics:")
        print(self.df.describe())
        
        if self.target_col:
            print(f"\nTarget variable ({self.target_col}) distribution:")
            print(self.df[self.target_col].value_counts())
            print(f"Target balance: {self.df[self.target_col].value_counts(normalize=True)}")
    
    def missing_values_analysis(self):
        """Analyze missing values"""
        print("\n" + "="*60)
        print("MISSING VALUES ANALYSIS")
        print("="*60)
        
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if len(missing_df) > 0:
            print("Columns with missing values:")
            print(missing_df)
            
            # Plot missing values heatmap
            plt.figure(figsize=(15, 8))
            sns.heatmap(self.df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
            plt.title('Missing Values Heatmap')
            plt.tight_layout()
            plt.savefig('missing_values_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("No missing values found in the dataset!")
    
    def target_distribution_analysis(self):
        """Analyze target variable distribution"""
        if not self.target_col:
            print("No target column identified")
            return
            
        print("\n" + "="*60)
        print("TARGET VARIABLE ANALYSIS")
        print("="*60)
        
        target_counts = self.df[self.target_col].value_counts()
        target_props = self.df[self.target_col].value_counts(normalize=True)
        
        print(f"Target distribution:")
        for val, count in target_counts.items():
            prop = target_props[val]
            label = "PCOS" if val == 1 else "No PCOS"
            print(f"  {label}: {count} ({prop:.1%})")
        
        # Plot target distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        target_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'])
        ax1.set_title('Target Variable Distribution (Counts)')
        ax1.set_xlabel('PCOS Status')
        ax1.set_ylabel('Count')
        ax1.set_xticklabels(['No PCOS', 'PCOS'], rotation=0)
        
        # Pie chart
        ax2.pie(target_counts.values, labels=['No PCOS', 'PCOS'], autopct='%1.1f%%', 
                colors=['skyblue', 'salmon'], startangle=90)
        ax2.set_title('Target Variable Distribution (Proportion)')
        
        plt.tight_layout()
        plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def correlation_analysis(self):
        """Analyze feature correlations"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Calculate correlation matrix for numerical features
        numerical_df = self.df[self.numerical_cols + [self.target_col] if self.target_col else self.numerical_cols]
        correlation_matrix = numerical_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Show top correlations with target
        if self.target_col:
            target_corr = correlation_matrix[self.target_col].abs().sort_values(ascending=False)
            print(f"\nTop 10 features correlated with {self.target_col}:")
            print(target_corr.head(11)[1:])  # Exclude self-correlation
    
    def feature_distributions(self):
        """Plot feature distributions"""
        print("\n" + "="*60)
        print("FEATURE DISTRIBUTIONS")
        print("="*60)
        
        # Plot numerical features
        if self.numerical_cols:
            n_cols = 4
            n_rows = (len(self.numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
            axes = axes.ravel() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for i, col in enumerate(self.numerical_cols):
                if i < len(axes):
                    self.df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide empty subplots
            for i in range(len(self.numerical_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def target_vs_features_analysis(self):
        """Analyze features by target variable"""
        if not self.target_col:
            return
            
        print("\n" + "="*60)
        print("TARGET vs FEATURES ANALYSIS")
        print("="*60)
        
        # Box plots for numerical features by target
        important_features = ['age_yrs', 'bmi', 'weight_kg', 'height_cm', 
                            'fsh_miu_ml', 'lh_miu_ml', 'tsh_miu_l', 'amh_ng_ml']
        
        # Filter features that exist in the dataset
        available_features = [f for f in important_features if f in self.numerical_cols]
        
        if available_features:
            n_cols = 3
            n_rows = (len(available_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
            axes = axes.ravel() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for i, feature in enumerate(available_features):
                if i < len(axes):
                    sns.boxplot(data=self.df, x=self.target_col, y=feature, ax=axes[i])
                    axes[i].set_title(f'{feature} by PCOS Status')
                    axes[i].set_xlabel('PCOS Status')
            
            # Hide empty subplots
            for i in range(len(available_features), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('target_vs_features.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def statistical_tests(self):
        """Perform statistical tests"""
        print("\n" + "="*60)
        print("STATISTICAL TESTS")
        print("="*60)
        
        if not self.target_col:
            return
        
        from scipy.stats import ttest_ind, chi2_contingency
        
        # T-tests for numerical features
        pcos_group = self.df[self.df[self.target_col] == 1]
        no_pcos_group = self.df[self.df[self.target_col] == 0]
        
        print("T-test results (p-values < 0.05 are significant):")
        print("-" * 50)
        
        significant_features = []
        for feature in self.numerical_cols[:10]:  # Test top 10 features
            try:
                stat, p_value = ttest_ind(pcos_group[feature].dropna(), 
                                        no_pcos_group[feature].dropna())
                print(f"{feature:25}: p-value = {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
                if p_value < 0.05:
                    significant_features.append(feature)
            except Exception as e:
                print(f"{feature:25}: Error - {e}")
        
        print(f"\nSignificant features (p < 0.05): {len(significant_features)}")
        print(f"Features: {significant_features}")
    
    def generate_summary_report(self):
        """Generate summary report"""
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)
        
        print(f"Dataset: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        print(f"Target variable: {self.target_col}")
        
        if self.target_col:
            target_dist = self.df[self.target_col].value_counts(normalize=True)
            print(f"Class distribution: {dict(target_dist)}")
            
            # Check for class imbalance
            minority_class_ratio = min(target_dist.values)
            if minority_class_ratio < 0.3:
                print("⚠️  Class imbalance detected - consider using SMOTE or other balancing techniques")
        
        missing_count = self.df.isnull().sum().sum()
        print(f"Missing values: {missing_count} total")
        
        print(f"Numerical features: {len(self.numerical_cols)}")
        print(f"Categorical features: {len(self.categorical_cols)}")
        
        print("\nRecommendations:")
        print("- Use SMOTE for handling class imbalance")
        print("- Apply feature scaling for numerical features")
        print("- Consider feature selection based on correlation analysis")
        print("- Use cross-validation for model evaluation")
    
    def run_complete_eda(self):
        """Run complete EDA analysis"""
        print("Starting Comprehensive EDA Analysis...")
        print("="*60)
        
        # Load data
        self.load_processed_data()
        
        # Run all analyses
        self.data_overview()
        self.missing_values_analysis()
        self.target_distribution_analysis()
        self.correlation_analysis()
        self.feature_distributions()
        self.target_vs_features_analysis()
        self.statistical_tests()
        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("EDA ANALYSIS COMPLETED!")
        print("="*60)
        print("Generated plots:")
        print("- missing_values_heatmap.png")
        print("- target_distribution.png")
        print("- correlation_matrix.png")
        print("- feature_distributions.png")
        print("- target_vs_features.png")


def main():
    """Main function to run EDA"""
    # Set up paths
    current_dir = Path(__file__).parent
    data_path = current_dir.parent / 'data' / 'raw'
    
    # Initialize analyzer
    analyzer = PCOSEDAAnalyzer(str(data_path))
    
    # Run complete EDA
    try:
        analyzer.run_complete_eda()
    except Exception as e:
        print(f"Error in EDA analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
