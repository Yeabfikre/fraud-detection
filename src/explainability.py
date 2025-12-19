"""
SHAP explainability module for fraud detection models
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from typing import Tuple, List, Dict, Any
import os

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Handles SHAP analysis and model interpretability
    """
    
    def __init__(self, model_name: str, dataset_type: str):
        """
        Initialize explainer
        
        Args:
            model_name: Name of model to explain
            dataset_type: 'fraud' or 'creditcard'
        """
        self.model_name = model_name
        self.dataset_type = dataset_type
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.X_sample = None
        
    def load_model(self, models_dir: str = 'models') -> Any:
        """
        Load trained model
        
        Args:
            models_dir: Directory containing saved models
            
        Returns:
            Loaded model
        """
        model_path = f"{models_dir}/{self.dataset_type}_{self.model_name}_model.joblib"
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        return self.model
    
    def load_data_sample(self, X: pd.DataFrame, y: pd.Series, 
                         sample_size: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and sample data for SHAP analysis
        
        Args:
            X: Feature matrix
            y: Target series
            sample_size: Number of samples to use
            
        Returns:
            Sampled X and y
        """
        # Take stratified sample
        from sklearn.model_selection import train_test_split
        _, X_sample, _, y_sample = train_test_split(
            X, y,
            train_size=sample_size,
            stratify=y,
            random_state=42
        )
        
        self.X_sample = X_sample
        return X_sample, y_sample
    
    def create_explainer(self) -> shap.Explainer:
        """
        Create SHAP explainer based on model type
        
        Returns:
            SHAP explainer
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.model_name == 'xgboost':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_name == 'lightgbm':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_name == 'random_forest':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_name == 'baseline':
            # For logistic regression, use KernelExplainer
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                self.X_sample[:100]  # Background sample
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")
        
        logger.info(f"Created {self.explainer.__class__.__name__}")
        return self.explainer
    
    def calculate_shap_values(self) -> np.ndarray:
        """
        Calculate SHAP values for sample data
        
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            self.create_explainer()
        
        logger.info("Calculating SHAP values...")
        
        # For tree models, can use all samples
        if hasattr(self.explainer, 'shap_values'):
            if self.model_name == 'baseline':
                self.shap_values = self.explainer.shap_values(self.X_sample)
            else:
                self.shap_values = self.explainer.shap_values(self.X_sample)
        else:
            # For KernelExplainer
            self.shap_values = self.explainer.shap_values(self.X_sample)
        
        return self.shap_values
    
    def plot_summary(self, max_features: int = 20, 
                     save_path: str = None) -> None:
        """
        Plot SHAP summary plot (global feature importance)
        
        Args:
            max_features: Number of features to show
            save_path: Path to save plot
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        plt.figure(figsize=(12, 10))
        
        if self.model_name == 'baseline':
            # KernelExplainer returns list for each class
            shap.summary_plot(
                self.shap_values[1],  # Class 1 (fraud)
                self.X_sample,
                max_display=max_features,
                show=False
            )
        else:
            shap.summary_plot(
                self.shap_values, 
                self.X_sample,
                max_display=max_features,
                show=False
            )
        
        plt.title(f"SHAP Summary Plot - {self.model_name.replace('_', ' ').title()}\n"
                 f"Dataset: {self.dataset_type.replace('_', ' ').title()}",
                 fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved summary plot to {save_path}")
        
        plt.show()
    
    def plot_force(self, instance_idx: int, 
                   save_path: str = None) -> None:
        """
        Plot SHAP force plot for single prediction
        
        Args:
            instance_idx: Index of instance to explain
            save_path: Path to save plot
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        plt.figure(figsize=(12, 6))
        
        if self.model_name == 'baseline':
            shap.force_plot(
                self.explainer.expected_value[1],
                self.shap_values[1][instance_idx],
                self.X_sample.iloc[instance_idx],
                matplotlib=True,
                show=False
            )
        else:
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[instance_idx],
                self.X_sample.iloc[instance_idx],
                matplotlib=True,
                show=False
            )
        
        plt.title(f"SHAP Force Plot - Instance {instance_idx}",
                 fontsize=14, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved force plot to {save_path}")
        
        plt.show()
    
    def plot_dependence(self, feature: str, 
                        save_path: str = None) -> None:
        """
        Plot SHAP dependence plot for specific feature
        
        Args:
            feature: Feature name to analyze
            save_path: Path to save plot
        """
        if self.shap_values is None:
            self.calculate_shap_values()
        
        plt.figure(figsize=(10, 6))
        
        if self.model_name == 'baseline':
            shap.dependence_plot(
                feature,
                self.shap_values[1],
                self.X_sample,
                show=False
            )
        else:
            shap.dependence_plot(
                feature,
                self.shap_values,
                self.X_sample,
                show=False
            )
        
        plt.title(f"SHAP Dependence Plot - {feature}",
                 fontsize=14, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved dependence plot to {save_path}")
        
        plt.show()
    
    def analyze_prediction_types(self, X_test: pd.DataFrame, 
                                 y_test: pd.Series) -> Dict[str, List[int]]:
        """
        Find indices of different prediction types for analysis
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with indices for each prediction type
        """
        # Load model and make predictions
        model = self.load_model()
        y_pred = model.predict(X_test)
        
        # Find prediction types
        tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
        fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
        fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
        tn_indices = np.where((y_test == 0) & (y_pred == 0))[0]
        
        results = {
            'true_positives': tp_indices.tolist(),
            'false_positives': fp_indices.tolist(),
            'false_negatives': fn_indices.tolist(),
            'true_negatives': tn_indices.tolist()
        }
        
        logger.info(f"Prediction type distribution:")
        for key, indices in results.items():
            logger.info(f"{key}: {len(indices)}")
        
        return results
    
    def compare_feature_importance(self, X: pd.DataFrame, 
                                   top_n: int = 10) -> pd.DataFrame:
        """
        Compare SHAP importance with built-in feature importance
        
        Args:
            X: Feature matrix
            top_n: Number of top features to compare
            
        Returns:
            DataFrame with comparison
        """
        model = self.load_model()
        
        # Get built-in feature importance
        if hasattr(model, 'feature_importances_'):
            builtin_importance = model.feature_importances_
        else:
            # Logistic regression - use coefficients
            builtin_importance = np.abs(model.coef_[0])
        
        # Get SHAP importance
        if self.shap_values is None:
            self.calculate_shap_values()
        
        if self.model_name == 'baseline':
            shap_importance = np.abs(self.shap_values[1]).mean(axis=0)
        else:
            shap_importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create comparison dataframe
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'builtin_importance': builtin_importance,
            'shap_importance': shap_importance
        })
        
        # Normalize
        importance_df['builtin_rank'] = importance_df['builtin_importance'].rank(ascending=False)
        importance_df['shap_rank'] = importance_df['shap_importance'].rank(ascending=False)
        
        # Sort by SHAP importance
        importance_df = importance_df.sort_values('shap_importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def generate_business_insights(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate actionable business insights from SHAP analysis
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of insights
        """
        logger.info("Generating business insights...")
        
        # Load model and calculate SHAP values if needed
        if self.shap_values is None:
            self.calculate_shap_values()
        
        # Get top features by SHAP importance
        if self.model_name == 'baseline':
            mean_shap = np.abs(self.shap_values[1]).mean(axis=0)
        else:
            mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        top_features_idx = np.argsort(mean_shap)[-10:][::-1]
        top_features = X.columns[top_features_idx]
        
        insights = {
            'top_features': top_features.tolist(),
            'top_shap_values': mean_shap[top_features_idx].tolist(),
            'recommendations': []
        }
        
        # Generate specific recommendations based on features
        for feature in top_features:
            if 'time_since_signup' in feature:
                insights['recommendations'].append({
                    'feature': feature,
                    'insight': 'Transactions very soon after signup are high risk',
                    'action': 'Flag transactions within 24 hours of signup for extra verification'
                })
            elif 'hour_of_day' in feature:
                insights['recommendations'].append({
                    'feature': feature,
                    'insight': 'Transaction timing patterns indicate fraud risk',
                    'action': 'Implement time-based transaction limits or alerts'
                })
            elif 'purchase_value' in feature or 'value_vs_user_avg' in feature:
                insights['recommendations'].append({
                    'feature': feature,
                    'insight': 'Unusually high purchase values are strong fraud indicators',
                    'action': 'Set dynamic thresholds for high-value transactions based on user history'
                })
            elif 'country_risk_score' in feature:
                insights['recommendations'].append({
                    'feature': feature,
                    'insight': 'Certain countries show higher fraud rates',
                    'action': 'Implement geo-blocking or additional verification for high-risk countries'
                })
        
        logger.info(f"Generated {len(insights['recommendations'])} recommendations")
        
        return insights


# Main function
def run_explainability_pipeline(
    dataset_type: str,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = 'outputs/shap',
    sample_size: int = 1000
) -> None:
    """
    Complete SHAP explainability pipeline
    
    Args:
        dataset_type: 'fraud' or 'creditcard'
        model_name: Name of model to explain
        X: Full feature matrix
        y: Full target series
        X_test: Test features
        y_test: Test targets
        output_dir: Directory to save plots
        sample_size: Number of samples for SHAP
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize explainer
    explainer = SHAPExplainer(model_name, dataset_type)
    
    # Load model
    explainer.load_model()
    
    # Load data sample
    X_sample, y_sample = explainer.load_data_sample(X, y, sample_size)
    
    # Calculate SHAP values
    explainer.calculate_shap_values()
    
    # Find prediction types
    pred_types = explainer.analyze_prediction_types(X_test, y_test)
    
    # Plots
    explainer.plot_summary(
        save_path=f"{output_dir}/{dataset_type}_{model_name}_summary.png"
    )
    
    # Force plots for different prediction types
    for pred_type in ['true_positives', 'false_positives', 'false_negatives']:
        if len(pred_types[pred_type]) > 0:
            idx = pred_types[pred_type][0]
            explainer.plot_force(
                idx,
                save_path=f"{output_dir}/{dataset_type}_{model_name}_force_{pred_type}.png"
            )
    
    # Compare importance
    importance_df = explainer.compare_feature_importance(X_sample)
    print("\nTop 10 Feature Importance Comparison:")
    print(importance_df.to_string(index=False))
    
    # Business insights
    insights = explainer.generate_business_insights(X_sample)
    print("\n=== Business Insights ===")
    for rec in insights['recommendations']:
        print(f"\n🔍 Feature: {rec['feature']}")
        print(f"💡 Insight: {rec['insight']}")
        print(f"📋 Action: {rec['action']}")
    
    # Save insights
    joblib.dump(insights, f"{output_dir}/{dataset_type}_{model_name}_insights.joblib")
    
    return insights