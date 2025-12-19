"""
Model training and evaluation module for fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, auc, f1_score,
    precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb
import shap
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import joblib

logger = logging.getLogger(__name__)


class FraudModelTrainer:
    """
    Handles model training, evaluation, and comparison for fraud detection
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize trainer
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.models = {}
        self.results = {}
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2) -> Tuple:
        """
        Split data with stratification
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion for test set
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data: test_size={test_size}")
        
        return train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.seed
        )
    
    def train_baseline_model(self, X_train: pd.DataFrame, 
                             y_train: pd.Series) -> LogisticRegression:
        """
        Train logistic regression baseline model
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained LogisticRegression model
        """
        logger.info("Training baseline logistic regression model...")
        
        # Use balanced class weighting to handle imbalance
        model = LogisticRegression(
            random_state=self.seed,
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear'  # Good for small datasets
        )
        
        model.fit(X_train, y_train)
        logger.info("Baseline model trained successfully")
        
        return model
    
    def train_random_forest(self, X_train: pd.DataFrame, 
                            y_train: pd.Series) -> RandomForestClassifier:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained RandomForest model
        """
        logger.info("Training Random Forest model...")
        
        # Use balanced class weighting
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=self.seed,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        logger.info("Random Forest model trained successfully")
        
        return model
    
    def train_xgboost(self, X_train: pd.DataFrame, 
                      y_train: pd.Series) -> xgb.XGBClassifier:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model...")
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            n_jobs=-1,
            eval_metric='aucpr'
        )
        
        model.fit(X_train, y_train)
        logger.info("XGBoost model trained successfully")
        
        return model
    
    def train_lightgbm(self, X_train: pd.DataFrame, 
                       y_train: pd.Series) -> lgb.LGBMClassifier:
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained LightGBM model
        """
        logger.info("Training LightGBM model...")
        
        # Calculate scale_pos_weight
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        logger.info("LightGBM model trained successfully")
        
        return model
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, 
                       y_test: pd.Series, model_name: str = "Model") -> Dict[str, float]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'pr_auc': pr_auc
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp
        })
        
        # Log results
        logger.info(f"\n{'='*50}")
        logger.info(f"{model_name} Results:")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"TP: {tp}, FP: {fp}")
        logger.info(f"FN: {fn}, TN: {tn}")
        logger.info(f"{'='*50}\n")
        
        return metrics
    
    def cross_validate_model(self, model: Any, X: pd.DataFrame, 
                             y: pd.Series, cv_folds: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform stratified cross-validation
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target variable
            cv_folds: Number of folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)
        
        scoring = ['f1', 'precision', 'recall']
        cv_results = {}
        
        for score in scoring:
            cv_scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring=score,
                n_jobs=-1
            )
            cv_results[score] = cv_scores
        
        # Log results
        logger.info(f"\nCross-validation results:")
        for score, scores in cv_results.items():
            logger.info(f"{score.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def train_and_evaluate_all(self, X_fraud: pd.DataFrame, y_fraud: pd.Series,
                              X_credit: pd.DataFrame, y_credit: pd.Series) -> Dict:
        """
        Train and evaluate all models for both datasets
        
        Args:
            X_fraud: Fraud features
            y_fraud: Fraud targets
            X_credit: Credit card features
            y_credit: Credit card targets
            
        Returns:
            Complete results dictionary
        """
        all_results = {}
        
        # Process each dataset
        datasets = {
            'fraud': (X_fraud, y_fraud),
            'creditcard': (X_credit, y_credit)
        }
        
        for dataset_name, (X, y) in datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {dataset_name.upper()} dataset")
            logger.info(f"{'='*60}")
            
            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            dataset_results = {}
            
            # Train baseline
            baseline_model = self.train_baseline_model(X_train, y_train)
            dataset_results['baseline'] = self.evaluate_model(
                baseline_model, X_test, y_test, "Baseline Logistic Regression"
            )
            self.models[f'{dataset_name}_baseline'] = baseline_model
            
            # Save baseline
            joblib.dump(
                baseline_model, 
                f'models/{dataset_name}_baseline_model.joblib'
            )
            
            # Train ensemble models
            models_config = {
                'random_forest': self.train_random_forest,
                'xgboost': self.train_xgboost,
                'lightgbm': self.train_lightgbm
            }
            
            for model_name, train_func in models_config.items():
                try:
                    model = train_func(X_train, y_train)
                    metrics = self.evaluate_model(model, X_test, y_test, 
                                                 f"{model_name.replace('_', ' ').title()}")
                    dataset_results[model_name] = metrics
                    self.models[f'{dataset_name}_{model_name}'] = model
                    
                    # Save model
                    joblib.dump(
                        model,
                        f'models/{dataset_name}_{model_name}_model.joblib'
                    )
                    
                    # Cross-validation
                    cv_results = self.cross_validate_model(model, X_train, y_train)
                    dataset_results[model_name]['cv_results'] = cv_results
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
            
            all_results[dataset_name] = dataset_results
        
        self.results = all_results
        return all_results
    
    def select_best_model(self, dataset_name: str, 
                         metric: str = 'f1_score') -> Tuple[str, Any]:
        """
        Select best model based on evaluation metric
        
        Args:
            dataset_name: Name of dataset ('fraud' or 'creditcard')
            metric: Metric to use for selection
            
        Returns:
            Tuple of (model_name, model_object)
        """
        if dataset_name not in self.results:
            raise ValueError(f"No results found for {dataset_name}")
        
        # Find model with best metric
        best_score = -np.inf
        best_model_name = None
        
        for model_name, metrics in self.results[dataset_name].items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = model_name
        
        logger.info(f"Best model for {dataset_name}: {best_model_name} ({metric}={best_score:.4f})")
        
        best_model_key = f"{dataset_name}_{best_model_name}"
        return best_model_name, self.models[best_model_key]


# Main function
def run_train_pipeline(data_dir: str = 'data/raw', 
                       models_dir: str = 'models') -> None:
    """
    Run complete training pipeline
    
    Args:
        data_dir: Directory containing raw data
        models_dir: Directory to save models
    """
    import os
    os.makedirs(models_dir, exist_ok=True)
    
    # Import preprocessor
    from src.data_preprocessor import prepare_all_data
    
    # Prepare data
    data_dict = prepare_all_data(data_dir, handle_imbalance=True)
    
    # Initialize trainer
    trainer = FraudModelTrainer()
    
    # Train and evaluate all models
    results = trainer.train_and_evaluate_all(
        data_dict['fraud'][0],
        data_dict['fraud'][1],
        data_dict['creditcard'][0],
        data_dict['creditcard'][1]
    )
    
    # Save results
    joblib.dump(results, 'models/training_results.joblib')
    joblib.dump(trainer, 'models/trainer_object.joblib')
    
    # Best models
    best_fraud_model = trainer.select_best_model('fraud')
    best_credit_model = trainer.select_best_model('creditcard')
    
    print("\n=== TRAINING COMPLETE ===")
    print(f"Best fraud model: {best_fraud_model[0]}")
    print(f"Best credit card model: {best_credit_model[0]}")