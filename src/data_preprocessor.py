"""
Comprehensive data preprocessing module for fraud detection
Handles both e-commerce and credit card datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging
from typing import Tuple, Dict, Any
import ipaddress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FraudDataPreprocessor:
    """
    Handles all preprocessing tasks for both fraud datasets
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize preprocessor with random seed for reproducibility
        """
        self.seed = seed
        self.scaler = StandardScaler()
        self.ip_country_map = None
        
    def load_data(self, fraud_path: str, ip_map_path: str, 
                  creditcard_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all three datasets
        
        Args:
            fraud_path: Path to Fraud_Data.csv
            ip_map_path: Path to IpAddress_to_Country.csv
            creditcard_path: Path to creditcard.csv
        
        Returns:
            Tuple of (fraud_df, ip_map_df, creditcard_df)
        """
        logger.info("Loading datasets...")
        
        # Load fraud data with datetime parsing
        fraud_df = pd.read_csv(fraud_path, parse_dates=['purchase_time', 'signup_time'])
        
        # Load IP mapping data
        ip_map_df = pd.read_csv(ip_map_path)
        
        # Load credit card data
        creditcard_df = pd.read_csv(creditcard_path)
        
        logger.info(f"Loaded fraud data: {fraud_df.shape}")
        logger.info(f"Loaded IP mapping: {ip_map_df.shape}")
        logger.info(f"Loaded credit card data: {creditcard_df.shape}")
        
        return fraud_df, ip_map_df, creditcard_df
    
    def clean_fraud_data(self, fraud_df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive cleaning of e-commerce fraud data
        
        Args:
            fraud_df: Raw fraud dataframe
            
        Returns:
            Cleaned fraud dataframe
        """
        logger.info("Cleaning fraud data...")
        
        # Make a copy to avoid modifying original
        df = fraud_df.copy()
        
        # Check initial shape
        initial_shape = df.shape
        
        # 1. Check for duplicates
        duplicates = df.duplicated().sum()
        logger.info(f"Found {duplicates} duplicate rows")
        if duplicates > 0:
            df = df.drop_duplicates()
        
        # 2. Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
            # For this dataset, drop rows with missing values (they're minimal)
            df = df.dropna()
            logger.info("Dropped rows with missing values")
        
        # 3. Correct data types
        df['user_id'] = df['user_id'].astype('int64')
        df['device_id'] = df['device_id'].astype('str')
        df['source'] = df['source'].astype('category')
        df['browser'] = df['browser'].astype('category')
        df['sex'] = df['sex'].astype('category')
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        
        # 4. Remove outliers in purchase_value
        Q1 = df['purchase_value'].quantile(0.01)
        Q3 = df['purchase_value'].quantile(0.99)
        before_outliers = df.shape[0]
        df = df[(df['purchase_value'] >= Q1) & (df['purchase_value'] <= Q3)]
        logger.info(f"Removed {before_outliers - df.shape[0]} outliers from purchase_value")
        
        logger.info(f"Cleaned fraud data: {initial_shape} -> {df.shape}")
        return df
    
    def clean_creditcard_data(self, creditcard_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean credit card dataset
        
        Args:
            creditcard_df: Raw credit card dataframe
            
        Returns:
            Cleaned credit card dataframe
        """
        logger.info("Cleaning credit card data...")
        
        df = creditcard_df.copy()
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        logger.info(f"Found {duplicates} duplicate rows")
        if duplicates > 0:
            df = df.drop_duplicates()
        
        # No missing values expected in this dataset
        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"Found {missing} missing values")
        
        # Remove extreme outliers in Amount
        Q1 = df['Amount'].quantile(0.01)
        Q3 = df['Amount'].quantile(0.99)
        before_outliers = df.shape[0]
        df = df[(df['Amount'] >= Q1) & (df['Amount'] <= Q3)]
        logger.info(f"Removed {before_outliers - df.shape[0]} outliers from Amount")
        
        return df
    
    def convert_ip_to_int(self, ip_address: str) -> int:
        """
        Convert IP address string to integer representation
        
        Args:
            ip_address: IP address as string
            
        Returns:
            Integer representation of IP
        """
        try:
            return int(ipaddress.ip_address(ip_address))
        except:
            return np.nan
    
    def merge_ip_country(self, fraud_df: pd.DataFrame, 
                         ip_map_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge fraud data with country mapping using IP ranges
        
        Args:
            fraud_df: Cleaned fraud dataframe
            ip_map_df: IP to country mapping dataframe
            
        Returns:
            Fraud dataframe with country information
        """
        logger.info("Converting IP addresses to countries...")
        
        # Convert IP addresses to integer format
        fraud_df['ip_int'] = fraud_df['ip_address'].apply(self.convert_ip_to_int)
        
        # Sort IP map for efficient merging
        ip_map_df = ip_map_df.sort_values('lower_bound_ip_address').reset_index(drop=True)
        
        # Use pandas merge_asof to find country for each IP
        # First, filter out rows with invalid IPs
        valid_ips = fraud_df.dropna(subset=['ip_int'])
        
        # Merge using asof to find the range
        merged = pd.merge_asof(
            valid_ips.sort_values('ip_int'),
            ip_map_df,
            left_on='ip_int',
            right_on='lower_bound_ip_address',
            direction='backward'
        )
        
        # Filter to keep only IPs within the ranges
        merged = merged[
            (merged['ip_int'] >= merged['lower_bound_ip_address']) & 
            (merged['ip_int'] <= merged['upper_bound_ip_address'])
        ]
        
        logger.info(f"Successfully mapped {len(merged)} IPs to countries")
        
        return merged
    
    def engineer_time_features(self, fraud_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features for fraud data
        
        Args:
            fraud_df: Fraud dataframe with datetime columns
            
        Returns:
            Dataframe with new time features
        """
        logger.info("Engineering time-based features...")
        
        df = fraud_df.copy()
        
        # Extract hour of day (0-23)
        df['hour_of_day'] = df['purchase_time'].dt.hour
        
        # Extract day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        
        # Create weekend flag
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Calculate time since signup in hours
        df['time_since_signup'] = (
            df['purchase_time'] - df['signup_time']
        ).dt.total_seconds() / 3600
        
        # Create signup_purchase_same_day feature
        df['signup_purchase_same_day'] = (
            df['purchase_time'].dt.date == df['signup_time'].dt.date
        ).astype(int)
        
        # Create time bins
        df['time_of_day_bin'] = pd.cut(
            df['hour_of_day'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        return df
    
    def engineer_transaction_features(self, fraud_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer transaction pattern features
        
        Args:
            fraud_df: Fraud dataframe
            
        Returns:
            Dataframe with transaction features
        """
        logger.info("Engineering transaction features...")
        
        df = fraud_df.copy()
        
        # Sort by user and time for proper calculations
        df = df.sort_values(['user_id', 'purchase_time'])
        
        # Transaction frequency per user
        user_stats = df.groupby('user_id').agg({
            'user_id': 'count',
            'purchase_value': ['mean', 'std', 'sum']
        }).round(2)
        
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns]
        user_stats = user_stats.rename(columns={
            'user_id_count': 'transactions_per_user',
            'purchase_value_mean': 'avg_purchase_value',
            'purchase_value_std': 'std_purchase_value',
            'purchase_value_sum': 'total_purchase_value'
        })
        
        # Merge user stats back to main dataframe
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Transaction velocity: transactions per hour
        df['purchase_time_hour'] = df['purchase_time'].dt.floor('H')
        hourly_txns = df.groupby(['user_id', 'purchase_time_hour']).size().reset_index(name='txns_per_hour')
        df = df.merge(hourly_txns, on=['user_id', 'purchase_time_hour'], how='left')
        
        # Device and browser risk scores
        device_risk = df.groupby('device_id')['class'].mean().to_dict()
        browser_risk = df.groupby('browser')['class'].mean().to_dict()
        country_risk = df.groupby('country')['class'].mean().to_dict()
        
        df['device_risk_score'] = df['device_id'].map(device_risk).fillna(0)
        df['browser_risk_score'] = df['browser'].map(browser_risk).fillna(0)
        df['country_risk_score'] = df['country'].map(country_risk).fillna(0)
        
        # Purchase value relative to user's average
        df['value_vs_user_avg'] = df['purchase_value'] / df['avg_purchase_value']
        
        # Flag high-value transactions
        df['is_high_value'] = (df['purchase_value'] > df['purchase_value'].quantile(0.9)).astype(int)
        
        return df
    
    def encode_categorical_features(self, fraud_df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding
        
        Args:
            fraud_df: Fraud dataframe with categorical columns
            
        Returns:
            Encoded dataframe
        """
        logger.info("Encoding categorical features...")
        
        df = fraud_df.copy()
        
        # Identify categorical columns
        cat_cols = ['source', 'browser', 'sex', 'country', 'time_of_day_bin']
        
        # One-hot encode with drop_first=True to avoid multicollinearity
        for col in cat_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        
        return df
    
    def scale_features(self, X: pd.DataFrame, 
                       exclude_cols: list = None) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            X: Feature matrix
            exclude_cols: Columns to exclude from scaling
            
        Returns:
            Scaled feature matrix
        """
        logger.info("Scaling features...")
        
        if exclude_cols is None:
            exclude_cols = []
        
        # Identify columns to scale
        cols_to_scale = [col for col in X.columns if col not in exclude_cols]
        
        # Fit and transform
        X_scaled = X.copy()
        if len(cols_to_scale) > 0:
            X_scaled[cols_to_scale] = self.scaler.fit_transform(X[cols_to_scale])
        
        return X_scaled
    
    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series, 
                         method: str = 'smote', sampling_strategy: float = 0.3) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using SMOTE or undersampling
        
        Args:
            X: Feature matrix
            y: Target variable
            method: 'smote' or 'undersample'
            sampling_strategy: Ratio for resampling
            
        Returns:
            Resampled X and y
        """
        logger.info(f"Handling imbalance with {method}...")
        
        original_dist = y.value_counts()
        logger.info(f"Original class distribution:\n{original_dist}")
        
        if method == 'smote':
            # Use SMOTE to generate synthetic samples
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=self.seed
            )
            X_res, y_res = smote.fit_resample(X, y)
        elif method == 'undersample':
            # Undersample majority class
            undersampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=self.seed
            )
            X_res, y_res = undersampler.fit_resample(X, y)
        else:
            raise ValueError("method must be 'smote' or 'undersample'")
        
        logger.info(f"Resampled class distribution:\n{y_res.value_counts()}")
        
        return X_res, y_res
    
    def prepare_fraud_data(self, fraud_df: pd.DataFrame, 
                           ip_map_df: pd.DataFrame,
                           handle_imbalance: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete pipeline to prepare fraud data for modeling
        
        Args:
            fraud_df: Raw fraud dataframe
            ip_map_df: IP mapping dataframe
            handle_imbalance: Whether to apply imbalance handling
            
        Returns:
            X_features, y_target
        """
        logger.info("Starting complete fraud data preparation...")
        
        # Step 1: Clean data
        df = self.clean_fraud_data(fraud_df)
        
        # Step 2: Merge with IP country mapping
        df = self.merge_ip_country(df, ip_map_df)
        
        # Step 3: Engineer features
        df = self.engineer_time_features(df)
        df = self.engineer_transaction_features(df)
        
        # Step 4: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 5: Separate features and target
        target_col = 'class'
        exclude_from_scaling = [col for col in df.columns if 'class' in col.lower()]
        
        X = df.drop(columns=[target_col, 'purchase_time', 'signup_time', 
                           'ip_address', 'device_id', 'purchase_time_hour'])
        y = df[target_col]
        
        # Step 6: Scale features
        X_scaled = self.scale_features(X)
        
        # Step 7: Handle imbalance if requested
        if handle_imbalance:
            X_final, y_final = self.handle_imbalance(X_scaled, y, method='smote')
        else:
            X_final, y_final = X_scaled, y
        
        logger.info(f"Final feature matrix shape: {X_final.shape}")
        logger.info(f"Feature columns: {list(X_final.columns)}")
        
        return X_final, y_final
    
    def prepare_creditcard_data(self, creditcard_df: pd.DataFrame,
                                handle_imbalance: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete pipeline to prepare credit card data for modeling
        
        Args:
            creditcard_df: Raw credit card dataframe
            handle_imbalance: Whether to apply imbalance handling
            
        Returns:
            X_features, y_target
        """
        logger.info("Starting complete credit card data preparation...")
        
        # Step 1: Clean data
        df = self.clean_creditcard_data(creditcard_df)
        
        # Step 2: Separate features and target
        X = df.drop(columns=['Class'])
        y = df['Class']
        
        # Step 3: Scale the Amount feature (V1-V28 are already scaled)
        scaler = StandardScaler()
        X['Amount'] = scaler.fit_transform(X[['Amount']])
        
        # Step 4: Handle imbalance
        if handle_imbalance:
            X_final, y_final = self.handle_imbalance(X, y, method='smote', 
                                                    sampling_strategy=0.1)
        else:
            X_final, y_final = X, y
        
        logger.info(f"Final feature matrix shape: {X_final.shape}")
        
        return X_final, y_final


# Convenience function
def prepare_all_data(data_dir: str = 'data/raw', 
                     handle_imbalance: bool = True) -> Dict[str, Any]:
    """
    Prepare all datasets in one go
    
    Args:
        data_dir: Directory containing raw data files
        handle_imbalance: Whether to apply imbalance handling
        
    Returns:
        Dictionary with prepared data
    """
    preprocessor = FraudDataPreprocessor()
    
    # Load data
    fraud_path = f"{data_dir}/Fraud_Data.csv"
    ip_map_path = f"{data_dir}/IpAddress_to_Country.csv"
    creditcard_path = f"{data_dir}/creditcard.csv"
    
    fraud_df, ip_map_df, creditcard_df = preprocessor.load_data(
        fraud_path, ip_map_path, creditcard_path
    )
    
    # Prepare fraud data
    X_fraud, y_fraud = preprocessor.prepare_fraud_data(
        fraud_df, ip_map_df, handle_imbalance
    )
    
    # Prepare credit card data
    X_credit, y_credit = preprocessor.prepare_creditcard_data(
        creditcard_df, handle_imbalance
    )
    
    return {
        'fraud': (X_fraud, y_fraud),
        'creditcard': (X_credit, y_credit)
    }