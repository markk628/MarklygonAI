import pandas as pd
from typing import List, Optional, Tuple, Dict, Union
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from src.config.config import PRICE_FEATURES, VOLUME_FEATURES, TECHNICAL_FEATURES, VOLATILLITY_FEATURES, RETURNS_FEATURES, TEMPORAL_FEATURES

import warnings
warnings.filterwarnings('ignore')


class FinancialDataPreprocessor:
    """
    Comprehensive preprocessor for financial timeseries data with outlier handling and scaling
    """
    
    def __init__(self, 
                 scaling_method: str = 'robust',
                 outlier_method: str = 'winsorize',
                 outlier_threshold: float = 0.01,
                 feature_groups: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the preprocessor
        
        Args:
            scaling_method: 'robust', 'standard', 'minmax', or 'none'
            outlier_method: 'winsorize', 'clip', or 'none'
            outlier_threshold: Percentile threshold for outlier handling (e.g., 0.01 = 1%)
            feature_groups: Dictionary grouping features for different preprocessing
        """
        self.scaling_method = scaling_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.feature_groups = feature_groups or self._get_default_feature_groups()
        
        # Initialize scalers dictionary
        self.scalers = {}
        self.outlier_bounds = {}
        self.is_fitted = False
        
    # def _get_default_feature_groups(self) -> Dict[str, List[str]]:
    #     """Define default feature groups for different preprocessing strategies"""
    #     return {
    #         'price': ['open', 'high', 'low', 'close', 'vwap'],
    #         'volume': ['volume', 'transactions'],
    #         'technical': [
    #             'stochrsi_k_14_1min', 'stochrsi_d_14_1min', 'rsi_14_1min',
    #             'macd_12_26_9_1min', 'macd_signal_12_26_9_1min', 'macd_hist_12_26_9_1min',
    #             'roc_10_1min', 'obv_1min', 'ema_3_1min', 'ema_9_1min', 'ema_21_1min',
    #             'plusdi_20_1min', 'minusdi_20_1min', 'adx_20_1min',
    #             'bband_upper_20_1min', 'bband_lower_20_1min',
    #             'atr_14_1min', 'cci_20_1min', 'mfi_14_1min'
    #         ],
    #         'volatility': ['volume_rolling_std_15', 'log_return_rolling_std_15'],
    #         'returns': ['close_diff_1', 'log_return_1'],
    #         'temporal': [
    #             'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos',
    #             'day_sin', 'day_cos', 'month_sin', 'month_cos',
    #             'quarter_sin', 'quarter_cos'
    #         ]
    #     }
    
    def _get_default_feature_groups(self) -> Dict[str, List[str]]:
        """Define default feature groups for different preprocessing strategies"""
        return {
            'price': PRICE_FEATURES,
            'volume': VOLUME_FEATURES,
            'technical': TECHNICAL_FEATURES,
            'volatility': VOLATILLITY_FEATURES,
            'returns': RETURNS_FEATURES,
            'temporal': TEMPORAL_FEATURES
        }
    
    def _handle_outliers(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle outliers using specified method"""
        data = data.copy()
        
        if self.outlier_method == 'none':
            return data
        
        # Get temporal features to exclude
        temporal_features = self.feature_groups.get('temporal', [])
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Skip temporal features
            if col in temporal_features:
                continue
                
            if self.outlier_method == 'winsorize':
                # Winsorize: cap values at specified percentiles
                lower = data[col].quantile(self.outlier_threshold)
                upper = data[col].quantile(1 - self.outlier_threshold)
                
                # Store bounds for later use
                self.outlier_bounds[col] = {'lower': lower, 'upper': upper}
                
                # Apply winsorization
                data[col] = data[col].clip(lower=lower, upper=upper)
                
            elif self.outlier_method == 'clip':
                # Clip based on IQR
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 3 * IQR
                upper = Q3 + 3 * IQR
                
                # Store bounds for later use
                self.outlier_bounds[col] = {'lower': lower, 'upper': upper}
                
                # Apply clipping
                data[col] = data[col].clip(lower=lower, upper=upper)
        
        return data
    
    def _create_scaler(self, method: str):
        """Create appropriate scaler based on method"""
        if method == 'robust':
            # RobustScaler is best for financial data with outliers
            return RobustScaler(quantile_range=(5, 95))
        elif method == 'standard':
            return StandardScaler()
        elif method == 'minmax':
            return MinMaxScaler(feature_range=(-1, 1))
        else:
            return None
    
    def fit(self, data: pd.DataFrame) -> 'FinancialDataPreprocessor':
        """
        Fit the preprocessor on training data
        
        Args:
            data: Training dataframe
            
        Returns:
            Self for chaining
        """
        # Reset state
        self.scalers = {}
        self.outlier_bounds = {}
        
        # Handle outliers and fit scalers for each feature group
        for group_name, features in self.feature_groups.items():
            # Get columns that exist in the data
            existing_features = [f for f in features if f in data.columns]
            
            if not existing_features:
                continue
            
            # Skip temporal features for outlier handling
            if group_name != 'temporal':
                # Handle outliers first (fit only)
                if self.outlier_method != 'none':
                    _ = self._handle_outliers(data, existing_features)
            
            # Create and fit scaler
            if self.scaling_method != 'none' and group_name != 'temporal':
                scaler = self._create_scaler(self.scaling_method)
                if scaler is not None:
                    # Apply outlier handling before fitting scaler
                    clean_data = self._apply_outlier_bounds(data[existing_features].copy(), existing_features)
                    scaler.fit(clean_data)
                    self.scalers[group_name] = scaler
        
        self.is_fitted = True
        return self
    
    def _apply_outlier_bounds(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply previously fitted outlier bounds"""
        data = data.copy()
        
        # Get temporal features to exclude
        temporal_features = self.feature_groups.get('temporal', [])
        
        for col in columns:
            # Skip temporal features
            if col in temporal_features:
                continue
                
            if col in self.outlier_bounds:
                bounds = self.outlier_bounds[col]
                data[col] = data[col].clip(lower=bounds['lower'], upper=bounds['upper'])
        
        return data
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        data = data.copy()
        
        # Apply transformations for each feature group
        for group_name, features in self.feature_groups.items():
            # Get columns that exist in the data
            existing_features = [f for f in features if f in data.columns]
            
            if not existing_features:
                continue
            
            # Apply outlier handling (using fitted bounds)
            if group_name != 'temporal' and self.outlier_method != 'none':
                data[existing_features] = self._apply_outlier_bounds(
                    data[existing_features], existing_features
                )
            
            # Apply scaling
            if group_name in self.scalers:
                scaler = self.scalers[group_name]
                data[existing_features] = scaler.transform(data[existing_features])
        
        return data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled data back to original scale
        
        Args:
            data: Scaled data
            
        Returns:
            Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        
        data = data.copy()
        
        # Apply inverse transformations for each feature group
        for group_name, features in self.feature_groups.items():
            # Get columns that exist in the data
            existing_features = [f for f in features if f in data.columns]
            
            if not existing_features:
                continue
            
            # Apply inverse scaling
            if group_name in self.scalers:
                scaler = self.scalers[group_name]
                data[existing_features] = scaler.inverse_transform(data[existing_features])
        
        return data
    
    def save(self, filepath: Union[str, Path]):
        """
        Save the fitted preprocessor
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a clean state dictionary without problematic references
        state = {
            'scaling_method': str(self.scaling_method),
            'outlier_method': str(self.outlier_method),
            'outlier_threshold': float(self.outlier_threshold),
            'feature_groups': {k: list(v) for k, v in self.feature_groups.items()},
            'scalers': self.scalers,
            'outlier_bounds': self.outlier_bounds,
            'is_fitted': bool(self.is_fitted)
        }
        
        joblib.dump(state, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'FinancialDataPreprocessor':
        """
        Load a fitted preprocessor
        
        Args:
            filepath: Path to the saved preprocessor
            
        Returns:
            Loaded preprocessor
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        state = joblib.load(filepath)
        
        # Create new instance with explicit string conversion
        preprocessor = cls(
            scaling_method=str(state['scaling_method']),
            outlier_method=str(state['outlier_method']),
            outlier_threshold=float(state['outlier_threshold']),
            feature_groups=state['feature_groups']
        )
        
        # Restore fitted state
        preprocessor.scalers = state['scalers']
        preprocessor.outlier_bounds = state['outlier_bounds']
        preprocessor.is_fitted = state['is_fitted']
        
        return preprocessor
    
    def get_preprocessing_info(self) -> Dict:
        """Get information about the preprocessing configuration"""
        info = {
            'scaling_method': self.scaling_method,
            'outlier_method': self.outlier_method,
            'outlier_threshold': self.outlier_threshold,
            'is_fitted': self.is_fitted,
            'feature_groups': {k: len(v) for k, v in self.feature_groups.items()},
            'num_scalers': len(self.scalers),
            'num_outlier_bounds': len(self.outlier_bounds)
        }
        return info


def preprocess_financial_data(train_data: pd.DataFrame,
                              valid_data: Optional[pd.DataFrame] = None,
                              test_data: Optional[pd.DataFrame] = None,
                              scaling_method: str = 'robust',
                              outlier_method: str = 'winsorize',
                              save_preprocessor: bool = True,
                              preprocessor_path: str = 'preprocessor.pkl') -> Tuple:
    """
    Convenience function to preprocess financial data
    
    Args:
        train_data: Training dataframe
        valid_data: Validation dataframe (optional)
        test_data: Test dataframe (optional)
        scaling_method: Scaling method to use
        outlier_method: Outlier handling method
        save_preprocessor: Whether to save the fitted preprocessor
        preprocessor_path: Path to save the preprocessor
        
    Returns:
        Tuple of (preprocessor, train_processed, val_processed, test_processed)
    """
    # Create and fit preprocessor
    preprocessor = FinancialDataPreprocessor(
        scaling_method=scaling_method,
        outlier_method=outlier_method
    )
    
    # Fit on training data only
    train_processed = preprocessor.fit_transform(train_data)
    
    # Transform validation and test data if provided
    val_processed = None
    test_processed = None
    
    if valid_data is not None:
        val_processed = preprocessor.transform(valid_data)
    
    if test_data is not None:
        test_processed = preprocessor.transform(test_data)
    
    # Save preprocessor if requested
    if save_preprocessor:
        preprocessor.save(preprocessor_path)
    
    return preprocessor, train_processed, val_processed, test_processed