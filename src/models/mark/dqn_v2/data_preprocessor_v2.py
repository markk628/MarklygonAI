import pandas as pd
from typing import List, Optional, Tuple, Dict, Union
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from src.config.config import STOCK_FEATURES_V2

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
        
    def _get_default_feature_groups(self) -> Dict[str, List[str]]:
        """Define default feature groups for different preprocessing strategies using STOCK_FEATURES_V2"""
        
        # Features that should NOT be winsorized/scaled (already properly bounded)
        excluded_features = [
            # Temporal features (cyclical encoding, already in [-1, 1])
            'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos', 
            'day_sin', 'day_cos', 'month_sin', 'month_cos', 
            'quarter_sin', 'quarter_cos',
            
            # Binary indicators (already in {0, 1})
            'vol_regime_5m', 'vol_regime_15m',
            'high_volume_regime_5m', 'high_volume_regime_15m',
            
            # Proportions/percentages (already in [0, 1])
            'momentum_persistence_bullish_1m', 'momentum_persistence_bearish_1m',
            'momentum_persistence_bullish_5m', 'momentum_persistence_bearish_5m', 
            'momentum_persistence_bullish_15m', 'momentum_persistence_bearish_15m',
            'morning_volume_ratio', 'lunch_volume_ratio', 'afternoon_volume_ratio',
            'bb_percent_b',  # Bollinger Band position [0, 1]
            
            # Correlation coefficients (already in [-1, 1])
            'volume_price_corr_5m', 'volume_price_corr_15m',
            'volume_persistence_5m', 'volume_persistence_15m',
            
            # Technical indicators with natural bounds (consider normalizing but not winsorizing)
            'rsi_7m', 'rsi_14m',  # [0, 100] range
            'stoch_k', 'stoch_d',  # [0, 100] range  
            'williams_r',  # [-100, 0] range
            'mfi',  # [0, 100] range
            'ultosc'  # [0, 100] range
        ]
        
        # All features that should be scaled (everything except excluded)
        scalable_features = [f for f in STOCK_FEATURES_V2 if f not in excluded_features]
        
        # Separate temporal features for reference
        temporal_features = [f for f in excluded_features if f.endswith(('_sin', '_cos'))]
        
        # Other bounded features that don't need preprocessing
        bounded_features = [f for f in excluded_features if not f.endswith(('_sin', '_cos'))]
        
        return {
            'scalable': scalable_features,      # Features that get winsorized + scaled
            'temporal': temporal_features,      # Cyclical time features [-1, 1]
            'bounded': bounded_features        # Other bounded features [0,1] or [-1,1]
        }
    
    def _handle_outliers(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle outliers using specified method"""
        data = data.copy()
        
        if self.outlier_method == 'none':
            return data
        
        # Get features to exclude from outlier handling
        temporal_features = self.feature_groups.get('temporal', [])
        bounded_features = self.feature_groups.get('bounded', [])
        excluded_features = temporal_features + bounded_features
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Skip temporal and bounded features
            if col in excluded_features:
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
            
            # Skip temporal and bounded features for outlier handling
            if group_name not in ['temporal', 'bounded']:
                # Handle outliers first (fit only)
                if self.outlier_method != 'none':
                    _ = self._handle_outliers(data, existing_features)
            
            # Create and fit scaler (only for scalable features)
            if self.scaling_method != 'none' and group_name == 'scalable':
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
        
        # Get features to exclude from outlier handling
        temporal_features = self.feature_groups.get('temporal', [])
        bounded_features = self.feature_groups.get('bounded', [])
        excluded_features = temporal_features + bounded_features
        
        for col in columns:
            # Skip temporal and bounded features
            if col in excluded_features:
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
            
            # Apply outlier handling (using fitted bounds) - skip temporal and bounded features
            if group_name not in ['temporal', 'bounded'] and self.outlier_method != 'none':
                data[existing_features] = self._apply_outlier_bounds(
                    data[existing_features], existing_features
                )
            
            # Apply scaling (only to scalable features)
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
            
            # Apply inverse scaling (only to scalable features that were scaled)
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