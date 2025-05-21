import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
import joblib
import torch
from torch import nn
import warnings
warnings.filterwarnings('ignore')

from src.config.config import (
    WINDOW_SIZE,
    DEVICE
)

class FeatureProcessor:
    """
    Modular feature selection/extraction for stock trading data
    Compatible with rolling window scaling and DQN models
    """
    def __init__(self, 
                 window_size=WINDOW_SIZE,
                 scaler_type='standard',  # 'standard', 'minmax', or None
                 selection_method='pca',  # 'pca', 'mutual_info', 'f_regression', 'autoencoder', 'combined', None
                 n_components=22,         # Number of features to select
                 correlation_threshold=0.85,  # Threshold for removing highly correlated features
                 device=DEVICE):
        
        self.window_size = window_size
        self.scaler_type = scaler_type
        self.selection_method = selection_method
        self.n_components = n_components
        self.correlation_threshold = correlation_threshold
        self.device = device
        
        # Initialize scalers and selectors
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
            
        # Initialize feature selectors/extractors
        self.selector = None
        self.autoencoder = None
        self.selected_features = None
        self.feature_importance = None
        
        # Lists to track feature groups
        self.price_features = ['open', 'high', 'low', 'close', 'vwap']
        self.volume_features = ['volume', 'transactions']
        self.momentum_features = [col for col in self.get_default_features() if 
                                 any(x in col for x in ['rsi', 'macd', 'roc', 'mfi'])]
        self.volatility_features = [col for col in self.get_default_features() if 
                                   any(x in col for x in ['atr', 'bband', 'std'])]
        self.trend_features = [col for col in self.get_default_features() if 
                              any(x in col for x in ['ema', 'sma', 'adx', 'di'])]
        self.oscillator_features = [col for col in self.get_default_features() if 
                                   any(x in col for x in ['stoch', 'cci', 'ultosc'])]
        self.time_features = [col for col in self.get_default_features() if 
                             any(x in col for x in ['minute', 'hour', 'day', 'month', 'quarter'])]
        self.lag_features = [col for col in self.get_default_features() if 'lag' in col]
        self.rolling_features = [col for col in self.get_default_features() if 'rolling' in col]
        
    def get_default_features(self):
        """Return the default feature list"""
        return [
            'open', 'high', 'low', 'close', 'transactions', 'volume', 'vwap', 
            'stochrsi_k_14_1min', 'stochrsi_d_14_1min', 'stochrsi_k_21_1min', 'stochrsi_d_21_1min', 
            'rsi_7_1min', 'rsi_14_1min', 'rsi_21_1min', 
            'macd_5_13_4_1min', 'macd_signal_5_13_4_1min', 'macd_hist_5_13_4_1min', 
            'macd_12_26_9_1min', 'macd_signal_12_26_9_1min', 'macd_hist_12_26_9_1min', 
            'roc_5_1min', 'roc_10_1min', 'roc_20_1min', 
            'ultosc_5_15_30_1min', 'ultosc_7_21_42_1min', 'ultosc_10_30_60_1min', 
            'plusdi_10_1min', 'plusdi_20_1min', 'plusdi_30_1min', 
            'minusdi_10_1min', 'minusdi_20_1min', 'minusdi_30_1min', 
            'adx_10_1min', 'adx_20_1min', 'adx_30_1min', 
            'cci_10_1min', 'cci_20_1min', 'cci_30_1min', 
            'ema_3_1min', 'ema_5_1min', 'ema_9_1min', 'ema_21_1min', 'ema_50_1min', 
            'sma_5_1min', 'sma_10_1min', 'sma_20_1min', 'sma_50_1min', 
            'obv_1min', 'mfi_7_1min', 'mfi_14_1min', 'mfi_21_1min', 
            'bband_upper_10_1min', 'bband_middle_10_1min', 'bband_lower_10_1min', 
            'bband_upper_20_1min', 'bband_middle_20_1min', 'bband_lower_20_1min', 
            'bband_upper_50_1min', 'bband_middle_50_1min', 'bband_lower_50_1min', 
            'atr_14_1min', 'atr_21_1min', 
            'minute', 'minute_sin', 'minute_cos', 
            'hour', 'hour_sin', 'hour_cos', 
            'day', 'day_sin', 'day_cos', 
            'month', 'month_sin', 'month_cos', 
            'quarter', 'quarter_sin', 'quarter_cos', 
            'time_since_last_significant_change', 
            'open_lag_1', 'high_lag_1', 'low_lag_1', 'close_lag_1', 'volume_lag_1', 'vwap_lag_1', 
            'open_lag_5', 'high_lag_5', 'low_lag_5', 'close_lag_5', 'volume_lag_5', 'vwap_lag_5', 
            'open_lag_10', 'high_lag_10', 'low_lag_10', 'close_lag_10', 'volume_lag_10', 'vwap_lag_10', 
            'close_rolling_mean_5', 'close_rolling_std_5', 
            'volume_rolling_mean_5', 'volume_rolling_std_5', 
            'close_rolling_mean_15', 'close_rolling_std_15', 
            'volume_rolling_mean_15', 'volume_rolling_std_15', 
            'close_rolling_mean_30', 'close_rolling_std_30', 
            'volume_rolling_mean_30', 'volume_rolling_std_30', 
            'close_diff_1', 'close_pct_change_1', 'log_return_1', 
            'high_low_range', 'close_open_range', 
            'log_return_rolling_std_15', 'log_return_rolling_std_30', 
            'high_low_ratio', 'close_open_ratio'
        ]
        
    def remove_highly_correlated(self, X):
        """Remove highly correlated features"""
        features_to_drop = ['minute', 'hour', 'day', 'month', 'quarter']
        if isinstance(X, pd.DataFrame):
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold) and column not in (self.price_features + self.time_features)] + features_to_drop
            return X.drop(columns=to_drop), to_drop
        else:
            df = pd.DataFrame(X)
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold) and column not in (self.price_features + self.time_features)] + features_to_drop
            return df.drop(columns=to_drop).values, to_drop
        
    def _build_autoencoder(self, input_dim):
        """Build a simple autoencoder for feature extraction"""
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim):
                super(Autoencoder, self).__init__()
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Linear(input_dim // 2, encoding_dim),
                    nn.ReLU()
                )
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Linear(input_dim // 2, input_dim),
                    nn.Tanh()  # Tanh to output values in range (-1, 1)
                )
                
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
                
            def encode(self, x):
                return self.encoder(x)
        
        return Autoencoder(input_dim, self.n_components).to(self.device)
        
    def _train_autoencoder(self, X, epochs=10, batch_size=32, learning_rate=0.001):
        """Train the autoencoder for feature extraction"""
        if isinstance(X, pd.DataFrame):
            X_tensor = torch.FloatTensor(X.values).to(self.device)
        else:
            X_tensor = torch.FloatTensor(X).to(self.device)
            
        self.autoencoder = self._build_autoencoder(X_tensor.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                outputs = self.autoencoder(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}')
        
        self.autoencoder.eval()
    
    def feature_ranking(self, X, y=None):
        """Rank features by importance using multiple methods"""
        if y is None:
            # If no target is provided, use the next close price as target
            if isinstance(X, pd.DataFrame):
                if 'close' in X.columns:
                    y = X['close'].shift(-1).iloc[:-1].values
                    X = X.iloc[:-1]
                else:
                    # Use PCA without target variable
                    pca = PCA(n_components=min(X.shape[1], 50))
                    pca.fit(X)
                    return pd.Series(pca.explained_variance_ratio_, 
                                    index=X.columns if isinstance(X, pd.DataFrame) else range(X.shape[1]))
            else:
                # Use PCA without target variable
                pca = PCA(n_components=min(X.shape[1], 50))
                pca.fit(X)
                return pd.Series(pca.explained_variance_ratio_, 
                                index=range(X.shape[1]))
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Rank by multiple methods
        rankings = {}
        
        # 1. Mutual Information
        try:
            mi_selector = SelectKBest(mutual_info_regression, k='all')
            mi_selector.fit(X, y)
            rankings['mutual_info'] = pd.Series(mi_selector.scores_, index=feature_names)
        except:
            pass
            
        # 2. F-regression
        try:
            f_selector = SelectKBest(f_regression, k='all')
            f_selector.fit(X, y)
            rankings['f_regression'] = pd.Series(f_selector.scores_, index=feature_names)
        except:
            pass
            
        # 3. PCA explained variance
        try:
            pca = PCA(n_components=min(X.shape[1], 50))
            pca.fit(X)
            loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), 
                                  columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                                  index=feature_names)
            # Sum absolute loadings across components, weighted by explained variance
            rankings['pca_weighted'] = loadings.abs().multiply(pca.explained_variance_ratio_, axis=1).sum(axis=1)
        except:
            pass
            
        # 4. Correlation with target
        try:
            if isinstance(X, pd.DataFrame):
                X_df = X.copy()
            else:
                X_df = pd.DataFrame(X, columns=feature_names)
                
            X_df['target'] = y
            correlations = X_df.corr()['target'].drop('target')
            rankings['correlation'] = correlations.abs()
        except:
            pass
        
        # Combine rankings (normalize each method's scores and average)
        combined_rank = None
        count = 0
        
        for method_name, ranking in rankings.items():
            normalized_rank = (ranking - ranking.min()) / (ranking.max() - ranking.min() + 1e-10)
            if combined_rank is None:
                combined_rank = normalized_rank
            else:
                combined_rank += normalized_rank
            count += 1
            
        if combined_rank is not None:
            combined_rank /= count
            return combined_rank.sort_values(ascending=False)
        else:
            # Fallback to PCA 
            pca = PCA(n_components=min(X.shape[1], 50))
            pca.fit(X)
            return pd.Series(pca.explained_variance_ratio_, 
                           index=feature_names)
                
    def fit(self, X, y=None, train_ae=False):
        """
        Fit the feature processor to the training data
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The training features
        y : pandas Series or numpy array, optional
            The target variable (next price movement for DQN)
        train_ae : bool, default=False
            Whether to train the autoencoder (more computationally intensive)
        """
        # Save original column names if dataframe
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        # Remove highly correlated features
        X_filtered, dropped_cols = self.remove_highly_correlated(X)
        print(f"Removed {len(dropped_cols)} highly correlated features")
        
        if isinstance(X, pd.DataFrame) and isinstance(X_filtered, pd.DataFrame):
            self.filtered_feature_names = X_filtered.columns.tolist()
        else:
            self.filtered_feature_names = [feat for i, feat in enumerate(self.feature_names) 
                                          if i not in dropped_cols]
        
        # Fit the scaler
        if self.scaler is not None:
            self.scaler.fit(X_filtered)
            X_scaled = self.scaler.transform(X_filtered)
        else:
            X_scaled = X_filtered
            
        # Feature selection/extraction
        if self.selection_method == 'pca':
            self.selector = PCA(n_components=min(self.n_components, X_scaled.shape[1]))
            self.selector.fit(X_scaled)
            self.feature_importance = self.selector.explained_variance_ratio_
            
        elif self.selection_method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_regression, k=self.n_components)
            self.selector.fit(X_scaled, y)
            self.feature_importance = self.selector.scores_
            self.selected_features = [self.filtered_feature_names[i] for i in self.selector.get_support(indices=True)]
            
        elif self.selection_method == 'f_regression':
            self.selector = SelectKBest(f_regression, k=self.n_components)
            self.selector.fit(X_scaled, y)
            self.feature_importance = self.selector.scores_
            self.selected_features = [self.filtered_feature_names[i] for i in self.selector.get_support(indices=True)]
            
        elif self.selection_method == 'autoencoder' and train_ae:
            self._train_autoencoder(X_scaled)
            # No explicit feature importance for autoencoder
            
        elif self.selection_method == 'combined':
            # Rank features by importance
            feature_ranks = self.feature_ranking(X_filtered, y)
            self.feature_importance = feature_ranks
            self.selected_features = feature_ranks.index[:self.n_components].tolist()
            
            # Create a selector based on the top features
            self.selector = SelectKBest(mutual_info_regression, k=self.n_components)
            self.selector.fit(X_scaled, y)
            
        return self
    
    def transform(self, X):
        """
        Transform features using the fitted processor
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            The features to transform
        
        Returns:
        --------
        numpy array
            The selected/extracted features
        """
        # Handle DataFrame or numpy array
        if isinstance(X, pd.DataFrame):
            X_filtered = X[self.filtered_feature_names]
        else:
            # This is more complex with numpy arrays - we need to select columns
            # Would need additional logic to track column indices
            X_filtered = X

        # Apply scaling
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_filtered)
        else:
            X_scaled = X_filtered
        
        # Apply feature selection/extraction
        if self.selection_method == 'pca' and self.selector is not None:
            return self.selector.transform(X_scaled)
            
        elif self.selection_method in ['mutual_info', 'f_regression', 'combined'] and self.selector is not None:
            return self.selector.transform(X_scaled)
            
        elif self.selection_method == 'autoencoder' and self.autoencoder is not None:
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            with torch.no_grad():
                encoded = self.autoencoder.encode(X_tensor)
            return encoded.cpu().numpy()
            
        elif self.selection_method is None:
            return X_scaled
            
        else:
            # Fallback: return the original filtered data
            return X_filtered
    
    def fit_transform(self, X, y=None, train_ae=False):
        """Fit and transform in one step"""
        self.fit(X, y, train_ae)
        return self.transform(X)
        
    def save(self, filepath):
        """Save the feature processor to a file"""
        if self.selection_method == 'autoencoder' and self.autoencoder is not None:
            # Save autoencoder separately
            torch.save(self.autoencoder.state_dict(), f"{filepath}_autoencoder.pt")
            # Temporarily set autoencoder to None for joblib serialization
            temp_ae = self.autoencoder
            self.autoencoder = None
            joblib.dump(self, filepath)
            # Restore autoencoder
            self.autoencoder = temp_ae
        else:
            joblib.dump(self, filepath)
            
    @classmethod
    def load(cls, filepath):
        """Load a feature processor from a file"""
        processor = joblib.load(filepath)
        # Check if there's a saved autoencoder
        try:
            ae_path = f"{filepath}_autoencoder.pt"
            if processor.selection_method == 'autoencoder' and os.path.exists(ae_path):
                input_dim = len(processor.filtered_feature_names)
                processor.autoencoder = processor._build_autoencoder(input_dim)
                processor.autoencoder.load_state_dict(torch.load(ae_path))
                processor.autoencoder.eval()
        except:
            pass
        return processor


class RollingWindowFeatureProcessor:
    """
    Apply feature processing to rolling windows of data
    Designed to work with DQN models for stock trading
    """
    def __init__(self, 
                 window_size=WINDOW_SIZE, 
                 feature_processor=None,
                 flatten_output=True):
        """
        Parameters:
        -----------
        window_size : int
            The size of the rolling window for observations
        feature_processor : FeatureProcessor object
            The feature processor to use on each window
        flatten_output : bool
            Whether to flatten the output for use with fully connected networks
        """
        self.window_size = window_size
        self.feature_processor = FeatureProcessor(window_size=window_size) if feature_processor is None else feature_processor
        self.flatten_output = flatten_output
        
    def fit(self, X, y=None, train_ae=False):
        """
        Fit the feature processor to the full dataset
        For time series, y would typically be the future price movement
        """
        if y is not None and len(y) == len(X):
            self.feature_processor.fit(X, y, train_ae)
        else:
            # Create synthetic target as next close price movement
            if isinstance(X, pd.DataFrame) and 'target' in X.columns:
                target = X['close'].pct_change().shift(-1).iloc[:-1]
                self.feature_processor.fit(X.iloc[:-1], target, train_ae)
            else:
                self.feature_processor.fit(X, None, train_ae)
        return self
    
    def transform_single_window(self, window):
        """Transform a single window of data"""
        return self.feature_processor.transform(window)
    
    def create_rolling_windows(self, X):
        """Create rolling windows from sequential data"""
        if len(X) < self.window_size:
            raise ValueError(f"Input data length {len(X)} is less than window size {self.window_size}")
            
        windows = []
        for i in range(len(X) - self.window_size + 1):
            windows.append(X[i:i+self.window_size])
        return windows
    
    def transform(self, X):
        """
        Transform features using rolling windows
        
        Parameters:
            X (pandas.DataFrame or numpy.ndarray): The features to transform
        
        Returns:
            numpy.ndarray: list of numpy arrays with processed features for each window
        """
        windows = self.create_rolling_windows(X)
        processed_windows = []
        
        for window in windows:
            processed = self.transform_single_window(window)
            if self.flatten_output:
                processed = processed.reshape(1, -1)
            processed_windows.append(processed)
            
        return processed_windows
    
    def get_state(self, X):
        """
        Get the processed state at time t for reinforcement learning.

        Args:
            X (pandas.DataFrame or numpy.ndarray): The full features dataset.

        Returns:
            numpy.ndarray: The processed state for the DQN.
        """
        processed = self.transform_single_window(X)
        if self.flatten_output:
            return processed.flatten()
        else:
            return processed
    
    def save(self, filepath):
        """Save the rolling window processor"""
        self.feature_processor.save(f"{filepath}_feature_processor")
        joblib.dump(self, filepath)
        
    @classmethod
    def load(cls, filepath):
        """Load a rolling window processor"""
        processor = joblib.load(filepath)
        processor.feature_processor = FeatureProcessor.load(f"{filepath}_feature_processor")
        return processor