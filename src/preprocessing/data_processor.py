import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from scipy import stats
import joblib
import torch
from torch import nn
import warnings
warnings.filterwarnings('ignore')

from src.config.config import (
    WINDOW_SIZE,
    DEVICE,
    PRICE_FEATURES,
    TEMPORAL_FEATURES
)

class FeatureProcessor:
    """
    Modular feature selection/extraction for stock trading data
    Compatible with rolling window scaling and DQN models
    Enhanced with outlier handling and distribution normalization
    """
    def __init__(self, 
                 window_size=WINDOW_SIZE,
                 scaler_type='standard',  # 'standard', 'minmax', 'quantile', 'power', or None
                 selection_method=None,  # 'pca', 'mutual_info', 'f_regression', 'autoencoder', 'combined', None
                 n_components=22,         # Number of features to select
                 # Outlier handling parameters
                 winsorize_limits=(0.01, 0.01),  # Lower and upper percentiles for winsorization
                 outlier_method='winsorize',  # 'winsorize', 'clip', 'zscore', or None
                 outlier_threshold=3.0,  # Z-score threshold for outlier detection
                 # Distribution normalization parameters
                 distribution_method='quantile',  # 'quantile', 'power', 'log', or None
                 quantile_output='uniform',  # 'uniform' or 'normal' for QuantileTransformer
                 power_method='yeo-johnson',  # 'yeo-johnson' or 'box-cox' for PowerTransformer
                 device=DEVICE):
        
        self.window_size = window_size
        self.scaler_type = scaler_type
        self.selection_method = selection_method
        self.n_components = n_components
        self.device = device
        
        # Outlier handling parameters
        self.winsorize_limits = winsorize_limits
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        # Distribution normalization parameters
        self.distribution_method = distribution_method
        self.quantile_output = quantile_output
        self.power_method = power_method
        
        # Initialize scalers and selectors
        self._initialize_scalers()
        
        # Initialize feature selectors/extractors
        self.selector = None
        self.autoencoder = None
        self.selected_features = None
        self.feature_importance = None
        
        # Store outlier bounds for consistent processing
        self.outlier_bounds = {}
        
    def _initialize_scalers(self):
        """Initialize scalers based on configuration"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'quantile':
            self.scaler = QuantileTransformer(output_distribution=self.quantile_output, 
                                            subsample=100000, random_state=42)
        elif self.scaler_type == 'power':
            self.scaler = PowerTransformer(method=self.power_method, standardize=True)
        else:
            self.scaler = None
            
        # Distribution transformer (separate from scaler)
        if self.distribution_method == 'quantile':
            self.distribution_transformer = QuantileTransformer(output_distribution=self.quantile_output,
                                                              subsample=100000, random_state=42)
        elif self.distribution_method == 'power':
            self.distribution_transformer = PowerTransformer(method=self.power_method, standardize=False)
        else:
            self.distribution_transformer = None
    
    def _winsorize_data(self, X, feature_names=None):
        """Apply winsorization to handle outliers"""
        if isinstance(X, pd.DataFrame):
            X_win = X.copy()
            feature_names = X.columns if feature_names is None else feature_names
            
            for col in feature_names:
                # Skip price and temporal features for winsorization (they have natural bounds)
                if col in PRICE_FEATURES + TEMPORAL_FEATURES:
                    continue
                    
                lower_bound = X[col].quantile(self.winsorize_limits[0])
                upper_bound = X[col].quantile(1 - self.winsorize_limits[1])
                
                # Store bounds for transform phase
                self.outlier_bounds[col] = (lower_bound, upper_bound)
                
                X_win[col] = X[col].clip(lower_bound, upper_bound)
                
            return X_win
        else:
            X_win = X.copy()
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            for i, col in enumerate(feature_names):
                # Skip price and temporal features
                if col in PRICE_FEATURES + TEMPORAL_FEATURES:
                    continue
                    
                lower_bound = np.percentile(X[:, i], self.winsorize_limits[0] * 100)
                upper_bound = np.percentile(X[:, i], (1 - self.winsorize_limits[1]) * 100)
                
                self.outlier_bounds[col] = (lower_bound, upper_bound)
                X_win[:, i] = np.clip(X[:, i], lower_bound, upper_bound)
                
            return X_win
    
    def _clip_outliers(self, X, feature_names=None):
        """Clip outliers based on IQR method"""
        if isinstance(X, pd.DataFrame):
            X_clipped = X.copy()
            feature_names = X.columns if feature_names is None else feature_names
            
            for col in feature_names:
                if col in PRICE_FEATURES + TEMPORAL_FEATURES:
                    continue
                    
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.outlier_bounds[col] = (lower_bound, upper_bound)
                X_clipped[col] = X[col].clip(lower_bound, upper_bound)
                
            return X_clipped
        else:
            X_clipped = X.copy()
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            for i, col in enumerate(feature_names):
                if col in PRICE_FEATURES + TEMPORAL_FEATURES:
                    continue
                    
                Q1 = np.percentile(X[:, i], 25)
                Q3 = np.percentile(X[:, i], 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.outlier_bounds[col] = (lower_bound, upper_bound)
                X_clipped[:, i] = np.clip(X[:, i], lower_bound, upper_bound)
                
            return X_clipped
    
    def _zscore_outliers(self, X, feature_names=None):
        """Remove outliers based on Z-score threshold"""
        if isinstance(X, pd.DataFrame):
            X_clean = X.copy()
            feature_names = X.columns if feature_names is None else feature_names
            
            for col in feature_names:
                if col in PRICE_FEATURES + TEMPORAL_FEATURES:
                    continue
                    
                z_scores = np.abs(stats.zscore(X[col], nan_policy='omit'))
                mean_val = X[col].mean()
                std_val = X[col].std()
                lower_bound = mean_val - self.outlier_threshold * std_val
                upper_bound = mean_val + self.outlier_threshold * std_val
                
                self.outlier_bounds[col] = (lower_bound, upper_bound)
                X_clean[col] = X[col].clip(lower_bound, upper_bound)
                
            return X_clean
        else:
            X_clean = X.copy()
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            for i, col in enumerate(feature_names):
                if col in PRICE_FEATURES + TEMPORAL_FEATURES:
                    continue
                    
                z_scores = np.abs(stats.zscore(X[:, i], nan_policy='omit'))
                mean_val = np.nanmean(X[:, i])
                std_val = np.nanstd(X[:, i])
                lower_bound = mean_val - self.outlier_threshold * std_val
                upper_bound = mean_val + self.outlier_threshold * std_val
                
                self.outlier_bounds[col] = (lower_bound, upper_bound)
                X_clean[:, i] = np.clip(X[:, i], lower_bound, upper_bound)
                
            return X_clean
    
    def _apply_log_transform(self, X, feature_names=None):
        """Apply log transformation for positive skewed data"""
        if isinstance(X, pd.DataFrame):
            X_log = X.copy()
            feature_names = X.columns if feature_names is None else feature_names
            
            for col in feature_names:
                # Skip price features and features with non-positive values
                if col in PRICE_FEATURES + TEMPORAL_FEATURES:
                    continue
                    
                # Check if all values are positive
                if (X[col] > 0).all():
                    # Check if data is positively skewed
                    skewness = stats.skew(X[col])
                    if skewness > 1:  # Moderate to high positive skew
                        X_log[col] = np.log1p(X[col])  # log1p for numerical stability
                        
            return X_log
        else:
            X_log = X.copy()
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            for i, col in enumerate(feature_names):
                if col in PRICE_FEATURES + TEMPORAL_FEATURES:
                    continue
                    
                # Check if all values are positive
                if np.all(X[:, i] > 0):
                    # Check if data is positively skewed
                    skewness = stats.skew(X[:, i])
                    if skewness > 1:
                        X_log[:, i] = np.log1p(X[:, i])
                        
            return X_log
    
    def handle_outliers(self, X, feature_names=None):
        """Apply selected outlier handling method"""
        if self.outlier_method == 'winsorize':
            return self._winsorize_data(X, feature_names)
        elif self.outlier_method == 'clip':
            return self._clip_outliers(X, feature_names)
        elif self.outlier_method == 'zscore':
            return self._zscore_outliers(X, feature_names)
        else:
            return X
    
    def normalize_distribution(self, X, fit=True):
        """Apply distribution normalization"""
        if self.distribution_method == 'log':
            return self._apply_log_transform(X)
        elif self.distribution_transformer is not None:
            if fit:
                return self.distribution_transformer.fit_transform(X)
            else:
                return self.distribution_transformer.transform(X)
        else:
            return X
    
    def apply_outlier_bounds_transform(self, X, feature_names=None):
        """Apply stored outlier bounds during transform phase"""
        if not self.outlier_bounds:
            return X
            
        if isinstance(X, pd.DataFrame):
            X_bounded = X.copy()
            for col in X.columns:
                if col in self.outlier_bounds:
                    lower_bound, upper_bound = self.outlier_bounds[col]
                    X_bounded[col] = X[col].clip(lower_bound, upper_bound)
            return X_bounded
        else:
            X_bounded = X.copy()
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            for i, col in enumerate(feature_names):
                if col in self.outlier_bounds:
                    lower_bound, upper_bound = self.outlier_bounds[col]
                    X_bounded[:, i] = np.clip(X[:, i], lower_bound, upper_bound)
            return X_bounded
        
    # TODO optimize autoencoder
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
        
        Args:
            X (pandas DataFrame or numpy.array): The training features
            y (pandas Series or numpy.array, optional): The target variable (next price movement for DQN)
            train_ae (bool): Whether to train the autoencoder (more computationally intensive)
        """
        # Save original column names if dataframe
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        print(f"Starting feature processing with {len(self.feature_names)} features")
        
        # Step 1: Handle outliers
        X_clean = self.handle_outliers(X, self.feature_names)
        if self.outlier_method:
            print(f"Applied {self.outlier_method} outlier handling")
        
        # Step 2: Apply distribution normalization (if different from scaling)
        if self.distribution_method and self.distribution_method != self.scaler_type:
            X_normalized = self.normalize_distribution(X_clean, fit=True)
            print(f"Applied {self.distribution_method} distribution normalization")
        else:
            X_normalized = X_clean
        
        if isinstance(X, pd.DataFrame) and isinstance(X_normalized, pd.DataFrame):
            self.filtered_feature_names = X_normalized.columns.tolist()
        else:
            self.filtered_feature_names = [feat for i, feat in enumerate(self.feature_names)]
        
        # Step 4: Fit the scaler
        if self.scaler is not None:
            self.scaler.fit(X_normalized)
            X_scaled = self.scaler.transform(X_normalized)
            print(f"Applied {self.scaler_type} scaling")
        else:
            X_scaled = X_normalized
            
        # Step 5: Feature selection/extraction
        if self.selection_method == 'pca':
            self.selector = PCA(n_components=min(self.n_components, X_scaled.shape[1]))
            self.selector.fit(X_scaled)
            self.feature_importance = self.selector.explained_variance_ratio_
            print(f"Applied PCA with {self.selector.n_components_} components")
            
        elif self.selection_method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_regression, k=self.n_components)
            self.selector.fit(X_scaled, y)
            self.feature_importance = self.selector.scores_
            self.selected_features = [self.filtered_feature_names[i] for i in self.selector.get_support(indices=True)]
            print(f"Applied mutual information feature selection with {self.n_components} features")
            
        elif self.selection_method == 'f_regression':
            self.selector = SelectKBest(f_regression, k=self.n_components)
            self.selector.fit(X_scaled, y)
            self.feature_importance = self.selector.scores_
            self.selected_features = [self.filtered_feature_names[i] for i in self.selector.get_support(indices=True)]
            print(f"Applied f-regression feature selection with {self.n_components} features")
            
        elif self.selection_method == 'autoencoder' and train_ae:
            self._train_autoencoder(X_scaled)
            print(f"Trained autoencoder with {self.n_components} encoding dimensions")
            
        elif self.selection_method == 'combined':
            # Rank features by importance
            feature_ranks = self.feature_ranking(X_normalized, y)
            self.feature_importance = feature_ranks
            self.selected_features = feature_ranks.index[:self.n_components].tolist()
            
            # Create a selector based on the top features
            self.selector = SelectKBest(mutual_info_regression, k=self.n_components)
            self.selector.fit(X_scaled, y)
            print(f"Applied combined feature ranking with {self.n_components} features")
            
        print("Feature processing fit completed")
        return self
    
    def transform(self, X):
        """
        Transform features using the fitted processor
        
        Args:
            X (pandas.DataFrame or numpy.ndarray): The features to transform
        
        Returns:
            numpy.array: The selected/extracted features
        """
        # Step 1: Apply outlier bounds from training
        X_clean = self.apply_outlier_bounds_transform(X, self.feature_names if hasattr(self, 'feature_names') else None)
        
        # Step 2: Apply distribution normalization (if fitted)
        if self.distribution_method and self.distribution_method != self.scaler_type and self.distribution_transformer is not None:
            X_normalized = self.normalize_distribution(X_clean, fit=False)
        else:
            X_normalized = X_clean
        
        # Step 3: Filter to selected features
        if isinstance(X_normalized, pd.DataFrame):
            X_filtered = X_normalized[self.filtered_feature_names]
        else:
            # For numpy arrays, assume same column order as training
            X_filtered = X_normalized

        # Step 4: Apply scaling
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_filtered)
        else:
            X_scaled = X_filtered
        
        # Step 5: Apply feature selection/extraction
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
        Args:
            window_size (int): The size of the rolling window for observations
            feature_processor (FeatureProcessor): The feature processor to use on each window
            flatten_output (bool): Whether to flatten the output for use with fully connected networks
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
        
        Args:
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