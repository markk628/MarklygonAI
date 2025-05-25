import numpy as np
import pandas as pd
from collections import deque
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from scipy.stats import mstats

from src.config.config import WINDOW_SIZE, PRICE_FEATURES, TEMPORAL_FEATURES

# def create_windows(X: pd.DataFrame,
#                    window_size: int=WINDOW_SIZE,
#                    winsorize_limits=(0.01, 0.99)):
#     windows = []
#     for i in range(window_size, len(X)):
#         power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
#         scaler = MinMaxScaler(feature_range=(-1, 1))
        
#         window_df = X.iloc[i-window_size:i]
#         winsorized_df = pd.DataFrame(
#             mstats.winsorize(window_df.values, 
#                            limits=winsorize_limits, 
#                            axis=0),
#             columns=window_df.columns,
#             index=window_df.index
#         )
        
#         transformed_df = power_transformer.fit_transform(winsorized_df)
#         scaled_df = scaler.fit_transform(transformed_df)
#         windows.append(scaled_df)
#     return windows
        

class StateScaler:
    def __init__(self, window_size=1000, adaptation_period=100, correlation_threshold=0.45):
        self.window_size = window_size
        self.adaptation_period = adaptation_period
        self.step_count = 0
        self.features = []
        self.correlation_threshold = correlation_threshold
        
        # Use deques for O(1) append/popleft operations
        self.rolling_data = {}
        
        # Cache frequently computed values
        self._stats_cache = {}
        self._cache_valid = {}
        
        # Pre-allocate arrays to avoid repeated allocations
        self._temp_array = np.empty(window_size, dtype=np.float32)
        
    def _get_or_create_metric(self, metric_name):
        """Get or create metric data structure"""
        if metric_name not in self.rolling_data:
            self.rolling_data[metric_name] = {
                'values': deque(maxlen=self.window_size),
                'min_val': float('inf'),
                'max_val': float('-inf'),
                'sum_val': 0.0,
                'sum_sq': 0.0,
                'count': 0
            }
            self._cache_valid[metric_name] = False
        return self.rolling_data[metric_name]
    
    def update_rolling_stats(self, metric_name, value):
        """Update rolling statistics - optimized version"""
        data = self._get_or_create_metric(metric_name)
        
        # Handle window overflow efficiently
        if len(data['values']) == self.window_size:
            old_value = data['values'][0]  # Will be removed by deque
            data['sum_val'] -= old_value
            data['sum_sq'] -= old_value * old_value
        else:
            data['count'] += 1
        
        # Add new value
        data['values'].append(value)
        data['min_val'] = min(data['min_val'], value)
        data['max_val'] = max(data['max_val'], value)
        data['sum_val'] += value
        data['sum_sq'] += value * value
        
        # Invalidate cache
        self._cache_valid[metric_name] = False
    
    def _compute_stats_cached(self, metric_name):
        """Compute statistics with caching"""
        if self._cache_valid.get(metric_name, False):
            return self._stats_cache[metric_name]
        
        data = self.rolling_data[metric_name]
        count = len(data['values'])
        
        if count == 0:
            stats = {
                'min': 0.0, 'max': 1.0, 'mean': 0.0, 
                'std': 1.0, 'median': 0.0, 'q25': 0.0, 'q75': 1.0
            }
        else:
            mean = data['sum_val'] / count
            variance = max((data['sum_sq'] / count) - (mean * mean), 1e-8)
            std = variance ** 0.5  # Faster than np.sqrt for scalars
            
            # Only compute percentiles if we have enough data and need them
            if count > 10:
                # Convert deque to array efficiently
                values_len = len(data['values'])
                if values_len <= len(self._temp_array):
                    arr_view = self._temp_array[:values_len]
                    for i, val in enumerate(data['values']):
                        arr_view[i] = val
                else:
                    arr_view = np.fromiter(data['values'], dtype=np.float32)
                
                # Use faster percentile computation
                arr_view.sort()  # In-place sort
                p1_idx = max(0, int(0.01 * values_len))
                p99_idx = min(values_len - 1, int(0.99 * values_len))
                p25_idx = int(0.25 * values_len)
                p75_idx = int(0.75 * values_len)
                median_idx = values_len // 2
                
                stats = {
                    'min': arr_view[p1_idx],
                    'max': arr_view[p99_idx],
                    'mean': mean,
                    'std': std,
                    'median': arr_view[median_idx],
                    'q25': arr_view[p25_idx],
                    'q75': arr_view[p75_idx]
                }
            else:
                stats = {
                    'min': data['min_val'],
                    'max': data['max_val'],
                    'mean': mean,
                    'std': std,
                    'median': mean,
                    'q25': mean,
                    'q75': mean
                }
        
        # Cache results
        self._stats_cache[metric_name] = stats
        self._cache_valid[metric_name] = True
        return stats
    
    def scale_value(self, value, metric_name, method='adaptive'):
        """Optimized scaling with early returns"""
        # Update stats first
        self.update_rolling_stats(metric_name, value)
        
        # Fast path for adaptive method (most common)
        if method == 'adaptive':
            return self._adaptive_scale_fast(value, metric_name)
        elif method == 'robust_zscore':
            return self._robust_zscore_scale_fast(value, metric_name)
        elif method == 'rank_based':
            return self._rank_based_scale_fast(value, metric_name)
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def _adaptive_scale_fast(self, value, metric_name):
        """Optimized adaptive scaling"""
        data = self.rolling_data[metric_name]
        count = len(data['values'])
        
        # Early phase: simple operations only
        if count < 50:
            # Avoid np.clip and np.tanh - use faster alternatives
            clipped = max(-10.0, min(10.0, value))
            x = clipped / 5.0
            # Fast tanh approximation: tanh(x) ≈ x / (1 + |x|) for small x
            return x / (1.0 + abs(x)) if abs(x) < 2 else (1.0 if x > 0 else -1.0)
        
        # Get cached stats
        stats = self._compute_stats_cached(metric_name)
        
        # Medium phase
        if count < 200:
            return self._minmax_scale_fast(value, stats)
        
        # Mature phase: hybrid scaling
        minmax_scaled = self._minmax_scale_fast(value, stats)
        zscore_scaled = self._zscore_scale_fast(value, stats)
        return 0.7 * minmax_scaled + 0.3 * zscore_scaled
    
    def _minmax_scale_fast(self, value, stats):
        """Fast min-max scaling"""
        min_val, max_val = stats['min'], stats['max']
        
        if max_val == min_val:
            return 0.5
        
        range_val = max_val - min_val
        clipped_value = max(min_val, min(max_val, value))
        return (clipped_value - min_val) / range_val
    
    def _zscore_scale_fast(self, value, stats):
        """Fast z-score scaling"""
        mean_val, std_val = stats['mean'], stats['std']
        
        if std_val == 0:
            return 0.5
        
        z_score = (value - mean_val) / std_val
        x = z_score / 3.0
        # Fast tanh approximation
        return x / (1.0 + abs(x)) if abs(x) < 2 else (1.0 if x > 0 else -1.0)
    
    def _robust_zscore_scale_fast(self, value, metric_name):
        """Fast robust z-score"""
        data = self.rolling_data[metric_name]
        
        if len(data['values']) < 10:
            x = value / 5.0
            return x / (1.0 + abs(x)) if abs(x) < 2 else (1.0 if x > 0 else -1.0)
        
        stats = self._compute_stats_cached(metric_name)
        median = stats['median']
        iqr = stats['q75'] - stats['q25']
        
        if iqr == 0:
            return 0.5
        
        robust_z = (value - median) / iqr
        x = robust_z / 2.0
        return x / (1.0 + abs(x)) if abs(x) < 2 else (1.0 if x > 0 else -1.0)
    
    def _rank_based_scale_fast(self, value, metric_name):
        """Fast rank-based scaling"""
        data = self.rolling_data[metric_name]
        values_len = len(data['values'])
        
        if values_len < 10:
            return 0.5
        
        # Use only recent values for efficiency
        recent_count = min(100, values_len)
        
        # Count values less than current value
        rank = sum(1 for v in list(data['values'])[-recent_count:] if v < value)
        return rank / recent_count
    
    def scale_state_vector(self, state_dict, method='adaptive'):
        """Optimized state vector scaling"""
        # Pre-allocate result array
        result = np.empty(len(state_dict), dtype=np.float32)
        
        # Process all metrics
        for i, (metric_name, value) in enumerate(state_dict.items()):
            result[i] = self.scale_value(value, metric_name, method)
        
        self.step_count += 1
        return result
    
    def get_scaling_info(self, metric_name):
        """Get scaling information - simplified"""
        if metric_name not in self.rolling_data:
            return {
                'sample_count': 0,
                'scaling_phase': "early",
                'step_count': self.step_count
            }
        
        count = len(self.rolling_data[metric_name]['values'])
        return {
            'sample_count': count,
            'scaling_phase': ("early" if count < 50 else 
                             "medium" if count < 200 else "mature"),
            'step_count': self.step_count
        }
    
    def save_state(self):
        """Optimized state saving"""
        return {
            'rolling_data': {
                name: {
                    'values': list(data['values']),
                    'min_val': data['min_val'],
                    'max_val': data['max_val'],
                    'sum_val': data['sum_val'],
                    'sum_sq': data['sum_sq'],
                    'count': data['count']
                }
                for name, data in self.rolling_data.items()
            },
            'step_count': self.step_count,
            'window_size': self.window_size,
            'adaptation_period': self.adaptation_period
        }
    
    def load_state(self, state):
        """Optimized state loading"""
        self.rolling_data = {}
        self._stats_cache = {}
        self._cache_valid = {}
        
        for name, data in state['rolling_data'].items():
            self.rolling_data[name] = {
                'values': deque(data['values'], maxlen=self.window_size),
                'min_val': data['min_val'],
                'max_val': data['max_val'],
                'sum_val': data['sum_val'],
                'sum_sq': data['sum_sq'],
                'count': data['count']
            }
            self._cache_valid[name] = False
        
        self.step_count = state['step_count']
        self.window_size = state['window_size']
        self.adaptation_period = state['adaptation_period']
        
    def scale_stock_data(self, window: NDArray, winsorize_limits=(0.01, 0.99)) -> NDArray:
        power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        windsorized = mstats.winsorize(window, limits=winsorize_limits, axis=0)
        transformed = power_transformer.fit_transform(windsorized)
        return scaler.fit_transform(transformed)
    
    def remove_highly_correlated(self, X):
        """Remove highly correlated features"""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold) and column not in (PRICE_FEATURES + TEMPORAL_FEATURES)]
        print(f"Removed {len(to_drop)} highly correlated features")
        self.features = X.drop(columns=to_drop).columns
        print(len(self.features))


# Alternative: Simple Online Scaler for immediate use
# class SimpleOnlineScaler:
#     def __init__(self, alpha=0.01, initial_std=1.0):
#         self.alpha = alpha  # Learning rate for exponential moving average
#         self.initial_std = initial_std
        
#         # Online statistics
#         self.means = defaultdict(float)
#         self.vars = defaultdict(lambda: initial_std**2)
#         self.mins = defaultdict(lambda: float('inf'))
#         self.maxs = defaultdict(lambda: float('-inf'))
#         self.counts = defaultdict(int)
    
#     def update_stats(self, metric_name, value):
#         """Update online statistics"""
#         self.counts[metric_name] += 1
        
#         # Update min/max
#         self.mins[metric_name] = min(self.mins[metric_name], value)
#         self.maxs[metric_name] = max(self.maxs[metric_name], value)
        
#         # Exponential moving average for mean and variance
#         if self.counts[metric_name] == 1:
#             self.means[metric_name] = value
#             self.vars[metric_name] = self.initial_std**2
#         else:
#             # Update mean
#             old_mean = self.means[metric_name]
#             self.means[metric_name] += self.alpha * (value - old_mean)
            
#             # Update variance (Welford's online algorithm adaptation)
#             self.vars[metric_name] += self.alpha * ((value - old_mean) * (value - self.means[metric_name]) - self.vars[metric_name])
    
#     def scale_value(self, value, metric_name, method='zscore'):
#         """Scale a value using online statistics"""
#         self.update_stats(metric_name, value)
        
#         if method == 'zscore':
#             mean = self.means[metric_name]
#             std = np.sqrt(max(self.vars[metric_name], 1e-8))
#             z_score = (value - mean) / std
#             return np.tanh(z_score / 3.0)
        
#         elif method == 'minmax':
#             min_val = self.mins[metric_name]
#             max_val = self.maxs[metric_name]
            
#             if max_val == min_val:
#                 return 0.5
            
#             clipped = np.clip(value, min_val, max_val)
#             return (clipped - min_val) / (max_val - min_val)
        
#         else:
#             raise ValueError(f"Unknown method: {method}")
    
#     def scale_state_vector(self, state_dict, method='zscore'):
#         """Scale entire state vector"""
#         scaled_state = []
        
#         for metric_name, value in state_dict.items():
#             scaled_value = self.scale_value(value, metric_name, method)
#             scaled_state.append(scaled_value)
        
#         return np.array(scaled_state, dtype=np.float32)


# # Usage example
# if __name__ == "__main__":
#     # Initialize adaptive scaler
#     scaler = AdaptiveStateScaler(window_size=1000)
    
#     # Simulate training loop
#     for step in range(1000):
#         # Simulate some state metrics
#         state = {
#             'cpu_usage': np.random.normal(50, 20),
#             'memory_usage': np.random.exponential(30),
#             'network_io': np.random.gamma(2, 10),
#             'response_time': np.random.lognormal(2, 0.5)
#         }
        
#         # Scale the state
#         scaled_state = scaler.scale_state_vector(state)
        
#         # Print info every 100 steps
#         if step % 100 == 0:
#             print(f"Step {step}:")
#             for metric in state.keys():
#                 info = scaler.get_scaling_info(metric)
#                 print(f"  {metric}: {info['scaling_phase']} phase, {info['sample_count']} samples")
#             print()