import numpy as np
import pickle
from typing import List, Dict


class PortfolioStateNormalizer:
    """Normalizes portfolio states using statistics collected during warmup period"""
    
    def __init__(self, warmup_episodes: int = 50, update_frequency: int = 100):
        self.warmup_episodes = warmup_episodes
        self.update_frequency = update_frequency
        self.episode_count = 0
        
        # Feature indices that need normalization (unbounded features)
        self.normalize_features = [4]  # unrealized_pnl index
        self.clip_features = [3]       # position_ratio index (clip to 0-2)
        
        # Statistics storage
        self.feature_stats = {}
        self.warmup_data = {idx: [] for idx in self.normalize_features}
        self.is_fitted = False
        
    def collect_warmup_data(self, portfolio_states: List[np.ndarray]):
        """Collect portfolio states during warmup period"""
        if self.episode_count < self.warmup_episodes:
            for state in portfolio_states:
                for idx in self.normalize_features:
                    if idx < len(state):
                        self.warmup_data[idx].append(state[idx])
    
    def fit_normalizer(self):
        """Fit normalizer using collected warmup data"""
        if self.episode_count >= self.warmup_episodes and not self.is_fitted:
            print(f"Fitting portfolio normalizer with {self.warmup_episodes} episodes of data...")
            
            for idx in self.normalize_features:
                data = np.array(self.warmup_data[idx])
                if len(data) > 0:
                    # Use robust statistics (less sensitive to outliers)
                    median = np.median(data)
                    q25, q75 = np.percentile(data, [25, 75])
                    iqr = q75 - q25
                    
                    # Handle edge case where IQR is zero
                    if iqr < 1e-6:
                        iqr = max(abs(median), 0.01)  # Fallback scaling
                    
                    self.feature_stats[idx] = {
                        'median': median,
                        'iqr': iqr,
                        'q25': q25,
                        'q75': q75,
                        'min': np.min(data),
                        'max': np.max(data)
                    }
                    
                    print(f"Feature {idx} (unrealized_pnl) stats:")
                    print(f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
                    print(f"  Median: {median:.3f}, IQR: {iqr:.3f}")
            
            self.is_fitted = True
            # Clear warmup data to save memory
            self.warmup_data.clear()
            print("✅ Portfolio normalization ACTIVATED!")
    
    def fit_from_sample_data(self, sample_portfolio_states: List[np.ndarray]):
        """Pre-fit normalizer using sample data (alternative to warmup)"""
        if self.is_fitted:
            return
            
        print(f"Pre-fitting portfolio normalizer with {len(sample_portfolio_states)} sample states...")
        
        for idx in self.normalize_features:
            data = []
            for state in sample_portfolio_states:
                if idx < len(state):
                    data.append(state[idx])
            
            if len(data) > 0:
                data = np.array(data)
                median = np.median(data)
                q25, q75 = np.percentile(data, [25, 75])
                iqr = q75 - q25
                
                if iqr < 1e-6:
                    iqr = max(abs(median), 0.01)
                
                self.feature_stats[idx] = {
                    'median': median,
                    'iqr': iqr,
                    'q25': q25,
                    'q75': q75,
                    'min': np.min(data),
                    'max': np.max(data)
                }
                
                print(f"Pre-fit feature {idx} (unrealized_pnl) stats:")
                print(f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
                print(f"  Median: {median:.3f}, IQR: {iqr:.3f}")
        
        self.is_fitted = True
        print("✅ Portfolio normalizer PRE-FITTED - Training enabled from start!")
            
    def normalize_state(self, portfolio_state: np.ndarray) -> np.ndarray:
        """Normalize a single portfolio state"""
        if not self.is_fitted:
            return portfolio_state  # Return unchanged during warmup
            
        state = portfolio_state.copy()
        
        # Normalize unbounded features using robust scaling
        for idx in self.normalize_features:
            if idx < len(state) and idx in self.feature_stats:
                stats = self.feature_stats[idx]
                # Robust scaling: (x - median) / IQR
                state[idx] = (state[idx] - stats['median']) / stats['iqr']
                # Clip extreme outliers to [-3, 3] (roughly 3 IQRs)
                state[idx] = np.clip(state[idx], -3.0, 3.0)
        
        # Clip features that should be bounded
        for idx in self.clip_features:
            if idx < len(state):
                state[idx] = np.clip(state[idx], 0.0, 2.0)  # Allow up to 200% position ratio
        
        return state
    
    def increment_episode(self):
        """Call this after each episode"""
        self.episode_count += 1
        
        # Fit normalizer after warmup period
        if self.episode_count == self.warmup_episodes:
            self.fit_normalizer()
    
    def save(self, path: str):
        """Save normalizer state"""
        with open(path, 'wb') as f:
            pickle.dump({
                'feature_stats': self.feature_stats,
                'is_fitted': self.is_fitted,
                'episode_count': self.episode_count,
                'warmup_episodes': self.warmup_episodes
            }, f)
    
    def load(self, path: str):
        """Load normalizer state"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.feature_stats = data['feature_stats']
                self.is_fitted = data['is_fitted']
                self.episode_count = data['episode_count']
                self.warmup_episodes = data.get('warmup_episodes', 50)
            print(f"Loaded portfolio normalizer from {path}")
        except FileNotFoundError:
            print(f"No existing normalizer found at {path}, starting fresh") 