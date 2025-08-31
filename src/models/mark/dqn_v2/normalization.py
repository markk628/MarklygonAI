import numpy as np
import pickle
from typing import List, Dict


class PortfolioStateNormalizer:
    """Normalizes portfolio states using statistics collected during warmup period"""
    
    def __init__(self, warmup_episodes: int = 50, update_frequency: int = 100, num_portfolio_features: int = 13):
        self.warmup_episodes = warmup_episodes
        self.update_frequency = update_frequency
        self.episode_count = 0
        self.num_portfolio_features = num_portfolio_features
        
        # ADAPTIVE FEATURE LAYOUT - Works with both 13 and 20+ feature layouts
        if num_portfolio_features >= 20:
            # Enhanced 20-feature layout (with action masking)
            # 0-9: Basic portfolio + time features
            # 10-18: Action validity features (already scaled, skip normalization)
            # 19+: Raw counts/values
            self.normalize_features = [0, 1, 2, 4, 5, 19]  # invalid_actions moved to index 19
            self.clip_features = [3]  # position_ratio
            self.skip_features = list(range(10, 19))  # Action validity features (10-18)
            
            self.feature_names = {
                0: 'balance',
                1: 'position_value', 
                2: 'portfolio_value',
                4: 'unrealized_pnl',
                5: 'holding_time',
                19: 'invalid_actions'  # Updated index
            }
        else:
            # Legacy 13-feature layout (original DQN)
            # 0-12: All features need processing
            self.normalize_features = [0, 1, 2, 4, 5, 12]  # invalid_actions at index 12
            self.clip_features = [3]  # position_ratio
            self.skip_features = []  # No features to skip
            
            self.feature_names = {
                0: 'balance',
                1: 'position_value', 
                2: 'portfolio_value',
                4: 'unrealized_pnl',
                5: 'holding_time',
                12: 'invalid_actions'  # Original index
            }
        
        # Statistics storage
        self.feature_stats = {}
        self.warmup_data = {idx: [] for idx in self.normalize_features}
        self.is_fitted = False
        
        print(f"📊 PortfolioStateNormalizer initialized:")
        print(f"   • Portfolio features: {num_portfolio_features}")
        print(f"   • Features to normalize: {self.normalize_features}")
        print(f"   • Features to skip: {self.skip_features}")
        print(f"   • Features to clip: {self.clip_features}")
    
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
            print(f"Layout: {self.num_portfolio_features} features")
            
            for idx in self.normalize_features:
                data = np.array(self.warmup_data[idx])
                if len(data) > 0:
                    # Use robust statistics (less sensitive to outliers)
                    median = np.median(data)
                    q25, q75 = np.percentile(data, [25, 75])
                    iqr = q75 - q25
                    
                    # Handle edge case where IQR is zero
                    if iqr < 1e-6:
                        # For monetary values, use a reasonable minimum scale
                        if idx in [0, 1, 2]:  # balance, position_value, portfolio_value
                            iqr = max(abs(median) * 0.01, 1000.0)  # At least $1000 or 1% of median
                        elif idx == 5:  # holding_time
                            iqr = max(1.0, abs(median) * 0.1)  # At least 1 step
                        elif idx in [12, 19]:  # invalid_actions (old or new index)
                            iqr = max(1.0, abs(median) * 0.1)  # At least 1 action
                        else:
                            iqr = max(abs(median) * 0.01, 0.01)  # General fallback
                    
                    self.feature_stats[idx] = {
                        'median': median,
                        'iqr': iqr,
                        'q25': q25,
                        'q75': q75,
                        'min': np.min(data),
                        'max': np.max(data)
                    }
                    
                    feature_name = self.feature_names.get(idx, f'feature_{idx}')
                    print(f"  {feature_name} stats:")
                    print(f"    Range: [{np.min(data):.2f}, {np.max(data):.2f}]")
                    print(f"    Median: {median:.2f}, IQR: {iqr:.2f}")
            
            self.is_fitted = True
            # Clear warmup data to save memory
            self.warmup_data.clear()
            print("✅ Portfolio normalization ACTIVATED for all raw features!")
            if self.skip_features:
                print(f"📌 Skipped features {self.skip_features} (action validity signals preserved)")
    
    def fit_from_sample_data(self, sample_portfolio_states: List[np.ndarray]):
        """Pre-fit normalizer using sample data (alternative to warmup)"""
        if self.is_fitted:
            return
            
        print(f"Pre-fitting portfolio normalizer with {len(sample_portfolio_states)} sample states...")
        print(f"Layout: {self.num_portfolio_features} features")
        
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
                
                # Handle edge case where IQR is zero
                if iqr < 1e-6:
                    if idx in [0, 1, 2]:  # monetary values
                        iqr = max(abs(median) * 0.01, 1000.0)
                    elif idx == 5:  # holding_time
                        iqr = max(1.0, abs(median) * 0.1)
                    elif idx in [12, 19]:  # invalid_actions (old or new index)
                        iqr = max(1.0, abs(median) * 0.1)
                    else:
                        iqr = max(abs(median) * 0.01, 0.01)
                
                self.feature_stats[idx] = {
                    'median': median,
                    'iqr': iqr,
                    'q25': q25,
                    'q75': q75,
                    'min': np.min(data),
                    'max': np.max(data)
                }
                
                feature_name = self.feature_names.get(idx, f'feature_{idx}')
                print(f"  {feature_name} stats:")
                print(f"    Range: [{np.min(data):.2f}, {np.max(data):.2f}]")
                print(f"    Median: {median:.2f}, IQR: {iqr:.2f}")
        
        self.is_fitted = True
        print("✅ Portfolio normalizer PRE-FITTED for all raw features!")
        if self.skip_features:
            print(f"📌 Skipped features {self.skip_features} (action validity signals preserved)")
    
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
                
                # Clip extreme outliers to prevent exploding gradients
                if idx in [0, 1, 2]:  # monetary values - allow wider range
                    state[idx] = np.clip(state[idx], -5.0, 5.0)  # 5 IQRs
                elif idx in [5, 12, 19]:  # counts - more conservative clipping
                    state[idx] = np.clip(state[idx], -3.0, 3.0)  # 3 IQRs
                else:  # unrealized_pnl and others
                    state[idx] = np.clip(state[idx], -4.0, 4.0)  # 4 IQRs
        
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