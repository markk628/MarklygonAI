import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple, Optional, Dict
import random
from dataclasses import dataclass
import math
from datetime import datetime, time
from pathlib import Path

# Add portfolio normalizer imports
import pickle

from src.config.config import (
    DEVICE,
    DATA_DIR,
    MODELS_DIR,
    WINDOW_SIZE,
    EVALUATE_INTERVAL,
    INITIAL_BALANCE,
    TRANSACTION_FEE_PERCENT,
    MAX_POSITION_SIZE,
    BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    UPDATE_TARGET_EVERY,
    EPSILON_EARLY_STOPPING_THRESHOLD,
    STOCK_FEATURES_V2,
    NUM_EPISODES,
    TRAIN_RATIO,
    VALID_RATIO,
    TRAIN_INTERVAL,
    MINUTES_PER_TRADING_DAY
)
from src.utils.utils import create_directory
from src.web.models import app, db, BacktestHistory, ModelType, MarklygonModel

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
        
    def collect_warmup_data(self, portfolio_states: list):
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
            print("✅ Portfolio normalization ACTIVATED - Training will now resume!")
    
    def fit_from_sample_data(self, sample_portfolio_states: list):
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

class ArchitectureType(Enum):
    ORIGINAL = "original"
    IMPROVED = "improved"
    HYBRID = "hybrid"

@dataclass
class TradingConfig:
    """Configuration for the DQN agent"""
    # Environment parameters
    initial_balance: float = INITIAL_BALANCE
    transaction_fee_percent: float = TRANSACTION_FEE_PERCENT
    window_size: int = WINDOW_SIZE
    num_stock_features: int = len(STOCK_FEATURES_V2)
    num_portfolio_features: int = 12
    num_features: int = len(STOCK_FEATURES_V2) + 12  # Stock features + portfolio features
    num_actions: int = 3  # Hold, Buy, Sell
    max_position_size: float = MAX_POSITION_SIZE
    
    # Network architecture selection
    architecture_type: ArchitectureType = ArchitectureType.IMPROVED
    
    # Network parameters
    hidden_size: int = 512
    learning_rate: float = 0.0001
    
    # Training parameters
    batch_size: int = BATCH_SIZE
    gamma: float = 0.99
    tau: float = 0.005
    update_frequency: int = TRAIN_INTERVAL
    target_update_frequency: int = UPDATE_TARGET_EVERY
    
    # Experience replay
    buffer_size: int = REPLAY_BUFFER_SIZE
    
    # Exploration - Encourage active trading with strategic decisions
    epsilon_start: float = 1.0
    epsilon_end: float = 0.12  # Increased to encourage more exploration and activity
    epsilon_decay: float = 100000  # Slower decay to maintain activity longer
    
    # Prioritized replay
    use_prioritized_replay: bool = True
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    per_epsilon: float = 0.001  # Add missing PER epsilon
    
    # Double and Dueling DQN
    use_double_dqn: bool = True
    use_dueling_dqn: bool = True
    
    # Enhanced architecture options (for improved/hybrid)
    use_attention: bool = True
    use_residual_connections: bool = True
    transformer_layers: int = 2
    cnn_scales: list = None  # [3, 5, 7] if None
    
    # Portfolio state normalization
    use_portfolio_normalization: bool = True
    portfolio_warmup_episodes: int = 50
    portfolio_update_frequency: int = 100
    
    def __post_init__(self):
        if self.cnn_scales is None:
            self.cnn_scales = [3, 5, 7]


class FinancialTransformerBlock(nn.Module):
    """Transformer block optimized for financial time series"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feed forward network with financial-aware design
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class ImprovedDuelingNetwork(nn.Module):
    """Enhanced Dueling DQN with Transformer and financial-specific improvements"""
    
    def __init__(self, config: TradingConfig):
        super(ImprovedDuelingNetwork, self).__init__()
        self.config = config
        
        # Feature embedding for stock data
        self.feature_embedding = nn.Linear(config.num_stock_features, 128)
        
        # Positional encoding for time awareness
        self.pos_encoding = nn.Parameter(torch.randn(config.window_size, 128) * 0.02)
        
        # Multi-scale CNN branch (parallel processing at different scales)
        self.multiscale_cnn = nn.ModuleList([
            self._create_cnn_branch(128, [3, 5, 7][i], f'scale_{i}') 
            for i in range(3)
        ])
        
        # Transformer blocks for temporal modeling
        self.transformer_blocks = nn.ModuleList([
            FinancialTransformerBlock(128, nhead=8, dropout=0.1)
            for _ in range(2)
        ])
        
        # Attention pooling instead of max/average pooling
        self.attention_pool = nn.MultiheadAttention(128, 4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 128))
        
        # Portfolio branch
        self.portfolio_branch = nn.Sequential(
            nn.Linear(config.num_portfolio_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # Combined features: multiscale (128*3) + transformer (128) + portfolio (64) = 576
        combined_size = 128 * 3 + 128 + 64
        
        # Shared layers with residual connections
        self.shared_layers = nn.ModuleList([
            nn.Linear(combined_size, config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size)
        ])
        
        self.shared_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(3)
        ])
        
        # Value stream with improved architecture
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1)
        )
        
        # Advantage stream with improved architecture
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, config.num_actions)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_cnn_branch(self, in_channels, kernel_size, name):
        """Create a single-scale CNN branch"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
    
    def _initialize_weights(self):
        """Improved weight initialization for financial networks"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module in [self.value_stream[-1], self.advantage_stream[-1]]:
                    # Small initialization for output layers
                    nn.init.uniform_(module.weight, -3e-4, 3e-4)
                    nn.init.constant_(module.bias, 0)
                else:
                    # He initialization for hidden layers
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Split stock data and portfolio data
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, 0, self.config.num_stock_features:]
        
        # ========== Stock Data Processing ==========
        
        # Feature embedding
        embedded = self.feature_embedding(stock_data)  # (batch_size, window_size, 128)
        
        # Add positional encoding for time awareness
        embedded = embedded + self.pos_encoding.unsqueeze(0)
        
        # Multi-scale CNN processing
        multiscale_features = []
        stock_data_cnn = embedded.transpose(1, 2)  # (batch_size, 128, window_size)
        
        for cnn_branch in self.multiscale_cnn:
            features = cnn_branch(stock_data_cnn)  # (batch_size, 128, 1)
            features = features.squeeze(-1)  # (batch_size, 128)
            multiscale_features.append(features)
        
        # Transformer processing for temporal dependencies
        transformer_out = embedded
        for transformer_block in self.transformer_blocks:
            transformer_out = transformer_block(transformer_out)
        
        # Attention pooling for transformer features
        query = self.pool_query.expand(batch_size, -1, -1)  # (batch_size, 1, 128)
        pooled_features, _ = self.attention_pool(query, transformer_out, transformer_out)
        pooled_features = pooled_features.squeeze(1)  # (batch_size, 128)
        
        # ========== Portfolio Data Processing ==========
        portfolio_features = self.portfolio_branch(portfolio_data)
        
        # ========== Feature Combination ==========
        combined_features = torch.cat([
            *multiscale_features,  # 3 x 128 = 384
            pooled_features,       # 128
            portfolio_features     # 64
        ], dim=1)  # (batch_size, 576)
        
        # ========== Shared Processing with Residuals ==========
        x = combined_features
        for i, (linear, norm) in enumerate(zip(self.shared_layers, self.shared_norms)):
            if i == 0:
                # First layer (no residual)
                x = F.gelu(norm(linear(x)))
            else:
                # Subsequent layers with residual connections
                residual = x
                x = linear(x)
                if x.size() == residual.size():  # Only add residual if dimensions match
                    x = x + residual
                x = F.gelu(norm(x))
        
        # ========== Dueling Streams ==========
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Dueling combination with improved numerical stability
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values


class DuelingNetwork(ImprovedDuelingNetwork):
    """Use improved architecture by default"""
    pass


class DuelingNetworkOriginal(nn.Module):
    """Original Dueling DQN architecture"""
    
    def __init__(self, config: TradingConfig):
        super(DuelingNetworkOriginal, self).__init__()
        self.config = config
        
        # Stock data branch - 1D CNN for time series processing
        # Input: (batch_size, window_size, self.config.num_stock_features)
        self.stock_data_branch = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_stock_features, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1), # prevents overfitting and curse of dimensionality
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Portfolio state branch - Fully connected layers
        # Input: (batch_size, self.config.num_portfolio_features) - portfolio features
        self.portfolio_branch = nn.Sequential(
            nn.Linear(self.config.num_portfolio_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # Combined features: 256 (stock data branch output) + 64 (portfolio branch output) = 320
        combined_size = 256 + 64
        
        # Shared layers after combining branches
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.num_actions)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization for ReLU networks"""
        # Initialize stock data branch
        for layer in self.stock_data_branch:
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
        
        # Initialize portfolio branch
        for layer in self.portfolio_branch:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
        
        # Initialize shared layers
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
        
        # Initialize value stream
        for layer in self.value_stream:
            if isinstance(layer, nn.Linear):
                if layer == self.value_stream[-1]:
                    nn.init.uniform_(layer.weight, -3e-4, 3e-4)
                    nn.init.constant_(layer.bias, 0)
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.01)
        
        # Initialize advantage stream
        for layer in self.advantage_stream:
            if isinstance(layer, nn.Linear):
                if layer == self.advantage_stream[-1]:
                    nn.init.uniform_(layer.weight, -3e-4, 3e-4)
                    nn.init.constant_(layer.bias, 0)
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining value and advantage streams"""        
        # Split stock data and portfolio data
        stock_data = x[:, :, :self.config.num_stock_features]  # (batch_size, window_size, self.config.num_stock_features)
        portfolio_data = x[:, 0, self.config.num_stock_features:]  # (batch_size, self.config.num_portfolio_features)
        
        # Process stock data through CNN
        # Conv1d expects (batch_size, channels, window_size)
        stock_data = stock_data.permute(0, 2, 1)  # (batch_size, 40, window_size)
        stock_data_features = self.stock_data_branch(stock_data)  # (batch_size, 256, 1)
        stock_data_features = stock_data_features.squeeze(-1)  # (batch_size, 256)
        
        # Process portfolio data through FC layers
        portfolio_features = self.portfolio_branch(portfolio_data)  # (batch_size, 64)
        
        # Combine features
        combined_features = torch.cat([stock_data_features, portfolio_features], dim=1)  # (batch_size, 320)
        
        # Process through shared layers
        shared_features = self.shared_layers(combined_features)  # (batch_size, hidden_size)
        
        # Compute value and advantages
        value = self.value_stream(shared_features)
        advantages = self.advantage_stream(shared_features)
        
        # Combine using dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values


class HybridCNNLSTMNetwork(nn.Module):
    """Hybrid CNN-LSTM network combining convolutional and recurrent processing"""
    
    def __init__(self, config: TradingConfig):
        super(HybridCNNLSTMNetwork, self).__init__()
        self.config = config
        
        # Multi-scale CNN for local pattern extraction
        self.cnn_branches = nn.ModuleList([
            self._create_cnn_branch(config.num_stock_features, kernel_size)
            for kernel_size in config.cnn_scales
        ])
        
        # Combine CNN outputs
        cnn_output_size = len(config.cnn_scales) * 64
        self.cnn_combiner = nn.Sequential(
            nn.Linear(cnn_output_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Attention mechanism for LSTM outputs
        self.lstm_attention = nn.MultiheadAttention(256, 8, batch_first=True)
        
        # Portfolio branch
        self.portfolio_branch = nn.Sequential(
            nn.Linear(config.num_portfolio_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        
        # Combined processing
        combined_size = 256 + 64  # LSTM output + portfolio
        
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )
        
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.num_actions)
        )
        
        self._initialize_weights()
    
    def _create_cnn_branch(self, in_channels, kernel_size):
        """Create CNN branch for specific kernel size"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1)
        )
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                if hasattr(module, 'weight'):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Split stock and portfolio data
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, 0, self.config.num_stock_features:]
        
        # CNN processing for each timestep
        cnn_outputs = []
        for t in range(seq_len):
            timestep_data = stock_data[:, t, :].unsqueeze(2)  # (batch, features, 1)
            
            # Multi-scale CNN
            scale_features = []
            for cnn_branch in self.cnn_branches:
                features = cnn_branch(timestep_data)  # (batch, 64, 1) - timestep_data is already (batch, features, 1)
                features = features.squeeze(-1)  # (batch, 64)
                scale_features.append(features)
            
            # Combine scales
            combined = torch.cat(scale_features, dim=1)  # (batch, 64 * num_scales)
            combined = self.cnn_combiner(combined)  # (batch, 128)
            cnn_outputs.append(combined)
        
        # Stack CNN outputs for LSTM
        cnn_sequence = torch.stack(cnn_outputs, dim=1)  # (batch, seq_len, 128)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_sequence)  # (batch, seq_len, 256)
        
        # Attention pooling
        pooled_features, _ = self.lstm_attention(lstm_out, lstm_out, lstm_out)
        pooled_features = pooled_features.mean(dim=1)  # (batch, 256)
        
        # Portfolio processing
        portfolio_features = self.portfolio_branch(portfolio_data)  # (batch, 64)
        
        # Combine features
        combined_features = torch.cat([pooled_features, portfolio_features], dim=1)
        
        # Shared processing
        shared_features = self.shared_layers(combined_features)
        
        # Dueling streams
        value = self.value_stream(shared_features)
        advantages = self.advantage_stream(shared_features)
        
        # Dueling combination
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values


def analyze_trading_performance(results: dict) -> dict:
    """
    Analyze trading performance and provide recommendations for improvement
    
    Args:
        results: Results from train_dqn or run_standalone_backtest
        
    Returns:
        Dictionary with analysis and recommendations
    """
    multi_day_results = results['multi_day_test_results']
    individual_days = multi_day_results['individual_days']
    aggregate_stats = multi_day_results['aggregate_stats']
    
    # Calculate trading frequency analysis
    total_trades = [day['total_trades'] for day in individual_days]
    avg_trades_per_day = aggregate_stats['avg_trades']
    
    # Performance analysis
    returns = [day['total_return'] for day in individual_days]
    win_rate = aggregate_stats['win_rate']
    
    analysis = {
        'trading_frequency': {
            'avg_trades_per_day': avg_trades_per_day,
            'status': 'low' if avg_trades_per_day < 5 else 'moderate' if avg_trades_per_day < 15 else 'high',
            'recommendation': None
        },
        'performance': {
            'win_rate': win_rate,
            'avg_return': aggregate_stats['avg_return'],
            'consistency': 1.0 - (aggregate_stats['std_return'] / abs(aggregate_stats['avg_return'])) if aggregate_stats['avg_return'] != 0 else 0,
            'status': 'poor' if win_rate < 0.4 else 'fair' if win_rate < 0.6 else 'good'
        }
    }
    
    # Generate recommendations
    recommendations = []
    
    if analysis['trading_frequency']['status'] == 'low':
        recommendations.append({
            'issue': 'Low Trading Frequency',
            'description': f'Only {avg_trades_per_day:.1f} trades per day. Too conservative.',
            'solutions': [
                'Use create_aggressive_trading_config() for enhanced rewards',
                'Increase epsilon_end to maintain more exploration',
                'Reduce invalid action penalties',
                'Add stronger undertrading penalties'
            ]
        })
    
    if analysis['performance']['status'] in ['poor', 'fair']:
        recommendations.append({
            'issue': 'Suboptimal Performance',
            'description': f'Win rate: {win_rate:.1%}, Avg return: {aggregate_stats["avg_return"]:.1%}',
            'solutions': [
                'Increase trade incentive rewards',
                'Reduce loss penalties to encourage more risk-taking',
                'Implement momentum-based rewards',
                'Enhance position holding logic'
            ]
        })
    
    if len([r for r in returns if r > 0]) <= len(returns) // 2:
        recommendations.append({
            'issue': 'Poor Consistency',
            'description': 'More than half the days are losing money',
            'solutions': [
                'Increase learning rate for faster adaptation',
                'Use more frequent network updates',
                'Implement better market regime detection',
                'Add time-of-day context rewards'
            ]
        })
    
    analysis['recommendations'] = recommendations
    return analysis


def print_trading_analysis(results: dict):
    """Print a detailed analysis of trading performance with recommendations"""
    analysis = analyze_trading_performance(results)
    
    print("\n" + "="*70)
    print("🔍 TRADING PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Trading Frequency
    freq = analysis['trading_frequency']
    print(f"\n📊 Trading Frequency: {freq['avg_trades_per_day']:.1f} trades/day [{freq['status'].upper()}]")
    
    # Performance
    perf = analysis['performance']
    print(f"📈 Performance: {perf['win_rate']:.0%} win rate, {perf['avg_return']:.1%} avg return [{perf['status'].upper()}]")
    print(f"🎯 Consistency Score: {perf['consistency']:.2f}")
    
    # Recommendations
    if analysis['recommendations']:
        print(f"\n💡 IMPROVEMENT RECOMMENDATIONS:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"\n{i}. {rec['issue']}")
            print(f"   Problem: {rec['description']}")
            print("   Solutions:")
            for solution in rec['solutions']:
                print(f"   • {solution}")
    else:
        print(f"\n✅ Trading performance looks good! No major issues detected.")
    
    print("\n" + "="*70)


def create_aggressive_trading_config(base_config: TradingConfig = None) -> TradingConfig:
    """
    Create a more aggressive trading configuration to encourage higher trading frequency
    and better performance. Use this if your model is too conservative.
    
    Args:
        base_config: Base configuration to modify, or None to use default
        
    Returns:
        Enhanced TradingConfig for more active trading
    """
    if base_config is None:
        config = TradingConfig()
    else:
        # Copy the base config
        import copy
        config = copy.deepcopy(base_config)
    
    # Balanced exploration for smart active trading  
    config.epsilon_start = 1.0
    config.epsilon_end = 0.10  # Moderate final epsilon for strategic decisions
    config.epsilon_decay = 100000  # Balanced decay
    
    # Adjusted learning parameters for better exploration
    config.learning_rate = 0.0003  # Slightly higher learning rate
    config.tau = 0.01  # Faster target network updates
    
    # More frequent updates for faster learning
    config.update_frequency = max(1, config.update_frequency // 2)  # 2x more frequent updates
    
    # Adjusted buffer parameters
    config.alpha = 0.7  # Higher prioritization
    config.beta_start = 0.5  # Higher importance sampling
    
    print("🚀 Created SMART AGGRESSIVE trading configuration:")
    print(f"   • Balanced final epsilon: {config.epsilon_end}")
    print(f"   • Strategic epsilon decay: {config.epsilon_decay}")
    print(f"   • Higher learning rate: {config.learning_rate}")
    print(f"   • More frequent updates: every {config.update_frequency} steps")
    print(f"   • Technical analysis-based reward system")
    print(f"   • Smart invalid action penalties")
    print(f"   • Multi-indicator decision making (RSI, MACD, Momentum, Volume, etc.)")
    
    return config


def create_network(config: TradingConfig) -> nn.Module:
    """Factory function to create network based on config"""
    if config.architecture_type == ArchitectureType.ORIGINAL:
        return DuelingNetworkOriginal(config)
    elif config.architecture_type == ArchitectureType.IMPROVED:
        return ImprovedDuelingNetwork(config)
    elif config.architecture_type == ArchitectureType.HYBRID:
        return HybridCNNLSTMNetwork(config)
    else:
        raise ValueError(f"Unknown architecture type: {config.architecture_type}")


class PrioritizedReplayBufferGPU:
    """PER VRAM version"""
    
    def __init__(self, capacity: int, config: TradingConfig, device: torch.device=DEVICE):
        self.capacity = capacity
        self.config = config
        self.device = device
        self.position = 0
        self.size = 0

        self.states = torch.zeros((capacity, config.window_size, config.num_features), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, config.window_size, config.num_features), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)

        self.priorities = torch.ones(capacity, dtype=torch.float32, device=device) * 0.01
        self.max_priority = 1.0
        
    def push(self, 
             state: torch.Tensor, 
             action: int, 
             reward: float, 
             next_state: torch.Tensor, 
             done: bool):
        """save experience"""
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        
        # Store experience
        self.states[self.position] = state
        self.actions[self.position] = torch.tensor(action, dtype=torch.long, device=self.device)
        self.rewards[self.position] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.position] = next_state
        self.dones[self.position] = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        # Set priority to max for new experiences
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int, beta: float) -> Tuple[torch.Tensor, ...]:
        """Sample batch of prioritizedexperiences"""
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        
        # Ensure priorities are positive
        priorities = torch.clamp(priorities, min=self.config.per_epsilon)
        
        # Calculate probabilities
        probs = priorities ** self.config.alpha
        probs_sum = probs.sum()
        
        # Handle edge case where sum is 0 or very small or contains NaN/inf
        if probs_sum < 1e-10 or torch.isnan(probs_sum) or torch.isinf(probs_sum):
            probs = torch.ones_like(probs) / self.size
        else:
            probs = probs / probs_sum
            
        # Additional check for NaN/inf in probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones(self.size, device=self.device) / self.size
        
        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        # Ensure weights are valid
        weights = torch.where(torch.isfinite(weights), weights, torch.ones_like(weights))
        weights = torch.clamp(weights, min=0.01, max=1.0)
        
        # Gather experiences
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Update priorities based on TD errors"""
        priorities = torch.abs(td_errors) + self.config.per_epsilon
        priorities = torch.clamp(priorities, min=max(self.config.per_epsilon, 0.001), max=1e6)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())
    
    def __len__(self):
        return self.size


class TradingMode(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

class TradingEnvironment:
    """Stock trading environment"""
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 scaled_data: pd.DataFrame, 
                 config: TradingConfig, 
                 mode: TradingMode = TradingMode.TRAIN, 
                 device: torch.device=DEVICE,
                 minutes_per_day: int = None):
        self.data = data
        self.scaled_data = scaled_data
        self.config = config
        self.mode = mode
        self.device = device
        
        # Set minutes per day based on parameter or use regular market hours as default
        self.minutes_per_day = minutes_per_day if minutes_per_day is not None else MINUTES_PER_TRADING_DAY
        
        # Calculate daily episode boundaries
        self.total_days = len(data) // self.minutes_per_day
        self.episode_length = self.minutes_per_day
        self.current_day = 0
        
        # Validate data structure
        expected_total_minutes = self.total_days * self.minutes_per_day
        actual_minutes = len(data)
        unused_minutes = actual_minutes - expected_total_minutes
        
        if unused_minutes > 0:
            print(f"Warning: {unused_minutes} minutes of data will be unused due to incomplete trading days")
        
        # Ensure we have enough data for at least one complete episode with window
        min_required = self.minutes_per_day + config.window_size
        if actual_minutes < min_required:
            raise ValueError(f"Insufficient data: need at least {min_required} minutes, got {actual_minutes}")
        
        print(f"TradingEnvironment initialized:")
        print(f"  Minutes per day: {self.minutes_per_day}")
        print(f"  Total trading days: {self.total_days}")
        print(f"  Data coverage: {expected_total_minutes} / {actual_minutes} minutes")
        print(f"  Data utilization: {expected_total_minutes/actual_minutes*100:.1f}%")
        
        # Initialize tracking variables
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_portfolio_value = self.config.initial_balance
        self.position_entry_step = -1
        self.consecutive_invalid_actions = 0
        self.last_action = 0
        self.unrealized_pnl = 0.0
        
        # Portfolio state normalization
        self.portfolio_normalizer = None
        if config.use_portfolio_normalization:
            self.portfolio_normalizer = PortfolioStateNormalizer(
                warmup_episodes=config.portfolio_warmup_episodes,
                update_frequency=config.portfolio_update_frequency
            )
        
        self.reset()
        
    def reset(self, day_idx: Optional[int] = None) -> torch.Tensor:
        """Reset environment to initial state for a new trading day"""
        self.balance = self.config.initial_balance
        self.position = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.invalid_actions = 0
        
        # Reset enhanced tracking variables
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_portfolio_value = self.config.initial_balance
        self.position_entry_step = -1  # Track when position was entered
        self.consecutive_invalid_actions = 0
        self.last_action = 0  # Track last action (0=hold, 1=buy, 2=sell)
        self.unrealized_pnl = 0.0
        
        # Select which day to trade
        if day_idx is not None:
            self.current_day = day_idx
        elif self.mode == TradingMode.TRAIN:
            # Random day for training to ensure good exploration
            max_day = max(0, self.total_days - 1)
            self.current_day = np.random.randint(0, max_day + 1) if max_day >= 0 else 0
        else:
            # Sequential days for validation/testing
            self.current_day = getattr(self, 'last_day', 0)
            self.last_day = (self.current_day + 1) % max(1, self.total_days)
        
        # Set episode boundaries
        self.episode_start = self.current_day * self.minutes_per_day
        self.episode_end = min(self.episode_start + self.episode_length, len(self.data))
        
        # Ensure there is enough data for the window
        # Start the episode far enough in to have a full window
        min_start = max(self.episode_start, self.config.window_size - 1)
        
        # Ensure we don't start too late in the episode
        if min_start >= self.episode_end:
            raise ValueError(f"Episode {self.current_day} too short for window size {self.config.window_size}")
            
        self.current_step = min_start
        self.episode_steps_remaining = self.episode_end - self.current_step
        
        # Validate episode has minimum required steps
        if self.episode_steps_remaining < 10:  # Minimum reasonable episode length
            print(f"Warning: Very short episode {self.current_day}: only {self.episode_steps_remaining} steps")
        
        # Reset episode portfolio state collection for warmup
        self.episode_portfolio_states = []
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """Get current state with enhanced features for intraday trading"""
        # Get historical data
        start_idx = max(0, self.current_step - self.config.window_size + 1)
        end_idx = self.current_step + 1
        
        # If not enough historical data, pad with the earliest available data
        if start_idx < self.episode_start - self.config.window_size + 1:
            # Pad with the first available data point in the episode
            padding_needed = (self.episode_start - self.config.window_size + 1) - start_idx
            stock_data = self.scaled_data.iloc[start_idx:end_idx].values
            if padding_needed > 0:
                first_row = stock_data[0:1]  # Get first row
                padding = np.repeat(first_row, padding_needed, axis=0)
                stock_data = np.vstack([padding, stock_data])
        else:
            stock_data = self.scaled_data.iloc[start_idx:end_idx].values
        
        # Ensure there is exactly window_size rows
        if len(stock_data) < self.config.window_size:
            padding_needed = self.config.window_size - len(stock_data)
            first_row = stock_data[0:1] if len(stock_data) > 0 else self.scaled_data.iloc[0:1].values
            padding = np.repeat(first_row, padding_needed, axis=0)
            stock_data = np.vstack([padding, stock_data])
        elif len(stock_data) > self.config.window_size:
            stock_data = stock_data[-self.config.window_size:]
        
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.balance + (self.position * current_price)
        
        # Update unrealized P&L if holding position
        if self.position > 0:
            cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
            market_value = self.position * current_price
            self.unrealized_pnl = (market_value - cost_basis) / cost_basis
        else:
            self.unrealized_pnl = 0.0
        
        # Update max portfolio value for drawdown calculation
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        
        # Calculate intraday time features (regular market hours: 9:30 AM - 4:00 PM = 390 minutes)
        minutes_into_day = (self.current_step - self.episode_start) % self.minutes_per_day
        time_of_day_normalized = minutes_into_day / self.minutes_per_day  # 0 to 1
        
        # Market session features
        morning_session = 1.0 if minutes_into_day < 120 else 0.0  # (9:30-11:30)
        midday_session = 1.0 if 120 <= minutes_into_day < 270 else 0.0  # (11:30-2:00)
        afternoon_session = 1.0 if minutes_into_day >= 270 else 0.0  # (2:00-4:00)
        
        # Position timing features
        position_holding_time = (self.current_step - self.position_entry_step) if self.position_entry_step >= 0 else 0
        normalized_holding_time = min(position_holding_time / 60, 1.0)  # Normalize to 1 hour max
        
        # Enhanced portfolio features for intraday trading
        initial_balance = self.config.initial_balance
        normalized_balance = self.balance / initial_balance if initial_balance > 0 else 0
        normalized_position = self.position * current_price / initial_balance if initial_balance > 0 else 0
        normalized_portfolio_value = portfolio_value / initial_balance if initial_balance > 0 else 0
        
        # Position ratio and risk metrics
        position_ratio = self.position * current_price / portfolio_value if portfolio_value > 0 else 0
        
        # Action validity flags (match execution logic exactly)
        position_value = self.balance * self.config.max_position_size
        shares_to_buy = int(position_value / current_price) if current_price > 0 else 0
        total_cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)

        can_buy = 1.0 if (self.position == 0 and 
                          shares_to_buy > 0 and 
                          total_cost <= self.balance) else 0.0
        can_sell = 1.0 if self.position > 0 else 0.0
        
        # Portfolio features
        portfolio_features = [
            # Core current metrics (4)
            normalized_balance,           # Current cash available
            normalized_position,          # Current stock holdings
            normalized_portfolio_value,   # Current total value
            position_ratio,               # Current position size ratio
            
            # Current position status (2)
            self.unrealized_pnl,          # Current position P&L
            normalized_holding_time,      # How long holding current position
            
            # Current timing context (4)
            time_of_day_normalized,       # Where in trading day
            morning_session,              # Current market session
            midday_session,
            afternoon_session,
            
            # Current action validity (2)
            can_buy,                      # Can execute buy now
            can_sell                      # Can execute sell now
        ]
        
        # Replace any non-finite values with 0
        portfolio_features = [x if np.isfinite(x) else 0.0 for x in portfolio_features]
        
        # Convert to numpy array
        portfolio_state = np.array(portfolio_features, dtype=np.float32)
        
        # Collect state for warmup if normalizer exists and is in warmup phase
        if (self.portfolio_normalizer is not None and 
            not self.portfolio_normalizer.is_fitted):
            self.episode_portfolio_states.append(portfolio_state.copy())
        
        # Apply normalization if fitted
        if (self.portfolio_normalizer is not None and 
            self.portfolio_normalizer.is_fitted):
            portfolio_state = self.portfolio_normalizer.normalize_state(portfolio_state)
        
        # Convert to tensor
        portfolio_state = torch.tensor(portfolio_state, dtype=torch.float32, device=self.device)
        
        # Convert stock data to tensor
        stock_data_state = torch.tensor(stock_data.astype(np.float32), dtype=torch.float32, device=self.device)
        
        # Repeat portfolio state for each timestep and concatenate
        # This is needed for compatibility with the current state representation
        # The network will extract portfolio features from the first timestep
        portfolio_state_repeated = portfolio_state.unsqueeze(0).repeat(self.config.window_size, 1)
        
        # Concatenate stock data and portfolio features
        combined_state = torch.cat([stock_data_state, portfolio_state_repeated], dim=1)
        
        return combined_state
    
    def _is_invalid_action(self, action: int) -> bool:
        """Check if action is invalid"""
        current_price = self.data.iloc[self.current_step]['close']
        if action == 1:
            if self.position > 0 or self.balance <= 0:
                return True
            else:
                position_value = self.balance * self.config.max_position_size
                shares_to_buy = position_value / current_price
                cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)
                
                return cost > self.balance
        elif action == 2:
            return self.position == 0
        return False
    
    def _get_technical_features(self) -> dict:
        """Extract technical features for intelligent reward shaping"""
        try:
            current_row = self.data.iloc[self.current_step]
            
            # Momentum indicators
            momentum_1m = current_row.get('momentum_1m', 0)
            momentum_5m = current_row.get('momentum_5m', 0)
            momentum_15m = current_row.get('momentum_15m', 0)
            
            # Technical analysis indicators
            rsi_7m = current_row.get('rsi_7m', 50)  # Default to neutral
            rsi_14m = current_row.get('rsi_14m', 50)
            bb_percent_b = current_row.get('bb_percent_b', 0.5)  # Bollinger Band position
            macd = current_row.get('macd', 0)
            macd_signal = current_row.get('macd_signal', 0)
            
            # Volatility indicators  
            atr_5m = current_row.get('atr_5m', 0)
            volatility_5m = current_row.get('volatility_5m', 0)
            
            # Volume indicators
            volume_ratio_5m = current_row.get('volume_ratio_5m', 1.0)
            volume_spike_persistence = current_row.get('volume_spike_persistence', 0)
            
            # Price position indicators
            price_to_ema_5 = current_row.get('price_to_ema_5', 1.0)
            price_to_ema_15 = current_row.get('price_to_ema_15', 1.0)
            
            # Derived signals
            momentum_aligned = (momentum_1m > 0 and momentum_5m > 0 and momentum_15m > 0)
            momentum_conflicted = (momentum_1m * momentum_5m < 0) or (momentum_5m * momentum_15m < 0)
            
            rsi_oversold = rsi_7m < 30 or rsi_14m < 30
            rsi_overbought = rsi_7m > 70 or rsi_14m > 70
            rsi_neutral = 40 <= rsi_7m <= 60 and 40 <= rsi_14m <= 60
            
            macd_bullish = macd > macd_signal and macd > 0
            macd_bearish = macd < macd_signal and macd < 0
            
            high_volume = volume_ratio_5m > 1.5 or volume_spike_persistence > 0
            low_volume = volume_ratio_5m < 0.7
            
            price_above_emas = price_to_ema_5 > 1.0 and price_to_ema_15 > 1.0
            price_below_emas = price_to_ema_5 < 1.0 and price_to_ema_15 < 1.0
            
            bb_upper_range = bb_percent_b > 0.8  # Near upper Bollinger Band
            bb_lower_range = bb_percent_b < 0.2  # Near lower Bollinger Band
            bb_middle_range = 0.3 <= bb_percent_b <= 0.7  # Middle range
            
            return {
                # Raw indicators
                'momentum_1m': momentum_1m,
                'momentum_5m': momentum_5m, 
                'momentum_15m': momentum_15m,
                'rsi_7m': rsi_7m,
                'rsi_14m': rsi_14m,
                'bb_percent_b': bb_percent_b,
                'macd': macd,
                'macd_signal': macd_signal,
                'atr_5m': atr_5m,
                'volatility_5m': volatility_5m,
                'volume_ratio_5m': volume_ratio_5m,
                'volume_spike_persistence': volume_spike_persistence,
                'price_to_ema_5': price_to_ema_5,
                'price_to_ema_15': price_to_ema_15,
                
                # Derived trading signals
                'momentum_aligned': momentum_aligned,
                'momentum_conflicted': momentum_conflicted,
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought,
                'rsi_neutral': rsi_neutral,
                'macd_bullish': macd_bullish,
                'macd_bearish': macd_bearish,
                'high_volume': high_volume,
                'low_volume': low_volume,
                'price_above_emas': price_above_emas,
                'price_below_emas': price_below_emas,
                'bb_upper_range': bb_upper_range,
                'bb_lower_range': bb_lower_range,
                'bb_middle_range': bb_middle_range,
            }
            
        except Exception as e:
            # Fallback to neutral values if feature extraction fails
            return {
                'momentum_aligned': False,
                'momentum_conflicted': False,
                'rsi_oversold': False,
                'rsi_overbought': False,
                'rsi_neutral': True,
                'macd_bullish': False,
                'macd_bearish': False,
                'high_volume': False,
                'low_volume': False,
                'price_above_emas': False,
                'price_below_emas': False,
                'bb_upper_range': False,
                'bb_lower_range': False,
                'bb_middle_range': True,
            }
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
            
        current_price = self.data.iloc[self.current_step]['close']        
        reward = 0
        trade_executed = False
        invalid_action = self._is_invalid_action(action)
        
        # Calculate market session for context
        minutes_into_day = (self.current_step - self.episode_start) % self.minutes_per_day
        is_near_close = minutes_into_day >= (self.minutes_per_day - 30)  # Last 30 minutes
        
        # Get technical features
        tech_features = self._get_technical_features()
        
        if invalid_action:
            self.invalid_actions += 1
            self.consecutive_invalid_actions += 1
            # Slightly stronger penalty to reduce invalid actions
            penalty = 0.012 + (self.consecutive_invalid_actions * 0.006)
            reward = -min(penalty, 0.04)  # Moderate increase in penalty
        else:
            self.consecutive_invalid_actions = 0  # Reset on valid action
            
            if action == 1:  # Buy
                position_value = self.balance * self.config.max_position_size
                shares_to_buy = position_value / current_price
                cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)
                
                self.position = shares_to_buy
                self.balance -= cost
                self.entry_price = current_price
                self.position_entry_step = self.current_step
                trade_executed = True
                self.last_action = 1
                
                # Base reward for entering position
                reward = 0.012  # Slightly reduced base reward
                
                # Momentum Analysis
                if tech_features['momentum_aligned']:
                    reward += 0.015  # Strong bonus for aligned momentum
                elif tech_features['momentum_conflicted']:
                    reward -= 0.008  # Penalty for conflicted momentum
                
                # RSI Analysis (buy oversold, avoid overbought)
                if tech_features['rsi_oversold']:
                    reward += 0.010  # Good entry on oversold
                elif tech_features['rsi_overbought']:
                    reward -= 0.012  # Avoid buying overbought
                elif tech_features['rsi_neutral']:
                    reward += 0.003  # Small bonus for neutral RSI
                
                # MACD Analysis
                if tech_features['macd_bullish']:
                    reward += 0.008  # MACD bullish signal
                elif tech_features['macd_bearish']:
                    reward -= 0.006  # Avoid buying on bearish MACD
                
                # Volume Analysis (confirm moves with volume)
                if tech_features['high_volume']:
                    reward += 0.005  # Volume confirms move
                elif tech_features['low_volume']:
                    reward -= 0.003  # Low volume = weak signal
                
                # Price Position Analysis
                if tech_features['price_above_emas']:
                    reward += 0.004  # Trend following
                elif tech_features['price_below_emas']:
                    reward -= 0.002  # Against trend
                
                # Bollinger Band Analysis
                if tech_features['bb_lower_range']:
                    reward += 0.006  # Buy near lower band (oversold)
                elif tech_features['bb_upper_range']:
                    reward -= 0.008  # Avoid buying near upper band
                elif tech_features['bb_middle_range']:
                    reward += 0.002  # Neutral zone
                
                # Market condition bonus
                if (tech_features['momentum_aligned'] and 
                    tech_features['macd_bullish'] and 
                    tech_features['high_volume']):
                    reward += 0.012  # Perfect storm bonus
                
                # Time-based entry bonus
                if not is_near_close:
                    reward += 0.004  # Time bonus
                    
                # Base activity bonus
                reward += 0.003
                    
            elif action == 2:  # Sell
                revenue = self.position * current_price * (1 - self.config.transaction_fee_percent)
                cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
                profit = revenue - cost_basis
                
                self.balance += revenue
                self.position = 0
                trade_executed = True
                self.total_trades += 1
                self.last_action = 2
                
                # Calculate holding time bonus/penalty
                holding_time = self.current_step - self.position_entry_step if self.position_entry_step >= 0 else 1
                holding_time_factor = min(holding_time / 30, 1.0)  # Normalize to 30 minutes
                
                if profit > 0:
                    self.winning_trades += 1
                    self.total_profit += profit
                    # Base reward for profitable trades
                    profit_percentage = profit / cost_basis
                    reward = profit_percentage * 25 * (0.5 + 0.5 * holding_time_factor)
                    
                    # Technical analysis exit bonuses
                    # RSI Analysis (sell overbought is good)
                    if tech_features['rsi_overbought']:
                        reward += 0.012  # Great timing - sell overbought
                    elif tech_features['rsi_oversold']:
                        reward -= 0.004  # Poor timing - selling oversold
                    
                    # MACD Analysis (sell on bearish signals)
                    if tech_features['macd_bearish']:
                        reward += 0.008  # Good timing - MACD turning bearish
                    elif tech_features['macd_bullish']:
                        reward -= 0.003  # Might be early exit
                    
                    # Momentum Analysis (sell when momentum turns)
                    if tech_features['momentum_conflicted']:
                        reward += 0.010  # Good timing - momentum turning
                    elif tech_features['momentum_aligned'] and tech_features['momentum_1m'] > 0:
                        reward -= 0.005  # Selling into strong momentum
                    
                    # Bollinger Band Analysis
                    if tech_features['bb_upper_range']:
                        reward += 0.008  # Good exit near upper band
                    elif tech_features['bb_lower_range']:
                        reward -= 0.006  # Poor exit near lower band
                    
                    # Volume confirmation
                    if tech_features['high_volume']:
                        reward += 0.004  # Volume confirms exit
                    
                    # Perfect exit timing bonus
                    if (tech_features['rsi_overbought'] and 
                        tech_features['macd_bearish'] and 
                        tech_features['bb_upper_range']):
                        reward += 0.015  # Perfect exit conditions
                    
                    # Time-based exit bonus
                    if is_near_close:
                        reward += 0.005
                    
                    # Activity bonus
                    reward += 0.005
                    
                else:
                    self.losing_trades += 1
                    self.total_loss += abs(profit)
                    loss_percentage = abs(profit) / cost_basis
                    stop_loss_factor = 1.0 - holding_time_factor
                    reward = -loss_percentage * 10 * (0.3 + 0.7 * stop_loss_factor)
                    
                    # Technical analysis for loss mitigation
                    # Reward good stop-loss decisions based on technicals
                    if tech_features['momentum_conflicted'] or tech_features['macd_bearish']:
                        reward += 0.008  # Good stop-loss on technical breakdown
                    
                    if tech_features['rsi_oversold'] and loss_percentage < 0.02:
                        reward += 0.005  # Cut losses near oversold (might bounce)
                    
                    # Quick stop-loss rewards
                    if holding_time <= 10 and loss_percentage < 0.01:
                        reward += 0.006  # Quick small loss - good risk management
                    elif holding_time <= 5:
                        reward += 0.003
                    
                    # Small action bonus
                    reward += 0.002
                
                self.position_entry_step = -1
                
            else:  # Hold (action == 0)
                self.last_action = 0
                
                if self.position > 0:
                    # Base P&L holding logic
                    unrealized_pnl = ((current_price - self.entry_price) / self.entry_price)
                    
                    # Technical analysis for holding decisions
                    if unrealized_pnl > 0:  # Profitable position
                        # Base reward for profitable holding
                        if unrealized_pnl > 0.01:
                            reward = 0.003 * unrealized_pnl  # Hold big winners
                        else:
                            reward = 0.001 * unrealized_pnl  # Hold small winners
                        
                        # Technical confirmation for holding winners
                        if tech_features['momentum_aligned'] and tech_features['macd_bullish']:
                            reward += 0.004  # Strong technical support for holding
                        elif tech_features['rsi_overbought'] or tech_features['bb_upper_range']:
                            reward -= 0.008  # Should consider taking profits
                        elif tech_features['momentum_conflicted']:
                            reward -= 0.005  # Momentum turning, consider exit
                            
                    else:  # Losing position
                        reward = -0.002 * abs(unrealized_pnl)  # Penalty for holding losers
                        
                        # Technical analysis for holding losers
                        if tech_features['rsi_oversold'] and unrealized_pnl > -0.02:
                            reward += 0.003  # Might be oversold bounce opportunity
                        elif tech_features['momentum_conflicted'] or tech_features['macd_bearish']:
                            reward -= 0.006  # Technical breakdown, should exit
                        elif tech_features['bb_lower_range'] and unrealized_pnl > -0.01:
                            reward += 0.002  # Near support, might hold
                    
                    # Time-based holding penalties (encourage active management)
                    holding_time = self.current_step - self.position_entry_step if self.position_entry_step >= 0 else 0
                    if holding_time > 90:  # Very long hold
                        reward -= 0.008
                    elif holding_time > 60:
                        reward -= 0.005
                    elif holding_time > 30:
                        reward -= 0.002
                else:
                    # Technical analysis-based cash holding decisions
                    minutes_since_start = self.current_step - self.episode_start
                    
                    # Base inactivity penalties (but consider technical conditions)
                    base_penalty = 0
                    
                    if minutes_since_start > 15 and self.total_trades == 0:
                        # Escalating penalty for no trading at all
                        no_trade_penalty = 0.008 + (minutes_since_start - 15) * 0.0003
                        base_penalty = -min(no_trade_penalty, 0.04)
                    elif minutes_since_start > 30 and self.total_trades > 0:
                        # Penalty for too much time between trades
                        avg_time_per_trade = minutes_since_start / max(self.total_trades, 1)
                        if avg_time_per_trade > 25:
                            base_penalty = -0.006
                        elif avg_time_per_trade > 40:
                            base_penalty = -0.012
                    
                    # Sometimes cash is smart (reduce penalties)
                    if (tech_features['rsi_overbought'] and 
                        tech_features['bb_upper_range'] and 
                        tech_features['momentum_conflicted']):
                        base_penalty *= 0.5  # Reduce penalty - market might be topping
                        reward += 0.002  # Small bonus for avoiding overbought market
                    
                    elif (tech_features['momentum_aligned'] and 
                          tech_features['macd_bullish'] and 
                          tech_features['rsi_neutral']):
                        base_penalty *= 1.5  # Increase penalty - missing good setup
                        reward -= 0.003  # Penalty for missing bullish setup
                    
                    elif tech_features['high_volume'] and tech_features['momentum_aligned']:
                        base_penalty *= 1.3  # Increase penalty - missing volume breakout
                        reward -= 0.002
                    
                    # Apply the modified penalty
                    reward += base_penalty
                    
                    # Additional long-term inactivity penalty
                    if minutes_since_start > 60:
                        reward -= 0.004
                    
                    # Context-based rewards with technical confirmation
                    if self.current_step > self.episode_start + 5:
                        recent_return = (current_price - self.data.iloc[self.current_step - 5]['close']) / self.data.iloc[self.current_step - 5]['close']
                        
                        if recent_return < -0.005:  # Price dropped
                            if tech_features['rsi_oversold']:
                                reward += 0.001  # Good timing - avoid falling market
                            else:
                                reward += 0.0003  # Small reward for avoiding loss
                        elif recent_return > 0.005:  # Price rose
                            if tech_features['momentum_aligned']:
                                reward -= 0.002  # Penalty for missing strong move
                            else:
                                reward -= 0.0005  # Small penalty for missing opportunity
            
            # Portfolio-level rewards
            current_portfolio_value = self.balance + (self.position * current_price)
            portfolio_return = (current_portfolio_value - self.config.initial_balance) / self.config.initial_balance
            
            # Reward for maintaining/growing portfolio value
            if portfolio_return > 0:
                reward += 0.0002 * portfolio_return
            
            # Win Rate Bonus (encourage consistent profitability)
            if self.total_trades > 0:
                current_win_rate = self.winning_trades / self.total_trades
                if current_win_rate >= 0.6:  # High win rate bonus
                    reward += 0.002
                elif current_win_rate >= 0.5:  # Decent win rate
                    reward += 0.001
                elif current_win_rate < 0.3:  # Poor win rate penalty
                    reward -= 0.001
            
            # Profit/Loss Ratio Rewards (encourage good risk management)
            if self.winning_trades > 0 and self.losing_trades > 0:
                avg_profit = self.total_profit / self.winning_trades
                avg_loss = abs(self.total_loss) / self.losing_trades
                profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 1.0
                
                if profit_loss_ratio >= 2.0:  # Excellent risk/reward
                    reward += 0.003
                elif profit_loss_ratio >= 1.5:  # Good risk/reward
                    reward += 0.002
                elif profit_loss_ratio < 0.8:  # Poor risk/reward
                    reward -= 0.002
            
            # Drawdown Management (penalize excessive portfolio decline)
            drawdown = (self.max_portfolio_value - current_portfolio_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0
            if drawdown > 0.1:  # More than 10% drawdown
                reward -= 0.003 * drawdown  # Escalating penalty
            elif drawdown > 0.05:  # More than 5% drawdown
                reward -= 0.001 * drawdown
            
            # Smart Trade Frequency Management (encourage quality activity)
            minutes_elapsed = max(1, self.current_step - self.episode_start)
            trade_rate = self.total_trades / (minutes_elapsed / 60)  # Trades per hour
            
            # Balanced incentives for smart trading
            if 0.4 <= trade_rate <= 2.5:  # Optimal trading frequency (24-150 trades per day)
                reward += 0.010  # Large bonus for good activity level
            elif 0.15 <= trade_rate < 0.4:  # Moderate activity
                reward += 0.005  # Encourage more activity
            elif 0.05 <= trade_rate < 0.15:  # Low activity
                reward += 0.002  # Small encouragement
            elif trade_rate > 3.5:  # Overtrading penalty 
                reward -= 0.003 * (trade_rate - 3.5)
            elif trade_rate < 0.05 and minutes_elapsed > 25:  # Very low activity
                undertrading_penalty = 0.012 * (0.05 - trade_rate)
                reward -= undertrading_penalty
            elif trade_rate < 0.02 and minutes_elapsed > 20:  # Almost no activity
                reward -= 0.015
            
            # Risk management rewards
            # Reward for keeping reasonable position sizes
            if self.position > 0:
                position_pct = (self.position * current_price) / current_portfolio_value
                if 0.3 <= position_pct <= 0.8:  # Reasonable position size
                    reward += 0.0001
                elif position_pct > 0.9:  # Too concentrated
                    reward -= 0.002
            
            # Penalty for low cash reserves (risk management)
            cash_ratio = self.balance / current_portfolio_value if current_portfolio_value > 0 else 0
            if cash_ratio < 0.1 and self.position > 0:  # Less than 10% cash when holding position
                reward -= 0.001
        
        # Move to next step
        self.current_step += 1
        self.episode_steps_remaining -= 1
        
        # Check if episode is done (end of trading day or out of balance)
        done = (self.current_step >= self.episode_end or 
                self.episode_steps_remaining <= 0 or 
                self.balance <= 0)
        
        # Force close any open positions at end of day (realistic intraday trading)
        if done and self.position > 0:
            # Close position at current price
            revenue = self.position * current_price * (1 - self.config.transaction_fee_percent)
            cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
            profit = revenue - cost_basis
            
            self.balance += revenue
            self.position = 0
            self.total_trades += 1
            
            if profit > 0:
                self.winning_trades += 1
                self.total_profit += profit
                # Reward for profitable end-of-day close
                profit_percentage = profit / cost_basis
                reward += profit_percentage * 10
            else:
                self.losing_trades += 1
                self.total_loss += abs(profit)
                # Smaller penalty for end-of-day close (forced exit)
                loss_percentage = abs(profit) / cost_basis
                reward -= loss_percentage * 5
        
        # Get next state (or terminal state if done)
        if done:
            # Return current state as next state when episode is done
            self.current_step -= 1
            next_state = self._get_state()
            self.current_step += 1
        else:
            # For non-terminal transitions, use the state at the new current_step
            next_state = self._get_state()
        
        # Additional info
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price,
            'trade_executed': trade_executed,
            'invalid_action': invalid_action,
            'invalid_actions': self.invalid_actions,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'current_day': self.current_day,
            'episode_steps_remaining': self.episode_steps_remaining,
            'unrealized_pnl': getattr(self, 'unrealized_pnl', 0.0),
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'final_reward': reward  # Track the final reward for analysis
        }
        
        # Handle portfolio normalizer episode completion
        if done and self.portfolio_normalizer is not None:
            # Collect episode data for warmup
            if not self.portfolio_normalizer.is_fitted:
                self.portfolio_normalizer.collect_warmup_data(self.episode_portfolio_states)
            
            # Increment episode counter and potentially fit normalizer
            self.portfolio_normalizer.increment_episode()
        
        return next_state, reward, done, info


class DoubleDuelingDQN:
    """Double Dueling DQN Agent with PER and configurable architectures"""
    
    def __init__(self, config: TradingConfig, device: torch.device=DEVICE):
        self.config = config
        self.device = device
        print(f"Using device: {device}")
        print(f"Using architecture: {config.architecture_type.value}")
        
        # Networks using factory function
        self.q_network = create_network(config).to(device)
        self.target_network = create_network(config).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Print network size for debugging
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        print(f"Network parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        self.optimizer = optim.AdamW(self.q_network.parameters(), 
                                    lr=config.learning_rate, 
                                    weight_decay=1e-5)  # Fixed weight_decay
        
        # Learning rate scheduler - reduces LR when validation reward plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
        
        # Replay buffer
        self.memory = PrioritizedReplayBufferGPU(config.buffer_size, config, device)
        
        # Training tracking
        self.steps_done = 0
        self.episodes_done = 0
        self.update_count = 0
        
        
    def select_action(self, state: torch.Tensor, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * math.exp(-1. * self.steps_done / self.config.epsilon_decay)
        
        if random.random() > epsilon:
            self.q_network.eval()  # Set to evaluation mode for deterministic inference
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                action = torch.argmax(q_values, dim=1).item()
            self.q_network.train()  # Set back to training mode
            return action
        else:
            return random.randrange(self.config.num_actions)
    
    def update(self) -> Dict[str, float]:
        """Perform one update step"""
        if len(self.memory) < self.config.batch_size:
            return {}
        
        # Skip training during portfolio normalization warmup period
        if hasattr(self, '_env_ref') and self._env_ref is not None:
            if (self._env_ref.portfolio_normalizer is not None and 
                not self._env_ref.portfolio_normalizer.is_fitted):
                return {'skipped': True, 'reason': 'portfolio_warmup'}
        
        try:
            # Calculate current beta for importance sampling
            beta = self.config.beta_start + (self.config.beta_end - self.config.beta_start) * min(1.0, self.steps_done / 100000)  # Fixed beta decay steps
            
            # Sample batch
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.config.batch_size, beta)
            
            # Compute current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Double DQN: use online network to select actions, target network to evaluate
            with torch.no_grad():
                if self.config.use_double_dqn:
                    next_actions = self.q_network(next_states).max(1)[1]
                    next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
                else:
                    next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
                    
                target_q_values = rewards.unsqueeze(1) + (self.config.gamma * next_q_values * ~dones.unsqueeze(1))
            
            # Compute TD errors for priority updates
            td_errors = (current_q_values - target_q_values).squeeze()
            
            # Clamp TD errors to prevent extreme values
            td_errors = torch.clamp(td_errors, min=-10.0, max=10.0)
            
            # Replace any NaN/inf values with zero
            td_errors = torch.where(torch.isfinite(td_errors), td_errors, torch.zeros_like(td_errors))
            
            # Update priorities in replay buffer
            self.memory.update_priorities(indices, td_errors.detach())
            
            # Compute weighted loss
            loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none').squeeze()).mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            # Soft update of target network using tau
            if self.config.tau > 0:
                for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_param.data.copy_(self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data)
            else:
                # Hard update every target_update_frequency steps
                self.update_count += 1
                if self.update_count % self.config.target_update_frequency == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
            
            return {
                'loss': loss.item(),
                'mean_q': current_q_values.mean().item(),
                'mean_td_error': td_errors.abs().mean().item()
            }
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA Error in update: {e}")
                print(f"Memory size: {len(self.memory)}")
                print(f"Steps done: {self.steps_done}")
                # Try to recover by clearing CUDA cache
                torch.cuda.empty_cache()
                return {}
            else:
                raise
    
    def train_episode(self, env: TradingEnvironment) -> Dict[str, float]:
        """Train for one episode"""
        # Set environment reference for warmup checking
        self._env_ref = env
        
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Track update metrics across the episode
        update_metrics = {
            'loss': [],
            'mean_q': [],
            'mean_td_error': []
        }
        
        while True:
            # Select and execute action
            action = self.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            self.memory.push(state, action, reward, next_state, done)
            
            # Update counters
            episode_reward += reward
            episode_steps += 1
            self.steps_done += 1
            
            # Perform update every update_frequency steps
            if self.steps_done % self.config.update_frequency == 0:
                update_info = self.update()
                # Track metrics if update occurred (and wasn't skipped)
                if update_info and 'skipped' not in update_info:
                    for key in ['loss', 'mean_q', 'mean_td_error']:
                        if key in update_info:
                            update_metrics[key].append(update_info[key])
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        self.episodes_done += 1
        
        # Episode statistics
        final_value = info['balance'] + (info['position'] * info['current_price'])
            
        # Calculate return
        total_return = (final_value - self.config.initial_balance) / self.config.initial_balance
        
        # Average the update metrics over the episode
        avg_update_metrics = {}
        for key, values in update_metrics.items():
            if values:
                avg_update_metrics[key] = sum(values) / len(values)
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': info['total_trades'],
            'winning_trades': info['winning_trades'],
            'losing_trades': info['losing_trades'],
            'invalid_actions': info['invalid_actions'],
            **avg_update_metrics
        }
    
    def save(self, path: str, portfolio_normalizer=None):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'update_count': self.update_count
        }, path)
        
        # Save portfolio normalizer separately if provided
        if portfolio_normalizer is not None:
            normalizer_path = path.replace('.pt', '_normalizer.pkl')
            portfolio_normalizer.save(normalizer_path)
    
    def load(self, path: str, portfolio_normalizer=None):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Load scheduler state if available (for backward compatibility)
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        self.update_count = checkpoint['update_count']
        
        # Load portfolio normalizer if provided
        if portfolio_normalizer is not None:
            normalizer_path = path.replace('.pt', '_normalizer.pkl')
            portfolio_normalizer.load(normalizer_path)


def filter_to_regular_hours(df):
    """Filter dataframe to regular market hours using UTC timestamps
    
    Regular market hours: 9:30 AM - 4:00 PM EST
    In UTC: 14:30 - 21:00 (EST, winter) or 13:30 - 20:00 (EDT, summer)
    Note: Assumes weekends are already filtered out during feature engineering
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Regular market hours filtering (handles DST automatically through pandas)
    # Convert to Eastern time temporarily just for filtering
    eastern_times = df['timestamp'].dt.tz_convert('US/Eastern')
    market_open = eastern_times.dt.time >= time(9, 30)
    market_close = eastern_times.dt.time < time(16, 0)
    
    # Apply filters and keep original UTC timestamps
    filtered_df = df[market_open & market_close].reset_index(drop=True)
    
    print(f"Data filtered: {len(df)} → {len(filtered_df)} rows ({len(filtered_df)/len(df)*100:.1f}%)")
    return filtered_df

def load_stock_data(data_path: str, cutoff: pd.Timestamp | None=None, cols_to_keep: list[str]=STOCK_FEATURES_V2) -> tuple[pd.DataFrame, datetime, datetime]:
    """
    Get saved csv data and filter to regular market hours
    """
    df = pd.read_csv(data_path)

    # Apply cutoff first if specified
    if cutoff:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure cutoff and data timestamps are timezone-aware and compatible
        if cutoff.tz is not None:
            # If cutoff has timezone, convert data timestamps to same timezone
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert(cutoff.tz)
        else:
            # If cutoff is naive, ensure data timestamps are also naive
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        df = df[df['timestamp'] >= cutoff]
        
    # Filter to regular market hours
    df = filter_to_regular_hours(df)
    
    # Get date range after filtering
    start_date = pd.to_datetime(df['timestamp'].iloc[0]).to_pydatetime()
    end_date = pd.to_datetime(df['timestamp'].iloc[-1]).to_pydatetime()
        
    return df[cols_to_keep], start_date, end_date

def save_backtest_results_to_db(model_type: ModelType,
                                ticker: str,
                                info: dict[str, float],
                                preprocessor_path: Optional[str] = None) -> tuple[int, str, str]:
    backtest_date = info['backtest_date']
    return_rate = info['return_rate']
    
    with app.app_context():
        db.create_all()
        model = MarklygonModel(
            model=model_type,
            ticker=ticker
        )
        db.session.add(model)
        db.session.flush()
        
        model_id = model.id
        # Create directory for this model - use absolute path
        model_dir = str(MODELS_DIR / 'dqn_v2' / str(model_id))
        create_directory(model_dir)
        
        # Set model path within the model's directory - use absolute path
        model_path = str(Path(model_dir) / 'model.pth')
        model.model_path = model_path

        backtest = BacktestHistory(
            model=model,
            backtest_date=backtest_date,
            start_date=info['start_date'],
            end_date=info['end_date'],
            initial_balance=info['initial_balance'],
            final_balance=info['final_balance'],
            net_profit=info['net_profit'],
            total_trades=info['total_trades'],
            winning_trades=info['winning_trades'],
            losing_trades=info['losing_trades'],
            return_rate=return_rate,
            max_drawdown=abs(info['max_drawdown']),
            sharpe_ratio=info['sharpe_ratio'],
            invalid_actions=info['invalid_actions'],
            preprocessor_path=preprocessor_path
        )

        db.session.add(backtest)
        db.session.commit()
    return model_id, model_path, model_dir


def create_model_in_db(model_type: ModelType, ticker: str) -> tuple[int, str, str]:
    """
    Create a model entry in the database and return model info
    
    Returns:
        tuple: (model_id, model_path, model_dir)
    """
    with app.app_context():
        db.create_all()
        model = MarklygonModel(
            model=model_type,
            ticker=ticker
        )
        db.session.add(model)
        db.session.flush()
        
        model_id = model.id
        # Create directory for this model - use absolute path
        model_dir = str(MODELS_DIR / 'dqn_v2' / str(model_id))
        create_directory(model_dir)
        
        # Set model path within the model's directory - use absolute path
        model_path = str(Path(model_dir) / 'model.pth')
        model.model_path = model_path
        
        db.session.commit()
    
    return model_id, model_path, model_dir


def save_backtest_to_existing_model(model_id: int, 
                                   info: dict[str, float], 
                                   preprocessor_path: Optional[str] = None) -> int:
    """
    Save a backtest result to an existing model
    
    Args:
        model_id: ID of existing model
        info: Backtest information dictionary
        preprocessor_path: Optional path to preprocessor
        
    Returns:
        backtest_id: ID of created backtest entry
    """
    backtest_date = info['backtest_date']
    return_rate = info['return_rate']
    
    with app.app_context():
        # Get the existing model
        model = MarklygonModel.query.get(model_id)
        if not model:
            raise ValueError(f"Model with ID {model_id} not found")
        
        backtest = BacktestHistory(
            model=model,
            backtest_date=backtest_date,
            start_date=info['start_date'],
            end_date=info['end_date'],
            initial_balance=info['initial_balance'],
            final_balance=info['final_balance'],
            net_profit=info['net_profit'],
            total_trades=info['total_trades'],
            winning_trades=info['winning_trades'],
            losing_trades=info['losing_trades'],
            return_rate=return_rate,
            max_drawdown=abs(info['max_drawdown']),
            sharpe_ratio=info['sharpe_ratio'],
            invalid_actions=info['invalid_actions'],
            preprocessor_path=preprocessor_path
        )

        db.session.add(backtest)
        db.session.commit()
        
        return backtest.id


def save_multi_day_backtest_to_db(model_type: ModelType,
                                 ticker: str,
                                 multi_day_results: dict,
                                 start_date,
                                 end_date,
                                 initial_balance: float,
                                 preprocessor_path: Optional[str] = None) -> tuple[int, str, str, list[int]]:
    """
    Save multi-day backtest results: one model with multiple backtest entries
    
    Args:
        model_type: Type of model (e.g., ModelType.DQN)
        ticker: Stock ticker symbol
        multi_day_results: Results from multi-day testing
        start_date: Start date of testing period
        end_date: End date of testing period  
        initial_balance: Initial trading balance
        preprocessor_path: Optional path to preprocessor
        
    Returns:
        tuple: (model_id, model_path, model_dir, backtest_ids)
    """
    from datetime import datetime, timezone
    
    # Create the model first
    model_id, model_path, model_dir = create_model_in_db(model_type, ticker)
    
    individual_days = multi_day_results['individual_days']
    aggregate_stats = multi_day_results['aggregate_stats']
    
    backtest_ids = []
    
    # Save aggregate summary backtest
    aggregate_info = {
        'backtest_date': datetime.now(timezone.utc),
        'start_date': start_date,
        'end_date': end_date,
        'initial_balance': initial_balance,
        'final_balance': aggregate_stats['avg_final_value'],
        'net_profit': aggregate_stats['avg_final_value'] - initial_balance,
        'total_trades': int(aggregate_stats['avg_trades']),
        'winning_trades': int(aggregate_stats['avg_winning_trades']),
        'losing_trades': int(aggregate_stats['avg_losing_trades']),
        'return_rate': aggregate_stats['avg_return'], 
        'max_drawdown': aggregate_stats['avg_max_drawdown'],
        'sharpe_ratio': aggregate_stats['avg_sharpe_ratio'],
        'invalid_actions': int(aggregate_stats['avg_invalid_actions']),
    }
    
    aggregate_backtest_id = save_backtest_to_existing_model(model_id, aggregate_info, preprocessor_path)
    backtest_ids.append(aggregate_backtest_id)
    
    # Save individual day backtests
    for i, day_result in enumerate(individual_days, 1):
        day_info = {
            'backtest_date': datetime.now(timezone.utc),
            'start_date': start_date,
            'end_date': end_date,
            'initial_balance': initial_balance,
            'final_balance': day_result['final_value'],
            'net_profit': day_result['final_value'] - initial_balance,
            'total_trades': day_result['total_trades'],
            'winning_trades': day_result['winning_trades'],
            'losing_trades': day_result['losing_trades'],
            'return_rate': day_result['total_return'], 
            'max_drawdown': day_result['max_drawdown'],
            'sharpe_ratio': day_result['sharpe_ratio'],
            'invalid_actions': day_result['invalid_actions'],
        }
        
        day_backtest_id = save_backtest_to_existing_model(model_id, day_info)
        backtest_ids.append(day_backtest_id)
    
    return model_id, model_path, model_dir, backtest_ids

def train_dqn(data_path: str,
              cutoff: pd.Timestamp,
              num_episodes: int = NUM_EPISODES, 
              save_interval: int = 100,
              validation_frequency: int = EVALUATE_INTERVAL,
              early_stopping_patience: int = 10,
              train_ratio: float = TRAIN_RATIO,
              valid_ratio: float = VALID_RATIO,
              use_preprocessing: bool = True,
              scaling_method: str = 'robust',
              outlier_method: str = 'winsorize',
              preprocessor_save_path: Optional[str] = None,
              architecture_type: ArchitectureType = ArchitectureType.IMPROVED):
    """
    Main training function with validation and early stopping
    
    Args:
        data_path: Path to CSV file with stock data
        cutoff: Timestamp to start data from
        num_episodes: Number of training episodes
        save_interval: Save model every N episodes
        validation_frequency: Run validation every N episodes
        early_stopping_patience: Stop if validation doesn't improve for N checks
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        use_preprocessing: Whether to apply preprocessing
        scaling_method: Method for scaling features ('robust', 'standard', 'minmax', 'none')
        outlier_method: Method for handling outliers ('winsorize', 'clip', 'none')
        preprocessor_save_path: Path to save the fitted preprocessor
        architecture_type: Architecture type for the DQN ('original', 'improved', 'hybrid')
    """    
    # Load data
    print("Loading data...")
    data, start_date, end_date = load_stock_data(data_path, cutoff)
    
    # Split data chronologically
    train_end = int(len(data) * train_ratio)
    valid_end = train_end + int(len(data) * valid_ratio)
    
    train_data = data.iloc[:train_end].copy().reset_index(drop=True)
    valid_data = data.iloc[train_end:valid_end].copy().reset_index(drop=True)
    test_data = data.iloc[valid_end:].copy().reset_index(drop=True)
    
    print(f"Data split: Train {len(train_data)}, Validation {len(valid_data)}, Test {len(test_data)}")
    
    # Apply preprocessing if requested
    preprocessor = None
    if use_preprocessing:
        print(f"\nApplying preprocessing (scaling: {scaling_method}, outliers: {outlier_method})...")
        from src.models.mark.dqn_v2.data_preprocessor_v2 import preprocess_financial_data
        
        # Set default preprocessor save path if not provided
        if preprocessor_save_path is None:
            from pathlib import Path
            data_name = Path(data_path).stem
            preprocessor_save_path = f"preprocessor_{data_name}.pkl"
        
        # Preprocess data
        preprocessor, train_data_scaled, valid_data_scaled, test_data_scaled = preprocess_financial_data(
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            scaling_method=scaling_method,
            outlier_method=outlier_method,
            save_preprocessor=True,
            preprocessor_path=preprocessor_save_path
        )
        print(f"Preprocessing complete. Preprocessor saved to: {preprocessor_save_path}")
    else:
        # If no preprocessing, use raw data
        train_data_scaled = train_data
        valid_data_scaled = valid_data
        test_data_scaled = test_data
    
    # Initialize configuration
    config = TradingConfig(architecture_type=architecture_type)
    
    # Create environments
    train_env = TradingEnvironment(train_data, train_data_scaled, config, mode=TradingMode.TRAIN)
    val_env = TradingEnvironment(valid_data, valid_data_scaled, config, mode=TradingMode.VAL)
    test_env = TradingEnvironment(test_data, test_data_scaled, config, mode=TradingMode.TEST)
    
    print(f"train_data_scaled.shape: {train_data_scaled.shape}")
    print(f"valid_data_scaled.shape: {valid_data_scaled.shape}")
    print(f"test_data_scaled.shape: {test_data_scaled.shape}")
    
    # Create agent
    agent = DoubleDuelingDQN(config)
    
    # Training metrics
    episode_rewards = []
    episode_returns = []
    episode_trades = []
    episode_invalid_actions = []
    
    validation_rewards = []
    validation_returns = []
    validation_trades = []
    validation_invalid_actions = []
    
    best_validation_return = float('-inf')
    patience_counter = 0
    best_model_state = None
    
    print("Starting training...")
    
    for episode in range(num_episodes):
        # Train one episode
        metrics = agent.train_episode(train_env)
        
        # Store metrics
        episode_rewards.append(metrics['episode_reward'])
        episode_returns.append(metrics['total_return'])
        episode_trades.append(metrics['total_trades'])
        episode_invalid_actions.append(metrics['invalid_actions'])
        
        # Print progress
        current_epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * math.exp(-1. * agent.steps_done / config.epsilon_decay)
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        print(f"  Reward: {metrics['episode_reward']:.4f}")
        print(f"  Return: {metrics['total_return']:.2%}")
        print(f"  Final Value: ${metrics['final_value']:,.2f}")
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Winning Trades: {metrics['winning_trades']}")
        print(f"  Losing Trades: {metrics['losing_trades']}")
        print(f"  Invalid Actions: {metrics['invalid_actions']}")
        print(f"  Steps: {metrics['episode_steps']}")
        print(f"  Epsilon: {current_epsilon:.4f} ({'Exploring' if current_epsilon > 0.1 else 'Exploiting'})")
        
        # Log portfolio normalization status
        if train_env.portfolio_normalizer is not None:
            if train_env.portfolio_normalizer.is_fitted:
                print(f"  Portfolio Normalizer: ✅ ACTIVE (training enabled)")
            else:
                progress = train_env.portfolio_normalizer.episode_count / train_env.portfolio_normalizer.warmup_episodes
                print(f"  Portfolio Normalizer: 🔥 WARMUP ({progress*100:.1f}%) - ⚠️  TRAINING PAUSED")
        
        # Validation (skip during portfolio normalization warmup)
        if (episode + 1) % validation_frequency == 0:
            if train_env.portfolio_normalizer is None or train_env.portfolio_normalizer.is_fitted:
                print("\nRunning validation...")
            else:
                print(f"\n⚠️  Validation SKIPPED (episode {episode+1}) - Portfolio normalization warmup in progress")
        
        if ((episode + 1) % validation_frequency == 0 and 
            (train_env.portfolio_normalizer is None or train_env.portfolio_normalizer.is_fitted)):
            
            # Run validation episode
            val_state = val_env.reset()
            val_reward = 0
            val_done = False
            
            while not val_done:
                val_action = agent.select_action(val_state, epsilon=0.0)  # No exploration
                val_next_state, val_r, val_done, val_info = val_env.step(val_action)
                val_reward += val_r
                val_state = val_next_state
            
            val_final_value = val_info['balance'] + (val_info['position'] * val_info['current_price'])
            val_return = (val_final_value - config.initial_balance) / config.initial_balance
            
            validation_rewards.append(val_reward)
            validation_returns.append(val_return)
            validation_trades.append(val_info['total_trades'])
            validation_invalid_actions.append(val_info['invalid_actions'])
            
            print(f"Validation Results:")
            print(f"  Return: {val_return:.2%}")
            print(f"  Final Value: ${val_final_value:,.2f}")
            print(f"  Trades: {val_info['total_trades']}")
            print(f"  Winning Trades: {val_info['winning_trades']}")
            print(f"  Losing Trades: {val_info['losing_trades']}")
            print(f"  Invalid Actions: {val_info['invalid_actions']}")
            
            # Update learning rate scheduler based on validation return
            agent.scheduler.step(val_return)
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(f"  Current LR: {current_lr:.2e}")
            
            # Early stopping check
            if val_return > best_validation_return:
                best_validation_return = val_return
                patience_counter = 0
                # Save best model state
                best_model_state = {
                    'q_network': agent.q_network.state_dict(),
                    'target_network': agent.target_network.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'scheduler': agent.scheduler.state_dict(),
                    'steps_done': agent.steps_done,
                    'episodes_done': agent.episodes_done,
                    'update_count': agent.update_count
                }
            else:
                patience_counter += 1
            
            # Epsilon-aware early stopping
            current_epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * math.exp(-1. * agent.steps_done / config.epsilon_decay)
            
            if patience_counter >= early_stopping_patience and current_epsilon <= EPSILON_EARLY_STOPPING_THRESHOLD:
                print(f"\nEpsilon-aware early stopping triggered at episode {episode+1}")
                print(f"Patience counter: {patience_counter}, Current epsilon: {current_epsilon:.3f}")
                print(f"Best validation return: {best_validation_return:.2%}")
                
                # Restore best model
                if best_model_state:
                    agent.q_network.load_state_dict(best_model_state['q_network'])
                    agent.target_network.load_state_dict(best_model_state['target_network'])
                    agent.optimizer.load_state_dict(best_model_state['optimizer'])
                    agent.scheduler.load_state_dict(best_model_state['scheduler'])
                    agent.steps_done = best_model_state['steps_done']
                    agent.episodes_done = best_model_state['episodes_done']
                    agent.update_count = best_model_state['update_count']
                break
            elif patience_counter >= early_stopping_patience:
                print(f"\nValidation plateaued but epsilon still high ({current_epsilon:.3f})")
                print(f"Continuing training... (patience reset to {early_stopping_patience // 2})")
                # Partially reset patience counter to give more chances
                patience_counter = early_stopping_patience // 2
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            checkpoint_path = f"dqn_checkpoint_episode_{episode}.pt"
            agent.save(checkpoint_path, train_env.portfolio_normalizer)
            print(f"Saved checkpoint at episode {episode}")
            
            # Log portfolio normalization status
            if train_env.portfolio_normalizer is not None:
                if train_env.portfolio_normalizer.is_fitted:
                    print(f"  Portfolio normalizer: ✅ ACTIVE (training enabled)")
                else:
                    progress = train_env.portfolio_normalizer.episode_count / train_env.portfolio_normalizer.warmup_episodes
                    print(f"  Portfolio normalizer: 🔥 WARMUP ({progress*100:.1f}%) - ⚠️  TRAINING PAUSED")
    
    # Multi-day test evaluation
    print("\n" + "="*50)
    print("MULTI-DAY TEST EVALUATION (6 DAYS)")
    print("="*50)
    
    # Run backtests on 6 different days
    num_test_days = min(6, test_env.total_days)
    test_days = np.linspace(0, test_env.total_days - 1, num_test_days, dtype=int)
    
    all_test_results = []
    all_portfolio_values = []
    all_price_histories = []
    all_action_histories = []
    
    for i, day_idx in enumerate(test_days):
        print(f"\nRunning backtest for day {day_idx + 1}/{test_env.total_days} (Test {i+1}/{num_test_days})...")
        
        # Reset environment to specific day
        test_state = test_env.reset(day_idx=day_idx)
        test_reward = 0
        test_done = False
        test_action_history = []
        test_portfolio_values = [config.initial_balance]
        test_price_history = []
        
        while not test_done:
            test_action = agent.select_action(test_state, epsilon=0.0)
            test_next_state, test_r, test_done, test_info = test_env.step(test_action)
            test_reward += test_r
            test_state = test_next_state
            
            # Track for plotting
            test_action_history.append(test_action)
            current_value = test_info['balance'] + (test_info['position'] * test_info['current_price'])
            test_portfolio_values.append(current_value)
            test_price_history.append(test_info['current_price'])
        
        # Calculate metrics for this day
        test_final_value = test_portfolio_values[-1]
        test_return = (test_final_value - config.initial_balance) / config.initial_balance
        
        # Calculate performance metrics
        returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(test_portfolio_values)
        drawdown = (test_portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calculate Sharpe ratio
        if len(returns) > 0:
            portfolio_volatility = np.std(test_portfolio_values) / np.mean(test_portfolio_values)
            if portfolio_volatility > 1e-8:
                sharpe = test_return / portfolio_volatility
            else:
                sharpe = test_return * 10
        else:
            sharpe = 0.0
        
        # Store results
        day_results = {
            'day_idx': day_idx,
            'final_value': test_final_value,
            'total_return': test_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': test_info['total_trades'],
            'winning_trades': test_info['winning_trades'],
            'losing_trades': test_info['losing_trades'],
            'invalid_actions': test_info['invalid_actions'],
            'episode_reward': test_reward
        }
        
        all_test_results.append(day_results)
        all_portfolio_values.append(test_portfolio_values)
        all_price_histories.append(test_price_history)
        all_action_histories.append(test_action_history)
        
        # Print day results
        print(f"  Day {day_idx + 1} Results:")
        print(f"    Final Value: ${test_final_value:,.2f}")
        print(f"    Return: {test_return:.2%}")
        print(f"    Sharpe: {sharpe:.2f}")
        print(f"    Max Drawdown: {max_drawdown:.2%}")
        print(f"    Trades: {test_info['total_trades']} (W:{test_info['winning_trades']}, L:{test_info['losing_trades']})")
        print(f"    Invalid Actions: {test_info['invalid_actions']}")
    
    # Calculate aggregate statistics
    returns = [r['total_return'] for r in all_test_results]
    final_values = [r['final_value'] for r in all_test_results]
    sharpe_ratios = [r['sharpe_ratio'] for r in all_test_results]
    max_drawdowns = [r['max_drawdown'] for r in all_test_results]
    total_trades = [r['total_trades'] for r in all_test_results]
    winning_trades = [r['winning_trades'] for r in all_test_results]
    losing_trades = [r['losing_trades'] for r in all_test_results]
    invalid_actions = [r['invalid_actions'] for r in all_test_results]
    
    print(f"\n{'='*50}")
    print("AGGREGATE TEST RESULTS")
    print(f"{'='*50}")
    print(f"Average Return: {np.mean(returns):.2%} ± {np.std(returns):.2%}")
    print(f"Best Return: {np.max(returns):.2%}")
    print(f"Worst Return: {np.min(returns):.2%}")
    print(f"Win Rate: {np.sum([r > 0 for r in returns]) / len(returns):.1%}")
    print(f"Average Final Value: ${np.mean(final_values):,.2f}")
    print(f"Average Sharpe Ratio: {np.mean(sharpe_ratios):.2f}")
    print(f"Average Max Drawdown: {np.mean(max_drawdowns):.2%}")
    print(f"Average Trades per Day: {np.mean(total_trades):.1f}")
    print(f"Average Winning Trades: {np.mean(winning_trades):.1f}")
    print(f"Average Losing Trades: {np.mean(losing_trades):.1f}")
    print(f"Average Invalid Actions: {np.mean(invalid_actions):.1f}")
    
    # Create comprehensive results dictionary
    results = {
        'agent': agent,
        'preprocessor': preprocessor,
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'episode_trades': episode_trades,
        'episode_invalid_actions': episode_invalid_actions,
        'validation_rewards': validation_rewards,
        'validation_returns': validation_returns,
        'validation_trades': validation_trades,
        'validation_invalid_actions': validation_invalid_actions,
        'multi_day_test_results': {
            'individual_days': all_test_results,
            'portfolio_values': all_portfolio_values,
            'price_histories': all_price_histories,
            'action_histories': all_action_histories,
            'aggregate_stats': {
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'best_return': np.max(returns),
                'worst_return': np.min(returns),
                'win_rate': np.sum([r > 0 for r in returns]) / len(returns),
                'avg_final_value': np.mean(final_values),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'avg_trades': np.mean(total_trades),
                'avg_winning_trades': np.mean(winning_trades),
                'avg_losing_trades': np.mean(losing_trades),
                'avg_invalid_actions': np.mean(invalid_actions)
            }
        },
        'start_date': start_date,
        'end_date': end_date
    }
    
    # Automatically plot the multi-day backtest results
    print(f"\n{'='*50}")
    print("GENERATING MULTI-DAY BACKTEST PLOT")
    print(f"{'='*50}")
    
    try:
        plot_save_path = "multi_day_backtest_results.png"
        plot_multi_day_backtests(results, save_path=plot_save_path, show_plot=False)
        print(f"✅ Multi-day backtest plot saved to: {plot_save_path}")
    except Exception as e:
        print(f"❌ Error generating plot: {e}")
        print("Plot generation failed, but training results are still available")
    
    # Automatically analyze performance and provide recommendations
    print_trading_analysis(results)
    
    return results


def plot_multi_day_backtests(results: dict, save_path: str = None, show_plot: bool = True):
    """
    Plot portfolio performance for multiple days on the same figure
    
    Args:
        results: Results dictionary from train_dqn function
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    
    # Extract multi-day test results
    multi_day_results = results['multi_day_test_results']
    portfolio_values = multi_day_results['portfolio_values']
    individual_days = multi_day_results['individual_days']
    aggregate_stats = multi_day_results['aggregate_stats']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Color palette for different days
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot 1: Portfolio Values
    ax1.set_title('Multi-Day Portfolio Performance Comparison', fontsize=16, fontweight='bold')
    
    for i, (portfolio_vals, day_info) in enumerate(zip(portfolio_values, individual_days)):
        day_idx = day_info['day_idx']
        return_pct = day_info['total_return']
        
        # Create time axis (minutes within trading day)
        time_points = list(range(len(portfolio_vals)))
        
        # Plot portfolio value
        color = colors[i % len(colors)]
        ax1.plot(time_points, portfolio_vals, 
                label=f'Day {day_idx + 1} (Return: {return_pct:.1%})', 
                color=color, linewidth=2, alpha=0.8)
        
        # Add final value annotation
        final_val = portfolio_vals[-1]
        ax1.annotate(f'${final_val:,.0f}', 
                    xy=(len(time_points)-1, final_val),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold')
    
    # Add horizontal line for initial balance
    initial_balance = results['agent'].config.initial_balance
    ax1.axhline(y=initial_balance, color='black', linestyle='--', alpha=0.5, 
                label=f'Initial Balance (${initial_balance:,.0f})')
    
    ax1.set_xlabel('Minutes into Trading Day')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Normalized Returns (all starting at 100%)
    ax2.set_title('Normalized Returns Comparison (Starting at 100%)', fontsize=14, fontweight='bold')
    
    for i, (portfolio_vals, day_info) in enumerate(zip(portfolio_values, individual_days)):
        day_idx = day_info['day_idx']
        return_pct = day_info['total_return']
        
        # Normalize to percentage returns starting at 100%
        normalized_returns = [(val / portfolio_vals[0]) * 100 for val in portfolio_vals]
        time_points = list(range(len(normalized_returns)))
        
        color = colors[i % len(colors)]
        ax2.plot(time_points, normalized_returns, 
                label=f'Day {day_idx + 1} (Final: {normalized_returns[-1]:.1f}%)', 
                color=color, linewidth=2, alpha=0.8)
        
        # Add final percentage annotation
        final_pct = normalized_returns[-1]
        ax2.annotate(f'{final_pct:.1f}%', 
                    xy=(len(time_points)-1, final_pct),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold')
    
    # Add horizontal line at 100%
    ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Break-even (100%)')
    
    ax2.set_xlabel('Minutes into Trading Day')
    ax2.set_ylabel('Portfolio Value (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    # Add aggregate statistics as text box
    stats_text = f"""Aggregate Statistics (6 Days):
    Average Return: {aggregate_stats['avg_return']:.1%} ± {aggregate_stats['std_return']:.1%}
    Best Return: {aggregate_stats['best_return']:.1%}
    Worst Return: {aggregate_stats['worst_return']:.1%}
    Win Rate: {aggregate_stats['win_rate']:.0%}
    Avg Trades/Day: {aggregate_stats['avg_trades']:.1f}
    Avg Sharpe Ratio: {aggregate_stats['avg_sharpe_ratio']:.2f}"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def run_standalone_backtest(agent, test_env, num_days: int = 6, plot_results: bool = True, save_plot_path: str = None):
    """
    Run a standalone multi-day backtest with an already trained agent
    
    Args:
        agent: Trained DoubleDuelingDQN agent
        test_env: TradingEnvironment for testing
        num_days: Number of days to test
        plot_results: Whether to plot the results
        save_plot_path: Path to save the plot
    
    Returns:
        Dictionary with backtest results
    """
    print(f"\n{'='*60}")
    print(f"STANDALONE MULTI-DAY BACKTEST ({num_days} DAYS)")
    print(f"{'='*60}")
    
    # Run backtests on multiple days
    num_test_days = min(num_days, test_env.total_days)
    test_days = np.linspace(0, test_env.total_days - 1, num_test_days, dtype=int)
    
    all_test_results = []
    all_portfolio_values = []
    all_price_histories = []
    all_action_histories = []
    
    for i, day_idx in enumerate(test_days):
        print(f"\nRunning backtest for day {day_idx + 1}/{test_env.total_days} (Test {i+1}/{num_test_days})...")
        
        # Reset environment to specific day
        test_state = test_env.reset(day_idx=day_idx)
        test_reward = 0
        test_done = False
        test_action_history = []
        test_portfolio_values = [agent.config.initial_balance]
        test_price_history = []
        
        while not test_done:
            test_action = agent.select_action(test_state, epsilon=0.0)
            test_next_state, test_r, test_done, test_info = test_env.step(test_action)
            test_reward += test_r
            test_state = test_next_state
            
            # Track for plotting
            test_action_history.append(test_action)
            current_value = test_info['balance'] + (test_info['position'] * test_info['current_price'])
            test_portfolio_values.append(current_value)
            test_price_history.append(test_info['current_price'])
        
        # Calculate metrics for this day
        test_final_value = test_portfolio_values[-1]
        test_return = (test_final_value - agent.config.initial_balance) / agent.config.initial_balance
        
        # Calculate performance metrics
        returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(test_portfolio_values)
        drawdown = (test_portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calculate Sharpe ratio
        if len(returns) > 0:
            portfolio_volatility = np.std(test_portfolio_values) / np.mean(test_portfolio_values)
            if portfolio_volatility > 1e-8:
                sharpe = test_return / portfolio_volatility
            else:
                sharpe = test_return * 10
        else:
            sharpe = 0.0
        
        # Store results
        day_results = {
            'day_idx': day_idx,
            'final_value': test_final_value,
            'total_return': test_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': test_info['total_trades'],
            'winning_trades': test_info['winning_trades'],
            'losing_trades': test_info['losing_trades'],
            'invalid_actions': test_info['invalid_actions'],
            'episode_reward': test_reward
        }
        
        all_test_results.append(day_results)
        all_portfolio_values.append(test_portfolio_values)
        all_price_histories.append(test_price_history)
        all_action_histories.append(test_action_history)
        
        # Print day results
        print(f"  Day {day_idx + 1} Results:")
        print(f"    Final Value: ${test_final_value:,.2f}")
        print(f"    Return: {test_return:.2%}")
        print(f"    Sharpe: {sharpe:.2f}")
        print(f"    Max Drawdown: {max_drawdown:.2%}")
        print(f"    Trades: {test_info['total_trades']} (W:{test_info['winning_trades']}, L:{test_info['losing_trades']})")
        print(f"    Invalid Actions: {test_info['invalid_actions']}")
    
    # Calculate aggregate statistics
    returns = [r['total_return'] for r in all_test_results]
    final_values = [r['final_value'] for r in all_test_results]
    sharpe_ratios = [r['sharpe_ratio'] for r in all_test_results]
    max_drawdowns = [r['max_drawdown'] for r in all_test_results]
    total_trades = [r['total_trades'] for r in all_test_results]
    winning_trades = [r['winning_trades'] for r in all_test_results]
    losing_trades = [r['losing_trades'] for r in all_test_results]
    invalid_actions = [r['invalid_actions'] for r in all_test_results]
    
    print(f"\n{'='*50}")
    print("AGGREGATE BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"Average Return: {np.mean(returns):.2%} ± {np.std(returns):.2%}")
    print(f"Best Return: {np.max(returns):.2%}")
    print(f"Worst Return: {np.min(returns):.2%}")
    print(f"Win Rate: {np.sum([r > 0 for r in returns]) / len(returns):.1%}")
    print(f"Average Final Value: ${np.mean(final_values):,.2f}")
    print(f"Average Sharpe Ratio: {np.mean(sharpe_ratios):.2f}")
    print(f"Average Max Drawdown: {np.mean(max_drawdowns):.2%}")
    print(f"Average Trades per Day: {np.mean(total_trades):.1f}")
    print(f"Average Winning Trades: {np.mean(winning_trades):.1f}")
    print(f"Average Losing Trades: {np.mean(losing_trades):.1f}")
    print(f"Average Invalid Actions: {np.mean(invalid_actions):.1f}")
    
    # Create results dictionary
    results = {
        'agent': agent,
        'multi_day_test_results': {
            'individual_days': all_test_results,
            'portfolio_values': all_portfolio_values,
            'price_histories': all_price_histories,
            'action_histories': all_action_histories,
            'aggregate_stats': {
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'best_return': np.max(returns),
                'worst_return': np.min(returns),
                'win_rate': np.sum([r > 0 for r in returns]) / len(returns),
                'avg_final_value': np.mean(final_values),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'avg_trades': np.mean(total_trades),
                'avg_winning_trades': np.mean(winning_trades),
                'avg_losing_trades': np.mean(losing_trades),
                'avg_invalid_actions': np.mean(invalid_actions)
            }
        }
    }
    
    # Plot results if requested
    if plot_results:
        plot_multi_day_backtests(results, save_path=save_plot_path)
    
    return results


if __name__ == "__main__":
    # Example of how to use different architectures
    def test_architectures():
        """Test different DQN architectures for financial time series"""
        from src.config.config import DATA_DIR
        
        print("="*80)
        print("DQN ARCHITECTURE COMPARISON FOR FINANCIAL TIME SERIES")
        print("="*80)
        
        # Test configurations
        architectures = [
            {
                "name": "Original CNN",
                "config": TradingConfig(
                    architecture_type=ArchitectureType.ORIGINAL,
                    hidden_size=512,
                    learning_rate=1e-4
                )
            },
            {
                "name": "Improved Transformer + Multi-scale CNN",
                "config": TradingConfig(
                    architecture_type=ArchitectureType.IMPROVED,
                    hidden_size=512,
                    learning_rate=1e-4,
                    transformer_layers=2,
                    use_attention=True,
                    cnn_scales=[3, 5, 7]
                )
            },
            {
                "name": "Hybrid CNN + LSTM",
                "config": TradingConfig(
                    architecture_type=ArchitectureType.HYBRID,
                    hidden_size=512,
                    learning_rate=8e-5,  # Slightly lower for LSTM stability
                    cnn_scales=[3, 5, 7]
                )
            }
        ]
        
        for arch in architectures:
            print(f"\n{'='*60}")
            print(f"TESTING: {arch['name']}")
            print(f"{'='*60}")
            
            config = arch['config']
            print(f"Configuration:")
            print(f"  Architecture: {config.architecture_type.value}")
            print(f"  Hidden Size: {config.hidden_size}")
            print(f"  Learning Rate: {config.learning_rate}")
            print(f"  Batch Size: {config.batch_size}")
            print(f"  Buffer Size: {config.buffer_size:,}")
            
            # Create network to show architecture details
            try:
                network = create_network(config)
                total_params = sum(p.numel() for p in network.parameters())
                trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
                
                print(f"\nNetwork Details:")
                print(f"  Total Parameters: {total_params:,}")
                print(f"  Trainable Parameters: {trainable_params:,}")
                print(f"  Memory Estimate: ~{total_params * 4 / 1024 / 1024:.1f} MB")
                
                # Show model structure
                print(f"\nArchitecture Summary:")
                if config.architecture_type == ArchitectureType.ORIGINAL:
                    print("  • Standard 1D CNN with BatchNorm")
                    print("  • ReLU activations")
                    print("  • Max pooling")
                    print("  • Simple dueling streams")
                    
                elif config.architecture_type == ArchitectureType.IMPROVED:
                    print("  • Multi-scale CNN (3, 5, 7 kernel sizes)")
                    print("  • Transformer blocks with self-attention")
                    print("  • GELU activations (better for financial data)")
                    print("  • GroupNorm instead of BatchNorm")
                    print("  • Positional encoding for time awareness")
                    print("  • Attention pooling")
                    print("  • Residual connections in shared layers")
                    
                elif config.architecture_type == ArchitectureType.HYBRID:
                    print("  • Multi-scale CNN for local patterns")
                    print("  • Bidirectional LSTM for temporal dependencies")
                    print("  • Attention mechanism for LSTM outputs")
                    print("  • GELU activations")
                    print("  • GroupNorm for stability")
                
                print(f"\nBest Use Cases:")
                if config.architecture_type == ArchitectureType.ORIGINAL:
                    print("  ✓ Baseline model")
                    print("  ✓ Quick prototyping") 
                    print("  ✓ Limited computational resources")
                    print("  ✓ Simple pattern recognition")
                    
                elif config.architecture_type == ArchitectureType.IMPROVED:
                    print("  ✓ Complex temporal relationships")
                    print("  ✓ Long-range dependencies")
                    print("  ✓ Multi-timeframe analysis")
                    print("  ✓ When you have sufficient data")
                    print("  ✓ Production deployment with good hardware")
                    
                elif config.architecture_type == ArchitectureType.HYBRID:
                    print("  ✓ Best of both worlds (CNN + RNN)")
                    print("  ✓ Sequential pattern recognition")
                    print("  ✓ Trend following strategies")
                    print("  ✓ Medium computational requirements")
                
                print(f"\nExpected Performance Characteristics:")
                if config.architecture_type == ArchitectureType.ORIGINAL:
                    print("  • Training Speed: Fast")
                    print("  • Memory Usage: Low")
                    print("  • Pattern Recognition: Basic")
                    print("  • Overfitting Risk: Medium")
                    
                elif config.architecture_type == ArchitectureType.IMPROVED:
                    print("  • Training Speed: Moderate")
                    print("  • Memory Usage: High")
                    print("  • Pattern Recognition: Advanced")
                    print("  • Overfitting Risk: Low (with proper regularization)")
                    
                elif config.architecture_type == ArchitectureType.HYBRID:
                    print("  • Training Speed: Moderate")
                    print("  • Memory Usage: Medium-High")
                    print("  • Pattern Recognition: Good")
                    print("  • Overfitting Risk: Medium")
                
                del network  # Free memory
                
            except Exception as e:
                print(f"  Error creating network: {e}")
        
        print(f"\n{'='*80}")
        print("TRAINING RECOMMENDATIONS")
        print(f"{'='*80}")
        print("For minute-level financial data (regular trading hours):")
        print("\n1. START with 'improved' architecture for best performance")
        print("   - Has attention mechanisms for temporal dependencies")
        print("   - Multi-scale feature extraction")
        print("   - Better activations for financial data")
        
        print("\n2. USE 'hybrid' if you want CNN+LSTM combination")
        print("   - Good balance of performance and efficiency")
        print("   - Excellent for trend-following strategies")
        
        print("\n3. FALLBACK to 'original' for:")
        print("   - Limited computational resources")
        print("   - Quick experiments")
        print("   - Baseline comparisons")
        
        print("\n4. HYPERPARAMETER TIPS:")
        print("   - Learning Rate: 1e-4 to 5e-5 for financial data")
        print("   - Batch Size: 32-128 (larger for more stable gradients)")
        print("   - Buffer Size: 500K+ for good experience diversity")
        print("   - Window Size: 20 minutes works well for intraday")
        print("   - Use preprocessing (robust scaling + winsorizing)")
        
        print("\n5. TRAINING BEST PRACTICES:")
        print("   - Use early stopping with patience")
        print("   - Monitor both training and validation metrics")
        print("   - Save model checkpoints regularly")
        print("   - Start with shorter episodes, increase gradually")
        print("   - Use prioritized experience replay")
        
        print(f"\n{'='*80}")
        print("Ready to train! Use:")
        print("config = TradingConfig(architecture_type='improved')")
        print("agent = DoubleDuelingDQN(config)")
        print(f"{'='*80}")
    
    # Run the demonstration
    test_architectures()
    
    # Uncomment to train with ENHANCED AGGRESSIVE TRADING:
    # 
    # # Create aggressive configuration for more active trading
    # aggressive_config = create_aggressive_trading_config()
    # 
    # results = train_dqn(
    #     data_path=f"{DATA_DIR}/feature_engineered/TSLA.csv",
    #     cutoff=pd.Timestamp('2020-01-01', tz='UTC'),
    #     num_episodes=100,
    #     use_preprocessing=True,
    #     scaling_method='robust',
    #     outlier_method='winsorize',
    #     architecture_type=ArchitectureType.IMPROVED
    # )
    # 
    # # The model will now use:
    # # ✅ Enhanced reward system (5x higher trade incentives)
    # # ✅ Reduced invalid action penalties  
    # # ✅ Stronger undertrading penalties
    # # ✅ Higher exploration (epsilon_end=0.15 vs 0.05)
    # # ✅ More frequent learning updates
    # 
    # # Plot the multi-day backtest results
    # plot_multi_day_backtests(results, save_path="aggressive_trading_backtest.png")
    # 
    # print("🎯 Expected improvements:")
    # print("   • 5-15 trades per day with smart timing")
    # print("   • Better win rates through technical analysis")
    # print("   • Quality over quantity trading approach")
    # print("   • Context-aware entry and exit decisions")
    # print("   • Reduced invalid actions with strategic trading")
    # 
    # # Or run a standalone backtest with a trained model:
    # # backtest_results = run_standalone_backtest(
    # #     agent=results['agent'], 
    # #     test_env=test_env, 
    # #     num_days=6, 
    # #     plot_results=True,
    # #     save_plot_path="standalone_backtest.png"
    # # )
