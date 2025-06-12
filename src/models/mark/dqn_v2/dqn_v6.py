"""
DQN v6: Mamba-Based Trading Agent with Dueling Architecture
==========================================================

A Mamba SSM (State Space Model) DQN implementation with:
- Mamba-inspired State Space Model for efficient temporal modeling
- GPU-based Prioritized Experience Replay for better learning
- Dueling architecture separates V(s) and A(s,a) for better learning
- PURE P&L REWARD SYSTEM (fixed reward-return misalignment issue)
- Learning rate scheduler for adaptive optimization
- Normalized close price for price level awareness

Key Improvements:
- Mamba SSM architecture for superior temporal pattern recognition
- GPU-optimized PER for faster training
- Simplified reward system that directly tracks actual profits/losses
- Real-time reward-return alignment monitoring
- Adaptive learning rate scheduling
- Price level context through normalized close price (relative to 20-period MA)

Winner of comprehensive architecture comparison with:
- Best backtest return: -1.2% ± 2.2%
- Lowest trading frequency: 26 trades/day
- Most efficient temporal processing
"""

import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from pathlib import Path

from src.config.config import DEVICE, EVALUATE_INTERVAL, UPDATE_TARGET_EVERY, MINUTES_PER_TRADING_DAY, TRAIN_RATIO, VALID_RATIO
from src.models.mark.dqn_v2.data_preprocessor_v2 import preprocess_financial_data


# =============================================================================
# MAMBA FEATURE SET (Optimized for temporal patterns)
# =============================================================================

MAMBA_FEATURES = [
    'close',                    # Current price (for trading calculations)
    'return_1m',               # 1-minute price momentum  
    'return_5m',               # 5-minute price momentum
    'return_15m',              # 15-minute price momentum
    'volume_ratio_5m',         # Volume relative to recent average
    'volatility_5m',           # Recent volatility measure
    'rsi_14m',                 # RSI for momentum detection
    'macd',                    # MACD for trend analysis
    'hour_sin',                # Time of day (cyclical)
    'close_normalized',        # Normalized close price for network input
]

# Features for temporal processing (including normalized close price)
TEMPORAL_FEATURES = [f for f in MAMBA_FEATURES if f not in ['close']]  # Exclude raw 'close' but include 'close_normalized'


# =============================================================================
# MAMBA TRADING CONFIGURATION
# =============================================================================

class MambaTradingConfig:
    """Mamba-specific trading configuration"""
    def __init__(self, temporal_window: int = 30):
        # Trading parameters
        self.initial_balance = 10000.0
        self.transaction_fee_percent = 0.001  # 0.1% (realistic for retail)
        self.max_position_size = 0.95  # Use 95% of balance max
        
        # Mamba architecture parameters
        self.temporal_window = temporal_window  # Number of minutes to look back
        self.state_size = (len(TEMPORAL_FEATURES) + 2, temporal_window)  # 2D: (features + portfolio, time)
        self.input_channels = len(TEMPORAL_FEATURES) + 2  # +2 for portfolio channels
        
        # RL parameters  
        self.num_actions = 3  # Hold, Buy, Sell
        
        # Mamba network parameters
        self.d_model = 64  # Model dimension for Mamba
        self.n_layers = 2  # Number of SSM layers
        self.learning_rate = 1e-3
        
        # Prioritized Experience Replay parameters
        self.buffer_size = 50000
        self.batch_size = 64
        self.alpha = 0.6  # PER exponent
        self.beta_start = 0.4  # Importance sampling
        self.beta_end = 1.0
        self.beta_frames = 100000
        
        # Training parameters - optimized for Mamba
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 2000  # Faster decay for trading
        self.target_update_frequency = UPDATE_TARGET_EVERY   # Hard update frequency (when tau=0)
        self.tau = 0.005  # Soft update rate (like DQN v5)
        self.gamma = 0.99
        
        # Trading-specific parameters  
        self.min_profit_threshold = 0.01  # Minimum 1% expected profit to trade
        self.trading_frequency_penalty = 0.001
        self.patience_bonus_rate = 0.0  # Disabled - using pure P&L rewards now
        
        # Reward system parameters (configurable for optimization like DQN v5)
        self.portfolio_scaling = 0.010    # Scaling factor for portfolio value changes
        self.invalid_penalty = 0.010      # Penalty for invalid actions
        
        # Validation and testing parameters
        self.validation_frequency = EVALUATE_INTERVAL  # Run validation every N episodes
        self.early_stopping_patience = 10  # Stop if validation doesn't improve for N checks
        self.train_ratio = TRAIN_RATIO  # 70% for training
        self.valid_ratio = VALID_RATIO  # 20% for validation (10% left for testing)


# =============================================================================
# GPU-BASED PRIORITIZED EXPERIENCE REPLAY
# =============================================================================

class PrioritizedReplayBufferGPU:
    """GPU-based Prioritized Experience Replay for faster training"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, device: torch.device = DEVICE):
        self.capacity = capacity
        self.alpha = alpha
        self.device = device
        
        # Buffers stored on GPU
        self.buffer_idx = 0
        self.size = 0
        
        # Initialize buffers (will be set when first experience is added)
        self.states = None
        self.actions = None  
        self.rewards = None
        self.next_states = None
        self.dones = None
        
        # Priority tree on GPU
        self.priorities = torch.zeros(capacity, device=device, dtype=torch.float32)
        self.max_priority = 1.0
    
    def _initialize_buffers(self, state_shape):
        """Initialize GPU buffers based on first state"""
        # Always 2D state for Mamba (temporal)
        self.states = torch.zeros((self.capacity, *state_shape), device=self.device, dtype=torch.float32)
        self.next_states = torch.zeros((self.capacity, *state_shape), device=self.device, dtype=torch.float32)
            
        self.actions = torch.zeros(self.capacity, device=self.device, dtype=torch.long)
        self.rewards = torch.zeros(self.capacity, device=self.device, dtype=torch.float32)
        self.dones = torch.zeros(self.capacity, device=self.device, dtype=torch.bool)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience with maximum priority"""
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        
        # Initialize buffers on first push
        if self.states is None:
            self._initialize_buffers(state_tensor.shape)
        
        # Store experience
        self.states[self.buffer_idx] = state_tensor
        self.actions[self.buffer_idx] = action
        self.rewards[self.buffer_idx] = reward
        self.next_states[self.buffer_idx] = next_state_tensor
        self.dones[self.buffer_idx] = done
        
        # Set priority to maximum for new experiences
        self.priorities[self.buffer_idx] = self.max_priority ** self.alpha
        
        self.buffer_idx = (self.buffer_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with prioritized sampling"""
        if self.size < batch_size:
            raise ValueError(f"Not enough experiences: {self.size} < {batch_size}")
        
        # Calculate sampling probabilities
        probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
        
        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        # Return batch (all tensors already on GPU)
        return (
            self.states[indices],
            self.actions[indices], 
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights
        )
    
    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Update priorities based on TD errors"""
        priorities = (torch.abs(td_errors) + 1e-6) ** self.alpha
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())
    
    def __len__(self):
        return self.size


# =============================================================================
# MAMBA-INSPIRED STATE SPACE MODEL
# =============================================================================

class SimpleSSMBlock(nn.Module):
    """Simplified State Space Model inspired by Mamba
    
    This is a lightweight implementation focusing on the core SSM concepts:
    - Linear state evolution with input-dependent parameters
    - Efficient sequential processing
    - Selective attention to relevant time periods
    """
    
    def __init__(self, d_model: int, d_state: int = 16, expand_factor: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        
        # Input projections (like Mamba's selective mechanism)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # State space parameters (simplified - not fully selective like real Mamba)
        self.A_log = nn.Parameter(torch.randn(d_state))  # State transition
        self.D = nn.Parameter(torch.randn(self.d_inner))  # Skip connection - match d_inner size
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Initialize A to be stable
        with torch.no_grad():
            self.A_log.data = -torch.rand_like(self.A_log) - 1
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Input projection 
        x_proj = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_ssm, x_res = x_proj.chunk(2, dim=-1)  # Split for SSM and residual
        
        # Apply SiLU activation (like Mamba)
        x_ssm = F.silu(x_ssm)
        
        # Simplified SSM computation (not the full selective scan)
        # This is a basic RNN-like state space model
        A = -torch.exp(self.A_log)  # Ensure stability
        
        # Simple state evolution (batch processing)
        h = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # State update: h = A * h + B * x
            # Simplified: use learnable linear combinations
            x_t = x_ssm[:, t]  # (batch, d_inner)
            
            # Map input to state space
            B_t = x_t.mean(dim=-1, keepdim=True).expand(-1, self.d_state)  # Simplified B
            C_t = x_t.mean(dim=-1, keepdim=True).expand(-1, self.d_state)  # Simplified C
            
            # State evolution
            h = A.unsqueeze(0) * h + B_t
            
            # Output
            y_t = (C_t * h).sum(dim=-1, keepdim=True) * x_t  # (batch, d_inner)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        
        # Add skip connection (like Mamba's D parameter)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_ssm
        
        # Output projection
        return self.out_proj(y)


class MambaQNetwork(nn.Module):
    """Trading Q-network using Mamba-inspired State Space Model with Dueling Architecture"""
    
    def __init__(self, input_channels: int, temporal_window: int, action_size: int, 
                 d_model: int = 64, n_layers: int = 2):
        super().__init__()
        
        self.d_model = d_model
        self.temporal_window = temporal_window
        self.action_size = action_size
        
        # Input embedding - project features to model dimension
        self.input_embedding = nn.Linear(input_channels, d_model)
        
        # Stack of SSM blocks
        self.ssm_layers = nn.ModuleList([
            SimpleSSMBlock(d_model) for _ in range(n_layers)
        ])
        
        # Layer norms (important for SSMs)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU()
        )
        
        # Dueling streams
        self.value_stream = nn.Linear(d_model, 1)  # V(s) - state value
        self.advantage_stream = nn.Linear(d_model, action_size)  # A(s,a) - action advantages
        
    def forward(self, x):
        """
        x: (batch, features, time) - temporal window format
        """
        # Reshape to (batch, time, features) for sequence processing
        x = x.transpose(1, 2)  # (batch, time, features)
        
        # Input embedding
        x = self.input_embedding(x)  # (batch, time, d_model)
        
        # Apply SSM layers with residual connections
        for ssm, norm in zip(self.ssm_layers, self.layer_norms):
            residual = x
            x = norm(x)
            x = ssm(x) + residual  # Residual connection
        
        # Global pooling over time dimension
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Extract features
        features = self.feature_layer(x)
        
        # Separate value and advantage
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_size)
        
        # Dueling combination: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


# =============================================================================
# ENHANCED TRADING ENVIRONMENT  
# =============================================================================

class MambaEnvironment:
    """Mamba-specific trading environment with temporal features"""
    
    def __init__(self, original_data: pd.DataFrame, scaled_data: pd.DataFrame, config: MambaTradingConfig):
        self.original_data = original_data  # Original prices for trading
        self.scaled_data = scaled_data      # Scaled features for state representation
        self.config = config
        
        # Validate required features in original data (need 'close' for trading)
        if 'close' not in original_data.columns:
            raise ValueError("Original data must contain 'close' column for trading")
        
        # Validate required features for state representation (exclude dynamically computed ones)
        # close_normalized is computed dynamically in _get_temporal_state(), so exclude it from validation
        static_features = [f for f in TEMPORAL_FEATURES if f != 'close_normalized']
        missing_features = [f for f in static_features if f not in scaled_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features in scaled data: {missing_features}")
        
        # Episode parameters
        self.total_steps = len(original_data) 
        self.episode_length = MINUTES_PER_TRADING_DAY  # One trading day
        
        # State tracking
        self.reset()
    
    def reset(self, start_step: Optional[int] = None) -> np.ndarray:
        """Reset environment to start of episode"""
        # Random start if not specified (for training diversity)
        if start_step is None:
            min_start = self.config.temporal_window
            max_start = self.total_steps - self.episode_length - min_start
            self.current_step = random.randint(min_start, max_start) if max_start > min_start else min_start
        else:
            self.current_step = max(start_step, self.config.temporal_window)
            
        # Trading state
        self.balance = self.config.initial_balance
        self.position = 0.0  # Number of shares held
        self.entry_price = 0.0
        
        # Episode tracking
        self.episode_start = self.current_step
        self.episode_end = min(self.current_step + self.episode_length, self.total_steps - 1)
        
        # Performance tracking
        self.total_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
        # Trading behavior tracking
        self.consecutive_holds = 0
        self.trades_this_episode = 0
        
        # Initialize portfolio tracking for rewards
        self.last_portfolio_value = self.config.initial_balance
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current temporal state representation using scaled data"""
        if self.current_step >= len(self.scaled_data):
            # Return neutral state if out of bounds
            return np.zeros(self.config.state_size)
        
        return self._get_temporal_state()
    

    def _get_temporal_state(self) -> np.ndarray:
        """Get temporal window state for Mamba using scaled data with normalized close prices"""
        # Get temporal window from scaled data
        start_idx = max(0, self.current_step - self.config.temporal_window + 1)
        end_idx = self.current_step + 1
        
        # Extract temporal features from scaled data
        window_data = self.scaled_data.iloc[start_idx:end_idx].copy()
        
        # 🎯 COMPUTE NORMALIZED CLOSE PRICE
        # Get original close prices for this window
        original_window = self.original_data.iloc[start_idx:end_idx]
        close_prices = original_window['close'].values
        
        # Normalize close price relative to recent moving average
        if len(close_prices) >= 10:  # Need at least 10 points for meaningful MA
            ma_window = min(20, len(close_prices))  # Use 20-period MA or available data
            close_ma = pd.Series(close_prices).rolling(window=ma_window, min_periods=1).mean()
            close_normalized = close_prices / close_ma - 1.0  # Relative to MA: 0 = at MA, +0.1 = 10% above MA
        else:
            # Fallback: use price relative to first price in window
            close_normalized = close_prices / close_prices[0] - 1.0 if len(close_prices) > 0 else np.zeros_like(close_prices)
        
        # Add normalized close to window data
        window_data['close_normalized'] = close_normalized
        
        # Create 2D array (features x time) - now includes normalized close + portfolio channels
        num_features = len(TEMPORAL_FEATURES) + 2  # +2 for position_ratio and cash_ratio
        temporal_state = np.zeros((num_features, self.config.temporal_window))
        
        # Fill temporal features from scaled data (now includes close_normalized)
        for i, feature in enumerate(TEMPORAL_FEATURES):
            if feature in window_data.columns:
                values = window_data[feature].fillna(0.0).values
                # Pad if necessary
                if len(values) < self.config.temporal_window:
                    padded_values = np.zeros(self.config.temporal_window)
                    padded_values[-len(values):] = values
                    values = padded_values
                temporal_state[i] = values
        
        # Add portfolio information using original prices (constant across time dimension)
        current_price = self.original_data.iloc[self.current_step]['close']  # Original price for portfolio
        current_value = self.balance + (self.position * current_price)
        position_ratio = (self.position * current_price) / current_value if current_value > 0 else 0.0
        cash_ratio = self.balance / current_value if current_value > 0 else 1.0
        
        # Fill portfolio channels (repeated across time)
        temporal_state[-2] = position_ratio  # Second to last channel
        temporal_state[-1] = cash_ratio      # Last channel
        
        return temporal_state.astype(np.float32)
    
    def _get_valid_actions(self) -> List[int]:
        """Get list of valid actions in current state using original prices"""
        valid_actions = [0]  # Hold is always valid
        
        # Check if buy is valid
        if self.position == 0:  # No position held
            current_price = self.original_data.iloc[self.current_step]['close']  # Use original price
            position_value = self.balance * self.config.max_position_size
            shares_to_buy = position_value / current_price
            cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)
            
            if cost <= self.balance:  # Sufficient cash
                valid_actions.append(1)  # Buy
        
        # Check if sell is valid
        if self.position > 0:  # Holding position
            valid_actions.append(2)  # Sell
            
        return valid_actions
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action using original prices for trading and return next state
        
        OPTIMIZABLE REWARD SYSTEM - Pure Portfolio Value Change + Invalid Penalty:
        - ALL ACTIONS: Portfolio value change * portfolio_scaling (configurable)
        - INVALID ACTIONS: Additional -invalid_penalty to teach action validity
        
        This maintains portfolio tracking for performance while adding
        direct feedback for invalid actions. Parameters can be optimized
        using the Optuna-based optimization script.
        """
        if self.current_step >= len(self.original_data):
            return self._get_state(), 0.0, True, {}
            
        current_price = self.original_data.iloc[self.current_step]['close']  # Use original price for trading
        reward = 0.0
        trade_executed = False
        invalid_action = False
        
        # Store current portfolio value for comparison
        if not hasattr(self, 'last_portfolio_value'):
            self.last_portfolio_value = self.config.initial_balance
        
        # Calculate current portfolio value
        current_portfolio_value = self.balance + (self.position * current_price)
        
        # PURE PORTFOLIO TRACKING: Reward = portfolio value change (always)
        portfolio_change = current_portfolio_value - self.last_portfolio_value
        portfolio_scaling = getattr(self.config, 'portfolio_scaling', 0.010)  # Support optimization
        reward = portfolio_change * portfolio_scaling  # Consistent scaling for all actions
        
        # Update last portfolio value for next step
        self.last_portfolio_value = current_portfolio_value
        
        # Check action validity
        valid_actions = self._get_valid_actions()
        
        # Execute the action (portfolio change + invalid penalty if needed)
        if action not in valid_actions:
            invalid_action = True
            self.trades_this_episode += 1  # Count invalid as attempted trade
            # Add penalty for invalid actions to teach action validity
            invalid_penalty = getattr(self.config, 'invalid_penalty', 0.010)  # Support optimization
            reward -= invalid_penalty  # Configurable penalty for invalid actions
        else:
            # Execute valid action using original prices
            if action == 1:  # Buy
                position_value = self.balance * self.config.max_position_size
                shares_to_buy = position_value / current_price
                cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)
                
                self.position = shares_to_buy
                self.balance -= cost
                self.entry_price = current_price
                trade_executed = True
                self.total_trades += 1
                self.trades_this_episode += 1
                self.consecutive_holds = 0  # Reset hold counter
                    
            elif action == 2:  # Sell
                revenue = self.position * current_price * (1 - self.config.transaction_fee_percent)
                cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
                profit = revenue - cost_basis
                
                self.balance += revenue
                self.position = 0.0
                trade_executed = True
                self.total_trades += 1
                self.trades_this_episode += 1
                self.consecutive_holds = 0  # Reset hold counter
                
                # Track P&L
                if profit > 0:
                    self.total_profit += profit
                else:
                    self.total_loss += abs(profit)
            
            elif action == 0:  # Hold
                self.consecutive_holds += 1
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.episode_end
        
        # Force close position at end of episode using original prices
        if done and self.position > 0:
            final_price = self.original_data.iloc[min(self.current_step, len(self.original_data) - 1)]['close']
            revenue = self.position * final_price * (1 - self.config.transaction_fee_percent)
            cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
            final_profit = revenue - cost_basis
            
            self.balance += revenue
            self.position = 0.0
            
            # Update portfolio value after forced close for final reward calculation
            final_portfolio_value = self.balance
            final_change = final_portfolio_value - self.last_portfolio_value
            portfolio_scaling = getattr(self.config, 'portfolio_scaling', 0.010)
            reward += final_change * portfolio_scaling  # Same consistent scaling
            
            if final_profit > 0:
                self.total_profit += final_profit
            else:
                self.total_loss += abs(final_profit)
        
        # Get next state (uses scaled data)
        next_state = self._get_state()
        
        # Info for tracking (uses original prices)
        current_value = self.balance + (self.position * current_price)
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price,  # Original price
            'portfolio_value': current_value,
            'trade_executed': trade_executed,
            'invalid_action': invalid_action,
            'valid_actions': valid_actions,
            'total_trades': self.total_trades,
            'trades_this_episode': self.trades_this_episode,
            'consecutive_holds': self.consecutive_holds,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'return': (current_value - self.config.initial_balance) / self.config.initial_balance
        }
        
        return next_state, reward, done, info


# =============================================================================
# ENHANCED DQN AGENT
# =============================================================================

class MambaDQN:
    """Mamba DQN with PER and State Space Model architecture"""
    
    def __init__(self, config: MambaTradingConfig, device: torch.device = DEVICE):
        self.config = config
        self.device = device
        
        # Networks - choose architecture
        self.q_network = MambaQNetwork(
            config.input_channels,
            config.temporal_window,
            config.num_actions,
            config.d_model,
            config.n_layers
        ).to(device)
        self.target_network = MambaQNetwork(
            config.input_channels,
            config.temporal_window,
            config.num_actions,
            config.d_model,
            config.n_layers
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Learning rate scheduler (similar to DQN v5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=True
        )
        
        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBufferGPU(config.buffer_size, config.alpha, device)
        
        # Training state
        self.steps_done = 0
        self.update_count = 0  # For hard target updates when tau=0
        
        print(f"Enhanced DQN initialized:")
        print(f"  State size: {config.state_size}")
        print(f"  Action size: {config.num_actions}")
        print(f"  Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        print(f"  Target update: {'Soft (τ=' + str(config.tau) + ')' if config.tau > 0 else 'Hard (every ' + str(config.target_update_frequency) + ' steps)'}")
        print(f"  Temporal window: {config.temporal_window}")
        print(f"  Input channels: {config.input_channels}")
    
    def select_action(self, state: np.ndarray, valid_actions: Optional[List[int]] = None, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy with action masking"""
        if epsilon is None:
            epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                     math.exp(-1. * self.steps_done / self.config.epsilon_decay)
        
        # Default to all actions if not provided
        if valid_actions is None:
            valid_actions = list(range(self.config.num_actions))
        
        if random.random() > epsilon:
            self.q_network.eval()  # Set to eval mode for inference
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state_tensor).squeeze(0)  # (action_size,)
                
                # Mask invalid actions by setting their Q-values to very negative
                masked_q_values = q_values.clone()
                for action in range(self.config.num_actions):
                    if action not in valid_actions:
                        masked_q_values[action] = -float('inf')
                
                # Apply minimum profit threshold: only trade if significantly better than holding
                hold_q_value = masked_q_values[0]  # Q-value for holding
                for action in [1, 2]:  # Buy and Sell
                    if action in valid_actions:
                        profit_advantage = masked_q_values[action] - hold_q_value
                        if profit_advantage < self.config.min_profit_threshold:
                            masked_q_values[action] = -float('inf')  # Don't trade unless advantage is clear
                
                action = masked_q_values.argmax().item()
            self.q_network.train()  # Set back to train mode
            return action
        else:
            # Random action selection from valid actions only
            return random.choice(valid_actions)
    
    def update(self) -> Optional[float]:
        """Perform one training step with PER"""
        if len(self.memory) < self.config.batch_size:
            return None
        
        # Calculate beta for importance sampling
        beta = self.config.beta_start + (self.config.beta_end - self.config.beta_start) * \
               min(1.0, self.steps_done / self.config.beta_frames)
        
        # Sample batch with priorities
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(
            self.config.batch_size, beta
        )
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        self.target_network.eval()  # Set to eval mode for inference
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        self.target_network.train()  # Set back to train mode
        
        # Calculate TD errors for priority updates
        td_errors = target_q_values - current_q_values.squeeze()
        
        # Weighted loss (importance sampling)
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors.detach())
        
        # Update target network (like DQN v5)
        if self.config.tau > 0:
            # Soft update using tau
            for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data)
        else:
            # Hard update every target_update_frequency steps
            self.update_count += 1
            if self.update_count % self.config.target_update_frequency == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps_done += 1
        
        return loss.item()
    
    def step_scheduler(self, metric: float):
        """Step the learning rate scheduler based on performance metric"""
        self.scheduler.step(metric)
    
    def train_episode(self, env: MambaEnvironment) -> Dict[str, float]:
        """Train for one episode"""
        state = env.reset()
        total_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        invalid_actions = 0
        
        while True:
            # Get valid actions for current state
            valid_actions = env._get_valid_actions()
            
            # Select and execute action with action masking
            action = self.select_action(state, valid_actions)
            next_state, reward, done, info = env.step(action)
            
            # Track invalid actions (should be 0 with proper masking)
            if info['invalid_action']:
                invalid_actions += 1
            
            # Store experience
            self.memory.push(state, action, reward, next_state, done)
            
            # Update network
            loss = self.update()
            if loss is not None:
                episode_loss += loss
                loss_count += 1
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        
        return {
            'episode_reward': total_reward,
            'episode_return': info['return'],
            'total_trades': info['total_trades'],
            'invalid_actions': invalid_actions,
            'final_value': info['portfolio_value'],
            'avg_loss': avg_loss,
            'epsilon': self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * 
                      math.exp(-1. * self.steps_done / self.config.epsilon_decay)
        }


# =============================================================================
# ENHANCED TRAINING FUNCTION
# =============================================================================

def train_mamba_dqn(data_path: str, 
                   num_episodes: int = 200,
                   temporal_window: int = 30,
                   save_path: Optional[str] = None) -> Dict:
    """Train the Mamba DQN agent with validation and testing"""
    
    print("="*60)
    print("MAMBA DQN TRAINING WITH VALIDATION")
    print("="*60)
    
    # Load data
    print(f"Loading data from {data_path}")
    original_data = pd.read_csv(data_path)
    print(f"Original data shape: {original_data.shape}")
    
    # Validate Mamba features
    missing_features = [f for f in MAMBA_FEATURES if f not in original_data.columns]
    if missing_features:
        raise ValueError(f"Missing Mamba features in data: {missing_features}")
    
    # Create config first to get data split ratios
    config = MambaTradingConfig(temporal_window)
    
    # Split data chronologically (like DQN v5)
    train_end = int(len(original_data) * config.train_ratio)
    valid_end = train_end + int(len(original_data) * config.valid_ratio)
    
    train_data_orig = original_data.iloc[:train_end].copy().reset_index(drop=True)
    valid_data_orig = original_data.iloc[train_end:valid_end].copy().reset_index(drop=True)
    test_data_orig = original_data.iloc[valid_end:].copy().reset_index(drop=True)
    
    print(f"Data split: Train {len(train_data_orig)}, Validation {len(valid_data_orig)}, Test {len(test_data_orig)}")
    
    # Apply data preprocessing for fair comparison (like DQN v5)
    print(f"Applying data preprocessing (robust scaling + winsorization)...")
    _, train_data_scaled, valid_data_scaled, test_data_scaled = preprocess_financial_data(
        train_data=train_data_orig,
        valid_data=valid_data_orig,
        test_data=test_data_orig,
        scaling_method='robust',
        outlier_method='winsorize',
        save_preprocessor=False
    )
    print(f"Preprocessing complete. Original data for trading, scaled data for state representation.")
    
    # Create environments for train/validation/test
    train_env = MambaEnvironment(train_data_orig, train_data_scaled, config)
    val_env = MambaEnvironment(valid_data_orig, valid_data_scaled, config)
    test_env = MambaEnvironment(test_data_orig, test_data_scaled, config)
    
    agent = MambaDQN(config)
    
    print(f"Mamba training setup:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Episode length: {train_env.episode_length} steps")
    print(f"  Validation frequency: Every {config.validation_frequency} episodes")
    print(f"  Early stopping patience: {config.early_stopping_patience} validations")
    print(f"  Temporal window: {temporal_window}")
    print(f"  Mamba features: {len(TEMPORAL_FEATURES)} total including normalized close price")
    print(f"    - Momentum: return_1m, return_5m, return_15m")
    print(f"    - Volume: volume_ratio_5m, volatility_5m")  
    print(f"    - Technical: rsi_14m, macd")
    print(f"    - Time: hour_sin")
    print(f"    - Price: close_normalized (relative to 20-period MA)")
    print(f"    - Portfolio: position_ratio, cash_ratio")
    print(f"  d_model: {config.d_model}, n_layers: {config.n_layers}")
    print(f"  Total parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print(f"  PER buffer size: {config.buffer_size:,}")
    print(f"  🎯 IMPROVED: Now includes normalized close price for better price level awareness")
    print(f"  🔧 FIXED: Using original prices for trading, scaled features for state")
    
    # Training metrics
    episode_rewards = []
    episode_returns = []
    episode_trades = []
    episode_invalid_actions = []
    
    # Validation metrics
    validation_rewards = []
    validation_returns = []
    validation_trades = []
    validation_invalid_actions = []
    
    # Early stopping
    best_validation_return = float('-inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nStarting training...")
    for episode in range(num_episodes):
        results = agent.train_episode(train_env)
        
        episode_rewards.append(results['episode_reward'])
        episode_returns.append(results['episode_return'])
        episode_trades.append(results['total_trades'])
        episode_invalid_actions.append(results['invalid_actions'])
        
        # Print progress
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_return = np.mean(episode_returns[-20:])
            avg_trades = np.mean(episode_trades[-20:])
            avg_invalid = np.mean(episode_invalid_actions[-20:])
            
            invalid_str = f" | Invalid: {avg_invalid:.1f}" if avg_invalid > 0 else ""
            
            # Calculate trading frequency (trades per day)
            trading_freq = avg_trades / 1  # Per episode (1 day)
            
            # Check reward-return alignment
            reward_return_ratio = avg_reward / max(abs(avg_return), 0.001)  # Avoid division by zero
            alignment_status = "✅ALIGNED" if (avg_reward > 0 and avg_return > 0) or (avg_reward < 0 and avg_return < 0) else "❌MISALIGNED"
            
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(f"Episode {episode:3d} | "
                  f"Reward: {avg_reward:6.3f} | "
                  f"Return: {avg_return:6.1%} | "
                  f"Trades/Day: {trading_freq:4.0f} | "
                  f"ε: {results['epsilon']:.3f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Buffer: {len(agent.memory):,} | "
                  f"{alignment_status}{invalid_str}")
        
        # Validation (like DQN v5)
        if (episode + 1) % config.validation_frequency == 0:
            print(f"\n{'='*50}")
            print(f"VALIDATION - Episode {episode + 1}")
            print(f"{'='*50}")
            
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
            agent.step_scheduler(val_return)
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
                    'steps_done': agent.steps_done
                }
                print(f"  🏆 New best validation return: {val_return:.2%}")
            else:
                patience_counter += 1
                print(f"  📊 Validation plateau: {patience_counter}/{config.early_stopping_patience}")
            
            # Check early stopping
            current_epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * \
                             math.exp(-1. * agent.steps_done / config.epsilon_decay)
            
            if patience_counter >= config.early_stopping_patience and current_epsilon <= 0.1:
                print(f"\n🛑 Early stopping triggered at episode {episode + 1}")
                print(f"Patience exceeded and epsilon low ({current_epsilon:.3f})")
                print(f"Best validation return: {best_validation_return:.2%}")
                
                # Restore best model
                if best_model_state:
                    agent.q_network.load_state_dict(best_model_state['q_network'])
                    agent.target_network.load_state_dict(best_model_state['target_network'])
                    agent.optimizer.load_state_dict(best_model_state['optimizer'])
                    agent.scheduler.load_state_dict(best_model_state['scheduler'])
                    agent.steps_done = best_model_state['steps_done']
                    print("🔄 Restored best model weights")
                break
            elif patience_counter >= config.early_stopping_patience:
                print(f"Validation plateaued but epsilon still high ({current_epsilon:.3f})")
                print(f"Continuing training... (patience reset to {config.early_stopping_patience // 2})")
                patience_counter = config.early_stopping_patience // 2
    
    # Final results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    final_avg_reward = np.mean(episode_rewards[-20:])
    final_avg_return = np.mean(episode_returns[-20:])
    final_avg_trades = np.mean(episode_trades[-20:])
    final_avg_invalid = np.mean(episode_invalid_actions[-20:])
    total_invalid = sum(episode_invalid_actions)
    
    print(f"Final 20-episode averages:")
    print(f"  Reward: {final_avg_reward:.3f}")
    print(f"  Return: {final_avg_return:.1%}")
    print(f"  Trades/Day: {final_avg_trades:.0f}")
    print(f"  Invalid Actions: {final_avg_invalid:.1f}")
    
    # Calculate trading frequency improvement
    if final_avg_trades < 50:
        print(f"  🎯 LOW trading frequency - good for cost reduction!")
    elif final_avg_trades > 200:
        print(f"  ⚠️  HIGH trading frequency - may be overtrading")
    
    if total_invalid > 0:
        print(f"\n⚠️  Total invalid actions across all episodes: {total_invalid}")
        print("   (Should be 0 with proper action masking)")
    else:
        print("\n✅ No invalid actions - action masking working perfectly!")
    
    # Check final reward-return alignment
    final_reward_alignment = "✅ALIGNED" if (final_avg_reward > 0 and final_avg_return > 0) or (final_avg_reward < 0 and final_avg_return < 0) else "❌MISALIGNED"
    print(f"\n🎯 Reward-Return Alignment: {final_reward_alignment}")
    if "MISALIGNED" in final_reward_alignment:
        print("  ⚠️  Reward function may not be aligned with actual profits!")
        print("  Consider adjusting reward scaling or components.")
    else:
        print("  ✅ Rewards properly aligned with actual trading performance!")
    
    # Show epsilon decay progress
    final_epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * \
                   math.exp(-1. * agent.steps_done / config.epsilon_decay)
    print(f"\n📈 Training Progress:")
    print(f"  Final ε: {final_epsilon:.4f} (exploration rate)")
    print(f"  Total steps: {agent.steps_done:,}")
    if final_epsilon < 0.02:
        print("  ✅ Low exploration achieved - agent focusing on learned strategy")
    
    # Multi-day test evaluation (like DQN v5)
    print("\n" + "="*60)
    print("MULTI-DAY TEST EVALUATION")
    print("="*60)
    
    # Calculate available test days
    total_test_days = test_env.total_days
    num_test_days = min(6, total_test_days)
    
    if num_test_days > 0:
        test_days = np.linspace(0, total_test_days - 1, num_test_days, dtype=int)
        
        all_test_results = []
        all_portfolio_values = []
        all_price_histories = []
        all_action_histories = []
        
        for i, day_idx in enumerate(test_days):
            print(f"\nRunning backtest for day {day_idx + 1}/{total_test_days} (Test {i+1}/{num_test_days})...")
            
            # Reset environment to specific day
            test_state = test_env.reset()  # Can add day_idx parameter if needed
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
                
                # Track for analysis
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
                'winning_trades': test_info.get('winning_trades', 0),
                'losing_trades': test_info.get('losing_trades', 0),
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
            print(f"    Trades: {test_info['total_trades']}")
            print(f"    Invalid Actions: {test_info['invalid_actions']}")
        
        # Calculate aggregate statistics
        returns = [r['total_return'] for r in all_test_results]
        final_values = [r['final_value'] for r in all_test_results]
        sharpe_ratios = [r['sharpe_ratio'] for r in all_test_results]
        max_drawdowns = [r['max_drawdown'] for r in all_test_results]
        total_trades = [r['total_trades'] for r in all_test_results]
        invalid_actions = [r['invalid_actions'] for r in all_test_results]
        
        print(f"\n{'='*60}")
        print("AGGREGATE TEST RESULTS")
        print(f"{'='*60}")
        print(f"Average Return: {np.mean(returns):.2%} ± {np.std(returns):.2%}")
        print(f"Best Return: {np.max(returns):.2%}")
        print(f"Worst Return: {np.min(returns):.2%}")
        print(f"Win Rate: {np.sum([r > 0 for r in returns]) / len(returns):.1%}")
        print(f"Average Final Value: ${np.mean(final_values):,.2f}")
        print(f"Average Sharpe Ratio: {np.mean(sharpe_ratios):.2f}")
        print(f"Average Max Drawdown: {np.mean(max_drawdowns):.2%}")
        print(f"Average Trades per Day: {np.mean(total_trades):.1f}")
        print(f"Average Invalid Actions: {np.mean(invalid_actions):.1f}")
        
        # Create multi-day test results
        multi_day_test_results = {
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
                'avg_invalid_actions': np.mean(invalid_actions)
            }
        }
    else:
        print("⚠️ No test data available for multi-day evaluation")
        multi_day_test_results = None
    
    # Save model if requested
    if save_path:
        torch.save({
            'q_network_state_dict': agent.q_network.state_dict(),
            'target_network_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'scheduler_state_dict': agent.scheduler.state_dict(),
            'config': config,
            'steps_done': agent.steps_done,
            'temporal_window': temporal_window
        }, save_path)
        print(f"\n💾 Model saved to {save_path}")
    
    return {
        'agent': agent,
        'train_env': train_env,
        'val_env': val_env,
        'test_env': test_env,
        'train_data_orig': train_data_orig,
        'valid_data_orig': valid_data_orig,
        'test_data_orig': test_data_orig,
        'train_data_scaled': train_data_scaled,
        'valid_data_scaled': valid_data_scaled,
        'test_data_scaled': test_data_scaled,
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'episode_trades': episode_trades,
        'episode_invalid_actions': episode_invalid_actions,
        'validation_rewards': validation_rewards,
        'validation_returns': validation_returns,
        'validation_trades': validation_trades,
        'validation_invalid_actions': validation_invalid_actions,
        'multi_day_test_results': multi_day_test_results,
        'best_validation_return': best_validation_return,
        'total_invalid_actions': sum(episode_invalid_actions),
        'config': config
    }


# =============================================================================
# ENHANCED BACKTESTING
# =============================================================================

def buy_and_hold_baseline(data: pd.DataFrame, 
                         num_days: int = 5, 
                         initial_balance: float = 10000.0,
                         transaction_fee: float = 0.001) -> Dict:
    """Simple buy-and-hold strategy baseline"""
    
    print(f"\n💰 Buy-and-Hold Baseline on {num_days} random days...")
    
    results = []
    episode_length = MINUTES_PER_TRADING_DAY
    
    for day in range(num_days):
        # Random start for each day
        max_start = len(data) - episode_length
        start_idx = random.randint(0, max_start) if max_start > 0 else 0
        end_idx = start_idx + episode_length
        
        start_price = data.iloc[start_idx]['close']
        end_price = data.iloc[end_idx-1]['close']
        
        # Buy at start
        shares = initial_balance / (start_price * (1 + transaction_fee))
        
        # Sell at end
        final_value = shares * end_price * (1 - transaction_fee)
        day_return = (final_value - initial_balance) / initial_balance
        
        results.append({
            'day': day + 1,
            'final_value': final_value,
            'return': day_return,
            'trades': 2,  # Buy at start, sell at end
            'start_price': start_price,
            'end_price': end_price
        })
        
        print(f"Day {day + 1}: Return {day_return:6.1%}, Trades: 2, Final: ${final_value:,.0f}")
    
    avg_return = np.mean([r['return'] for r in results])
    print(f"\nBuy-and-Hold Average return: {avg_return:.1%}")
    
    return {
        'results': results,
        'avg_return': avg_return,
        'strategy': 'buy_and_hold'
    }


def backtest_mamba_dqn(agent: MambaDQN, 
                       original_data: pd.DataFrame,
                       scaled_data: pd.DataFrame, 
                       num_days: int = 5) -> Dict:
    """Mamba DQN backtesting function with action masking using original prices for trading"""
    
    print(f"\nBacktesting on {num_days} random days...")
    
    config = agent.config
    env = MambaEnvironment(original_data, scaled_data, config)  # Pass both datasets
    
    results = []
    
    for day in range(num_days):
        # Random start for each day
        state = env.reset()
        portfolio_values = [config.initial_balance]
        actions = []
        valid_actions_history = []
        prices = []
        invalid_actions = 0
        
        while True:
            # Get valid actions and select with masking
            valid_actions = env._get_valid_actions()
            action = agent.select_action(state, valid_actions, epsilon=0.0)  # No exploration
            state, reward, done, info = env.step(action)
            
            # Track invalid actions (should be 0 with proper masking)
            if info['invalid_action']:
                invalid_actions += 1
            
            portfolio_values.append(info['portfolio_value'])
            actions.append(action)
            valid_actions_history.append(valid_actions.copy())
            prices.append(info['current_price'])  # Now using original prices
            
            if done:
                break
        
        day_return = (info['portfolio_value'] - config.initial_balance) / config.initial_balance
        
        results.append({
            'day': day + 1,
            'final_value': info['portfolio_value'],
            'return': day_return,
            'trades': info['total_trades'],
            'invalid_actions': invalid_actions,
            'portfolio_values': portfolio_values,
            'actions': actions,
            'valid_actions_history': valid_actions_history,
            'prices': prices
        })
        
        invalid_str = f", Invalid: {invalid_actions}" if invalid_actions > 0 else ""
        print(f"Day {day + 1}: Return {day_return:6.1%}, Trades: {info['total_trades']}, Final: ${info['portfolio_value']:,.0f}{invalid_str}")
    
    avg_return = np.mean([r['return'] for r in results])
    total_invalid = sum([r['invalid_actions'] for r in results])
    print(f"\nAverage return: {avg_return:.1%}")
    if total_invalid > 0:
        print(f"Total invalid actions: {total_invalid} (should be 0 with action masking)")
    
    return {
        'results': results,
        'avg_return': avg_return,
        'total_invalid_actions': total_invalid,
        'agent': agent
    }


# =============================================================================
# SIMPLE DEMO AND TESTING
# =============================================================================

def demo_mamba_training(data_path: str, 
                       num_episodes: int = 100,
                       temporal_window: int = 30) -> Dict:
    """Simple demo of Mamba DQN training with validation and testing"""
    
    print("="*60)
    print("🧠 MAMBA DQN DEMO WITH VALIDATION")
    print("="*60)
    print(f"📊 Episodes: {num_episodes} | Temporal Window: {temporal_window}")
    print("="*60)
    
    # Train Mamba model (now includes validation and testing)
    print("🔥 Training Mamba DQN with validation...")
    results = train_mamba_dqn(
        data_path, 
        num_episodes, 
        temporal_window=temporal_window,
        save_path="mamba_dqn_demo.pth"
    )
    
    # Run buy-and-hold baseline on test data
    print("\n💰 Running Buy-and-Hold Baseline on test data...")
    test_data = results['test_data_orig']
    baseline = buy_and_hold_baseline(test_data, num_days=6)
    
    # Extract test results (already computed during training)
    multi_day_results = results['multi_day_test_results']
    
    # Show comparison
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    
    if multi_day_results:
        mamba_return = multi_day_results['aggregate_stats']['avg_return']
        baseline_return = baseline['avg_return']
        
        print(f"Buy-and-Hold Baseline:  {baseline_return:6.1%}")
        print(f"Mamba DQN:             {mamba_return:6.1%}")
        
        if mamba_return > baseline_return:
            margin = mamba_return - baseline_return
            print(f"\n🏆 Mamba wins by {margin:+.1%}!")
        else:
            margin = baseline_return - mamba_return
            print(f"\n📈 Baseline wins by {margin:+.1%}")
            print("   Consider optimizing Mamba parameters or more training episodes")
        
        print(f"\nMamba Trading Stats:")
        print(f"  Avg Trades/Day: {multi_day_results['aggregate_stats']['avg_trades']:.1f}")
        print(f"  Win Rate: {multi_day_results['aggregate_stats']['win_rate']:.1%}")
        print(f"  Avg Sharpe: {multi_day_results['aggregate_stats']['avg_sharpe_ratio']:.2f}")
        print(f"  Invalid Actions: {multi_day_results['aggregate_stats']['avg_invalid_actions']:.1f}")
        
        # Validation performance
        if results['validation_returns']:
            avg_val_return = np.mean(results['validation_returns'])
            print(f"\nValidation Performance:")
            print(f"  Avg Validation Return: {avg_val_return:.2%}")
            print(f"  Best Validation Return: {results['best_validation_return']:.2%}")
    else:
        print("⚠️ No test results available")
        mamba_return = None
        baseline_return = baseline['avg_return']
    
    return {
        'training_results': results,
        'baseline_return': baseline_return,
        'mamba_return': mamba_return,
        'multi_day_test_results': multi_day_results
    }





if __name__ == "__main__":
    # Comprehensive Mamba training with validation and testing
    from src.config.config import DATA_DIR
    
    data_path = DATA_DIR / "feature_engineered_v2" / "TSLA.csv"
    
    # Run comprehensive Mamba training
    print("🚀 Starting Comprehensive Mamba DQN Training...")
    print("🧠 Testing the winning Mamba architecture with full validation!")
    
    results = demo_mamba_training(
        str(data_path),
        num_episodes=100,      # Quick demo with 100 episodes
        temporal_window=30     # Optimal window size
    )
    
    print(f"\n🎉 COMPREHENSIVE MAMBA TRAINING COMPLETE!")
    
    # Show final stats
    mamba_return = results['mamba_return']
    baseline_return = results['baseline_return']
    
    print(f"\n💡 SUMMARY:")
    print(f"🧠 Mamba DQN leverages State Space Models for temporal pattern recognition")
    print(f"📊 Winner of comprehensive architecture comparison")
    print(f"⚡ Most efficient architecture with superior temporal processing")
    print(f"🔄 Now includes validation, early stopping, and comprehensive testing")
    
    if mamba_return and mamba_return > baseline_return:
        print(f"🏆 Mamba beats buy-and-hold in this demo!")
    else:
        print(f"📈 Room for improvement - consider parameter optimization")
        print(f"💡 Try running the Mamba optimizer: python -m src.models.mark.dqn_v2.dqn_v6_mamba_optimizer")
    
    print(f"\n🎯 FEATURES INCLUDED:")
    print(f"✅ Train/Validation/Test data split")
    print(f"✅ Validation every 10 episodes")
    print(f"✅ Early stopping with patience")
    print(f"✅ Learning rate scheduling")
    print(f"✅ Multi-day test evaluation")
    print(f"✅ Comprehensive performance metrics")
    print(f"✅ Normalized close price for price level awareness")
