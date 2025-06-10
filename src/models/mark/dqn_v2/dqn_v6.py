"""
DQN v6: Enhanced Minimal Trading Agent with Dueling Architecture
===============================================================

A simplified but powerful DQN implementation with:
- Small state space (7 features vs 156) 
- GPU-based Prioritized Experience Replay for better learning
- Three architecture options with Dueling DQN:
  * Simple MLP (single row features)
  * Temporal CNN (2D temporal data)
  * Mamba SSM (state space model)
- Clean reward function (pure P&L + transaction costs)
- Easy comparison between approaches

Key Improvements:
- Dueling architecture separates V(s) and A(s,a) for better learning
- GPU-optimized PER for faster training
- Mamba-inspired SSM for efficient temporal modeling

This serves as a strong baseline with modern RL techniques.
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
from typing import Dict, Tuple, Optional
from pathlib import Path

from src.config.config import DEVICE, MINUTES_PER_TRADING_DAY


# =============================================================================
# MINIMAL FEATURE SET (7 features total)
# =============================================================================

MINIMAL_FEATURES = [
    'close',                    # Current price (for normalization - not used directly)
    'return_1m',               # 1-minute price momentum  
    'return_5m',               # 5-minute price momentum
    'return_15m',              # 15-minute price momentum
    'volume_ratio_5m',         # Volume relative to recent average
    'volatility_5m',           # Recent volatility measure
    'hour_sin',                # Time of day (cyclical)
]

# Features actually used in state (excluding 'close')
ACTUAL_FEATURES = [f for f in MINIMAL_FEATURES if f != 'close']

# For temporal CNN - these features work well in 2D format
TEMPORAL_FEATURES = [
    'return_1m', 'return_5m', 'return_15m',
    'volume_ratio_5m', 'volatility_5m', 
    'rsi_14m', 'macd'  # Add a couple technical indicators
]


# =============================================================================
# ENHANCED CONFIGURATION
# =============================================================================

class EnhancedTradingConfig:
    """Enhanced configuration with PER and temporal options"""
    def __init__(self, architecture: str = "mlp", temporal_window: int = 20):
        # Trading parameters
        self.initial_balance = 10000.0
        self.transaction_fee_percent = 0.001  # 0.1% (realistic for retail)
        self.max_position_size = 0.95  # Use 95% of balance max
        
        # Architecture choice: "mlp", "cnn", or "mamba"
        self.architecture = architecture
        self.temporal_window = temporal_window  # Number of minutes to look back
        
        # Backward compatibility
        self.use_temporal_cnn = architecture in ["cnn", "mamba"]
        
        # RL parameters  
        self.num_actions = 3  # Hold, Buy, Sell
        
        if architecture == "mlp":
            self.state_size = len(ACTUAL_FEATURES) + 2  # actual features + position + cash_ratio
        else:  # cnn or mamba
            self.state_size = (len(TEMPORAL_FEATURES), temporal_window)  # 2D: (features, time)
            self.input_channels = len(TEMPORAL_FEATURES)
        
        # Network parameters
        if architecture == "mlp":
            self.hidden_size = 64
        elif architecture == "cnn":
            self.hidden_size = 128
        else:  # mamba
            self.hidden_size = 64  # d_model for Mamba
        
        self.learning_rate = 1e-3
        
        # Prioritized Experience Replay parameters
        self.buffer_size = 50000  # Larger for PER
        self.batch_size = 64
        self.alpha = 0.6  # PER exponent
        self.beta_start = 0.4  # Importance sampling
        self.beta_end = 1.0
        self.beta_frames = 100000
        
        # Training parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 10000  
        self.target_update = 500   # Less frequent updates for stability
        self.gamma = 0.99


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
        if isinstance(state_shape, tuple):  # 2D state (temporal CNN)
            self.states = torch.zeros((self.capacity, *state_shape), device=self.device, dtype=torch.float32)
            self.next_states = torch.zeros((self.capacity, *state_shape), device=self.device, dtype=torch.float32)
        else:  # 1D state (MLP)
            self.states = torch.zeros((self.capacity, state_shape), device=self.device, dtype=torch.float32)
            self.next_states = torch.zeros((self.capacity, state_shape), device=self.device, dtype=torch.float32)
            
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


class MambaStyleQNetwork(nn.Module):
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
# ENHANCED NEURAL NETWORKS
# =============================================================================

class SimpleQNetwork(nn.Module):
    """Simple MLP with Dueling Architecture for single-row features"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        
        self.action_size = action_size
        
        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Dueling streams
        self.value_stream = nn.Linear(hidden_size, 1)  # V(s) - state value
        self.advantage_stream = nn.Linear(hidden_size, action_size)  # A(s,a) - action advantages
        
    def forward(self, x):
        features = self.feature_layer(x)
        
        # Separate value and advantage
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_size)
        
        # Dueling combination: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class TemporalCNNQNetwork(nn.Module):
    """CNN with Dueling Architecture for temporal 2D data"""
    
    def __init__(self, input_channels: int, temporal_window: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        
        self.action_size = action_size
        
        # CNN layers for temporal pattern extraction
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=7, padding=3)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
        
        # Calculate flattened size
        self.flattened_size = 64 * 8  # 64 channels * 8 pooled temporal dimension
        
        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(self.flattened_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Dueling streams
        self.value_stream = nn.Linear(hidden_size, 1)  # V(s) - state value
        self.advantage_stream = nn.Linear(hidden_size, action_size)  # A(s,a) - action advantages
        
    def forward(self, x):
        # x shape: (batch, features, time) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global pooling
        x = self.adaptive_pool(x)
        
        # Flatten and extract features
        x = x.view(x.size(0), -1)
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

class EnhancedEnvironment:
    """Enhanced environment supporting both single-row and temporal features"""
    
    def __init__(self, data: pd.DataFrame, config: EnhancedTradingConfig):
        self.data = data
        self.config = config
        
        # Validate required features
        if config.architecture == "mlp":
            required_features = MINIMAL_FEATURES  # Need 'close' for price, but use ACTUAL_FEATURES for state
        else:
            required_features = TEMPORAL_FEATURES
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Episode parameters
        self.total_steps = len(data) 
        self.episode_length = MINUTES_PER_TRADING_DAY  # One trading day
        
        # State tracking
        self.reset()
    
    def reset(self, start_step: Optional[int] = None) -> np.ndarray:
        """Reset environment to start of episode"""
        # Random start if not specified (for training diversity)
        if start_step is None:
            min_start = self.config.temporal_window if self.config.architecture != "mlp" else 0
            max_start = self.total_steps - self.episode_length - min_start
            self.current_step = random.randint(min_start, max_start) if max_start > min_start else min_start
        else:
            self.current_step = max(start_step, self.config.temporal_window if self.config.architecture != "mlp" else 0)
            
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
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step >= len(self.data):
            # Return neutral state if out of bounds
            if self.config.architecture == "mlp":
                return np.zeros(self.config.state_size)
            else:
                return np.zeros(self.config.state_size)
        
        if self.config.architecture == "mlp":
            return self._get_single_row_state()
        else:  # cnn or mamba
            return self._get_temporal_state()
    
    def _get_single_row_state(self) -> np.ndarray:
        """Get single-row state (like original v6)"""
        row = self.data.iloc[self.current_step]
        current_price = row['close']
        
        # Extract actual features (excluding 'close')
        features = []
        for feature in ACTUAL_FEATURES:
            value = row.get(feature, 0.0)
            if pd.isna(value):
                value = 0.0
            features.append(float(value))
        
        # Add position information
        current_value = self.balance + (self.position * current_price)
        position_ratio = (self.position * current_price) / current_value if current_value > 0 else 0.0
        cash_ratio = self.balance / current_value if current_value > 0 else 1.0
        
        features.extend([position_ratio, cash_ratio])
        
        return np.array(features, dtype=np.float32)
    
    def _get_temporal_state(self) -> np.ndarray:
        """Get temporal window state for CNN"""
        # Get temporal window
        start_idx = max(0, self.current_step - self.config.temporal_window + 1)
        end_idx = self.current_step + 1
        
        # Extract temporal features
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Create 2D array (features x time)
        temporal_state = np.zeros((len(TEMPORAL_FEATURES), self.config.temporal_window))
        
        for i, feature in enumerate(TEMPORAL_FEATURES):
            if feature in window_data.columns:
                values = window_data[feature].fillna(0.0).values
                # Pad if necessary
                if len(values) < self.config.temporal_window:
                    padded_values = np.zeros(self.config.temporal_window)
                    padded_values[-len(values):] = values
                    values = padded_values
                temporal_state[i] = values
        
        return temporal_state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        if self.current_step >= len(self.data):
            return self._get_state(), 0.0, True, {}
            
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0.0
        trade_executed = False
        
        # Execute action (same logic as before)
        if action == 1 and self.position == 0:  # Buy (only if no position)
            position_value = self.balance * self.config.max_position_size
            shares_to_buy = position_value / current_price
            cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)
            
            if cost <= self.balance:
                self.position = shares_to_buy
                self.balance -= cost
                self.entry_price = current_price
                trade_executed = True
                self.total_trades += 1
                
        elif action == 2 and self.position > 0:  # Sell (only if holding position)
            revenue = self.position * current_price * (1 - self.config.transaction_fee_percent)
            cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
            profit = revenue - cost_basis
            
            self.balance += revenue
            self.position = 0.0
            trade_executed = True
            self.total_trades += 1
            
            # Track P&L
            if profit > 0:
                self.total_profit += profit
            else:
                self.total_loss += abs(profit)
                
            # Simple reward: actual profit/loss
            reward = profit / self.config.initial_balance  # Normalize by initial balance
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.episode_end
        
        # Force close position at end of episode
        if done and self.position > 0:
            final_price = self.data.iloc[min(self.current_step, len(self.data) - 1)]['close']
            revenue = self.position * final_price * (1 - self.config.transaction_fee_percent)
            cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
            final_profit = revenue - cost_basis
            
            self.balance += revenue
            self.position = 0.0
            
            # Add final P&L to reward
            reward += final_profit / self.config.initial_balance
            
            if final_profit > 0:
                self.total_profit += final_profit
            else:
                self.total_loss += abs(final_profit)
        
        # Get next state
        next_state = self._get_state()
        
        # Info for tracking
        current_value = self.balance + (self.position * current_price)
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price,
            'portfolio_value': current_value,
            'trade_executed': trade_executed,
            'total_trades': self.total_trades,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'return': (current_value - self.config.initial_balance) / self.config.initial_balance
        }
        
        return next_state, reward, done, info


# =============================================================================
# ENHANCED DQN AGENT
# =============================================================================

class EnhancedDQN:
    """Enhanced DQN with PER and temporal CNN support"""
    
    def __init__(self, config: EnhancedTradingConfig, device: torch.device = DEVICE):
        self.config = config
        self.device = device
        
        # Networks - choose architecture
        if config.architecture == "mlp":
            self.q_network = SimpleQNetwork(config.state_size, config.num_actions, config.hidden_size).to(device)
            self.target_network = SimpleQNetwork(config.state_size, config.num_actions, config.hidden_size).to(device)
        elif config.architecture == "cnn":
            self.q_network = TemporalCNNQNetwork(
                config.input_channels, 
                config.temporal_window, 
                config.num_actions, 
                config.hidden_size
            ).to(device)
            self.target_network = TemporalCNNQNetwork(
                config.input_channels, 
                config.temporal_window, 
                config.num_actions, 
                config.hidden_size
            ).to(device)
        elif config.architecture == "mamba":
            self.q_network = MambaStyleQNetwork(
                config.input_channels,
                config.temporal_window,
                config.num_actions,
                config.hidden_size
            ).to(device)
            self.target_network = MambaStyleQNetwork(
                config.input_channels,
                config.temporal_window,
                config.num_actions,
                config.hidden_size
            ).to(device)
        else:
            raise ValueError(f"Unknown architecture: {config.architecture}")        
        
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBufferGPU(config.buffer_size, config.alpha, device)
        
        # Training state
        self.steps_done = 0
        
        network_type = {"mlp": "Dueling MLP", "cnn": "Dueling CNN", "mamba": "Dueling Mamba SSM"}[config.architecture]
        print(f"Enhanced DQN initialized ({network_type}):")
        print(f"  State size: {config.state_size}")
        print(f"  Action size: {config.num_actions}")
        print(f"  Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
        if config.architecture != "mlp":
            print(f"  Temporal window: {config.temporal_window}")
            print(f"  Input channels: {config.input_channels}")
    
    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                     math.exp(-1. * self.steps_done / self.config.epsilon_decay)
        
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.config.num_actions)
    
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
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)
        
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
        
        # Update target network
        if self.steps_done % self.config.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps_done += 1
        
        return loss.item()
    
    def train_episode(self, env: EnhancedEnvironment) -> Dict[str, float]:
        """Train for one episode"""
        state = env.reset()
        total_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        
        while True:
            # Select and execute action
            action = self.select_action(state)
            next_state, reward, done, info = env.step(action)
            
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
            'final_value': info['portfolio_value'],
            'avg_loss': avg_loss,
            'epsilon': self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * 
                      math.exp(-1. * self.steps_done / self.config.epsilon_decay)
        }


# =============================================================================
# ENHANCED TRAINING FUNCTION
# =============================================================================

def train_enhanced_dqn(data_path: str, 
                      num_episodes: int = 200,
                      architecture: str = "mlp",
                      temporal_window: int = 20,
                      save_path: Optional[str] = None) -> Dict:
    """Train the enhanced DQN agent"""
    
    arch_names = {"mlp": "Dueling MLP", "cnn": "Dueling CNN", "mamba": "Dueling Mamba SSM"}
    print("="*60)
    print(f"ENHANCED DQN TRAINING ({arch_names.get(architecture, architecture)})")
    print("="*60)
    
    # Load data
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    print(f"Data shape: {data.shape}")
    
    # Validate features
    if architecture == "mlp":
        required_features = MINIMAL_FEATURES  # Need 'close' for price, but use ACTUAL_FEATURES for state
    else:
        required_features = TEMPORAL_FEATURES
    missing_features = [f for f in required_features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    # Create config and environment
    config = EnhancedTradingConfig(architecture, temporal_window)
    env = EnhancedEnvironment(data, config)
    agent = EnhancedDQN(config)
    
    print(f"Training setup:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Episode length: {env.episode_length} steps")
    if architecture != "mlp":
        print(f"  Temporal window: {temporal_window}")
        print(f"  Temporal features: {TEMPORAL_FEATURES}")
    else:
        print(f"  State features: {ACTUAL_FEATURES} + [position_ratio, cash_ratio]")
    print(f"  Total parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print(f"  PER buffer size: {config.buffer_size:,}")
    
    # Training loop
    episode_rewards = []
    episode_returns = []
    episode_trades = []
    
    print(f"\nStarting training...")
    for episode in range(num_episodes):
        results = agent.train_episode(env)
        
        episode_rewards.append(results['episode_reward'])
        episode_returns.append(results['episode_return'])
        episode_trades.append(results['total_trades'])
        
        # Print progress
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_return = np.mean(episode_returns[-20:])
            avg_trades = np.mean(episode_trades[-20:])
            
            print(f"Episode {episode:3d} | "
                  f"Reward: {avg_reward:6.3f} | "
                  f"Return: {avg_return:6.1%} | "
                  f"Trades: {avg_trades:4.1f} | "
                  f"ε: {results['epsilon']:.3f} | "
                  f"Buffer: {len(agent.memory):,}")
    
    # Final results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    final_avg_reward = np.mean(episode_rewards[-20:])
    final_avg_return = np.mean(episode_returns[-20:])
    final_avg_trades = np.mean(episode_trades[-20:])
    
    print(f"Final 20-episode averages:")
    print(f"  Reward: {final_avg_reward:.3f}")
    print(f"  Return: {final_avg_return:.1%}")
    print(f"  Trades: {final_avg_trades:.1f}")
    
    # Save model if requested
    if save_path:
        torch.save({
            'q_network_state_dict': agent.q_network.state_dict(),
            'target_network_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'config': config,
            'steps_done': agent.steps_done,
            'architecture': architecture,
            'temporal_window': temporal_window
        }, save_path)
        print(f"Model saved to {save_path}")
    
    return {
        'agent': agent,
        'env': env,
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'episode_trades': episode_trades,
        'config': config
    }


# =============================================================================
# ENHANCED BACKTESTING
# =============================================================================

def backtest_enhanced_dqn(agent: EnhancedDQN, 
                         data: pd.DataFrame, 
                         num_days: int = 5) -> Dict:
    """Enhanced backtesting function"""
    
    print(f"\nBacktesting on {num_days} random days...")
    
    config = agent.config
    env = EnhancedEnvironment(data, config)
    
    results = []
    
    for day in range(num_days):
        # Random start for each day
        state = env.reset()
        portfolio_values = [config.initial_balance]
        actions = []
        prices = []
        
        while True:
            action = agent.select_action(state, epsilon=0.0)  # No exploration
            state, reward, done, info = env.step(action)
            
            portfolio_values.append(info['portfolio_value'])
            actions.append(action)
            prices.append(info['current_price'])
            
            if done:
                break
        
        day_return = (info['portfolio_value'] - config.initial_balance) / config.initial_balance
        
        results.append({
            'day': day + 1,
            'final_value': info['portfolio_value'],
            'return': day_return,
            'trades': info['total_trades'],
            'portfolio_values': portfolio_values,
            'actions': actions,
            'prices': prices
        })
        
        print(f"Day {day + 1}: Return {day_return:6.1%}, Trades: {info['total_trades']}, Final: ${info['portfolio_value']:,.0f}")
    
    avg_return = np.mean([r['return'] for r in results])
    print(f"\nAverage return: {avg_return:.1%}")
    
    return {
        'results': results,
        'avg_return': avg_return,
        'agent': agent
    }


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_all_architectures(data_path: str, 
                             num_episodes: int = 100,
                             temporal_window: int = 30) -> Dict:
    """Compare MLP vs CNN vs Mamba architectures"""
    
    print("="*80)
    print("DUELING ARCHITECTURE SHOWDOWN: MLP vs CNN vs Mamba")
    print("="*80)
    
    results = {}
    architectures = ["mlp", "cnn", "mamba"]
    arch_names = {"mlp": "Dueling MLP", "cnn": "Dueling CNN", "mamba": "Dueling Mamba SSM"}
    
    # Train all architectures
    for arch in architectures:
        print(f"\n🔥 Training {arch_names[arch]}...")
        arch_results = train_enhanced_dqn(
            data_path, 
            num_episodes, 
            architecture=arch,
            temporal_window=temporal_window,
            save_path=f"dqn_v6_{arch}.pth"
        )
        results[arch] = arch_results
    
    # Compare final performance
    print("\n" + "="*80)
    print("🏁 FINAL RESULTS")
    print("="*80)
    
    performance = {}
    for arch in architectures:
        arch_results = results[arch]
        performance[arch] = {
            'return': np.mean(arch_results['episode_returns'][-20:]),
            'trades': np.mean(arch_results['episode_trades'][-20:]),
            'params': sum(p.numel() for p in arch_results['agent'].q_network.parameters())
        }
        
        print(f"{arch_names[arch]}:")
        print(f"  Final Return: {performance[arch]['return']:6.1%}")
        print(f"  Avg Trades:   {performance[arch]['trades']:6.1f}")
        print(f"  Parameters:   {performance[arch]['params']:,}")
        print()
    
    # Find winner
    best_arch = max(architectures, key=lambda x: performance[x]['return'])
    best_return = performance[best_arch]['return']
    
    print(f"🏆 WINNER: {arch_names[best_arch]} ({best_return:.1%})")
    
    # Show efficiency (return per parameter)
    print(f"\n📊 EFFICIENCY (Return/1K params):")
    for arch in architectures:
        efficiency = performance[arch]['return'] / (performance[arch]['params'] / 1000)
        print(f"  {arch_names[arch]}: {efficiency:.2f}")
    
    return results


def compare_architectures(data_path: str, 
                         num_episodes: int = 100,
                         temporal_window: int = 30) -> Dict:
    """Compare MLP vs CNN architectures (backward compatibility)"""
    
    print("="*80)
    print("DUELING ARCHITECTURE COMPARISON: MLP vs CNN")
    print("="*80)
    
    results = {}
    
    # Train MLP version
    print("\n🔥 Training Simple MLP...")
    mlp_results = train_enhanced_dqn(
        data_path, 
        num_episodes, 
        architecture="mlp",
        save_path="dqn_v6_mlp.pth"
    )
    results['mlp'] = mlp_results
    
    # Train CNN version
    print("\n🔥 Training Temporal CNN...")
    cnn_results = train_enhanced_dqn(
        data_path, 
        num_episodes, 
        architecture="cnn",
        temporal_window=temporal_window,
        save_path="dqn_v6_cnn.pth"
    )
    results['cnn'] = cnn_results
    
    # Compare final performance
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    mlp_return = np.mean(mlp_results['episode_returns'][-20:])
    cnn_return = np.mean(cnn_results['episode_returns'][-20:])
    
    mlp_trades = np.mean(mlp_results['episode_trades'][-20:])
    cnn_trades = np.mean(cnn_results['episode_trades'][-20:])
    
    print(f"Dueling MLP:")
    print(f"  Final Return: {mlp_return:6.1%}")
    print(f"  Avg Trades:   {mlp_trades:6.1f}")
    print(f"  Parameters:   {sum(p.numel() for p in mlp_results['agent'].q_network.parameters()):,}")
    
    print(f"\nDueling CNN:")
    print(f"  Final Return: {cnn_return:6.1%}")
    print(f"  Avg Trades:   {cnn_trades:6.1f}")
    print(f"  Parameters:   {sum(p.numel() for p in cnn_results['agent'].q_network.parameters()):,}")
    
    winner = "CNN" if cnn_return > mlp_return else "MLP"
    margin = abs(cnn_return - mlp_return)
    print(f"\n🏆 Winner: {winner} (by {margin:.1%})")
    
    return results


if __name__ == "__main__":
    # Example usage - compare all three architectures
    from src.config.config import DATA_DIR
    
    data_path = DATA_DIR / "feature_engineered_v2" / "TSLA.csv"
    
    # Architecture showdown!
    results = compare_all_architectures(str(data_path), num_episodes=50)
    
    # Backtest all three
    data = pd.read_csv(data_path)
    
    print("\n🔍 Backtesting MLP...")
    mlp_backtest = backtest_enhanced_dqn(results['mlp']['agent'], data, num_days=3)
    
    print("\n🔍 Backtesting CNN...")
    cnn_backtest = backtest_enhanced_dqn(results['cnn']['agent'], data, num_days=3)
    
    print("\n🔍 Backtesting Mamba...")
    mamba_backtest = backtest_enhanced_dqn(results['mamba']['agent'], data, num_days=3)
