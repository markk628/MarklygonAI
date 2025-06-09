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
)
from src.utils.utils import create_directory
from src.web.models import app, db, BacktestHistory, ModelType, MarklygonModel


@dataclass
class TradingConfig:
    """Configuration for the DQN agent"""
    # Environment parameters
    initial_balance: float = INITIAL_BALANCE
    transaction_fee_percent: float = TRANSACTION_FEE_PERCENT
    window_size: int = WINDOW_SIZE
    num_stock_features: int = len(STOCK_FEATURES_V2)  # Use actual length from config
    num_portfolio_features: int = 17  # Expanded from 12 to 17 for complete state
    num_features: int = len(STOCK_FEATURES_V2) + 17  # Stock features + portfolio features
    num_actions: int = 3  # Hold, Buy, Sell
    max_position_size: float = MAX_POSITION_SIZE
    
    # Network architecture selection
    architecture_type: str = "improved"  # "original", "improved", "hybrid"
    
    # Network parameters
    hidden_size: int = 512
    learning_rate: float = 1e-4
    
    # Training parameters
    batch_size: int = BATCH_SIZE
    gamma: float = 0.99
    tau: float = 0.005
    update_frequency: int = TRAIN_INTERVAL
    target_update_frequency: int = UPDATE_TARGET_EVERY
    
    # Experience replay
    buffer_size: int = REPLAY_BUFFER_SIZE
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 175000
    
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
        
        # Enhanced portfolio branch with risk awareness
        self.portfolio_branch = nn.Sequential(
            nn.Linear(config.num_portfolio_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),  # Better activation for financial data
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
            nn.GroupNorm(4, 64),  # Better than BatchNorm for financial data
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
                features = cnn_branch(timestep_data.transpose(1, 2))  # (batch, 64, 1)
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


def create_network(config: TradingConfig) -> nn.Module:
    """Factory function to create network based on config"""
    if config.architecture_type == "original":
        return DuelingNetworkOriginal(config)
    elif config.architecture_type == "improved":
        return ImprovedDuelingNetwork(config)
    elif config.architecture_type == "hybrid":
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
        
        # Pre-allocate GPU tensors for the buffer
        self.states = torch.zeros((capacity, config.window_size, config.num_features), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, config.window_size, config.num_features), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        # Priority management - initialize with small positive values
        self.priorities = torch.ones(capacity, dtype=torch.float32, device=device) * 0.01
        self.max_priority = 1.0
        
    def push(self, 
             state: torch.Tensor, 
             action: int, 
             reward: float, 
             next_state: torch.Tensor, 
             done: bool):
        """save experience"""
        # Ensure tensors are on the correct device
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
            # Ensure we don't go beyond available complete days
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
        morning_session = 1.0 if minutes_into_day < 120 else 0.0  # First 2 hours (9:30-11:30)
        midday_session = 1.0 if 120 <= minutes_into_day < 270 else 0.0  # Middle 2.5 hours (11:30-2:00)
        afternoon_session = 1.0 if minutes_into_day >= 270 else 0.0  # Last 2 hours (2:00-4:00)
        
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
        drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0
        
        # Trading activity metrics
        trade_frequency = self.total_trades / max(1, minutes_into_day / 60)  # Trades per hour
        win_rate = self.winning_trades / max(1, self.total_trades)
        
        # Calculate average profit/loss per trade
        avg_profit_per_winning_trade = self.total_profit / max(1, self.winning_trades)
        avg_loss_per_losing_trade = abs(self.total_loss) / max(1, self.losing_trades)
        profit_loss_ratio = avg_profit_per_winning_trade / max(0.001, avg_loss_per_losing_trade)
        
        # Action sequence features
        normalized_invalid_actions = self.invalid_actions / max(1, self.current_step - self.episode_start)
        consecutive_invalid_penalty = min(self.consecutive_invalid_actions / 5.0, 1.0)
        
        # Position flag
        has_position_flag = 1.0 if self.position > 0 else 0.0
        
        # ✅ Portfolio features: Complete state representation (17 features)
        portfolio_features = [
            # Core metrics (4)
            normalized_balance,
            normalized_position,
            normalized_portfolio_value,
            position_ratio,
            
            # Performance metrics (4)
            self.unrealized_pnl,
            drawdown,
            win_rate,
            profit_loss_ratio,
            
            # Timing & activity (4)
            time_of_day_normalized,
            normalized_holding_time,
            trade_frequency,
            has_position_flag,
            
            # Market session indicators (3)
            morning_session,
            midday_session,
            afternoon_session,
            
            # Mistake tracking (2)  
            normalized_invalid_actions,
            consecutive_invalid_penalty
        ]
        
        # Replace any non-finite values with 0
        portfolio_features = [x if np.isfinite(x) else 0.0 for x in portfolio_features]
        
        # Create portfolio state vector
        portfolio_state = torch.tensor(np.array(portfolio_features), dtype=torch.float32, device=self.device)
        
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
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
            
        current_price = self.data.iloc[self.current_step]['close']        
        reward = 0
        trade_executed = False
        invalid_action = self._is_invalid_action(action)
        
        # Calculate market session for context
        minutes_into_day = (self.current_step - self.episode_start) % self.minutes_per_day
        is_near_close = minutes_into_day >= (self.minutes_per_day - 30)  # Last 30 minutes
        
        if invalid_action:
            self.invalid_actions += 1
            self.consecutive_invalid_actions += 1
            # Escalating penalty for consecutive invalid actions
            penalty = 0.01 + (self.consecutive_invalid_actions * 0.005)
            reward = -min(penalty, 0.05)  # Cap penalty at -0.05
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
                
                # Small positive reward for entering position (risk taking)
                reward = 0.002
                # Bonus for entering position when not near market close
                if not is_near_close:
                    reward += 0.001
                    
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
                    # Reward based on profit percentage with holding time factor
                    profit_percentage = profit / cost_basis
                    reward = profit_percentage * 20 * (0.5 + 0.5 * holding_time_factor)
                    # Bonus for profitable exit near market close
                    if is_near_close:
                        reward += 0.005
                else:
                    self.losing_trades += 1
                    self.total_loss += abs(profit)
                    # Penalty for losses, but reduced if exit was quick (stop-loss like)
                    loss_percentage = abs(profit) / cost_basis
                    stop_loss_factor = 1.0 - holding_time_factor  # Quick exit = less penalty
                    reward = -loss_percentage * 15 * (0.3 + 0.7 * stop_loss_factor)
                
                self.position_entry_step = -1
                
            else:  # Hold (action == 0)
                self.last_action = 0
                
                # Holding rewards/penalties based on market conditions and position
                if self.position > 0:
                    # Reward for holding profitable positions
                    unrealized_pnl = ((current_price - self.entry_price) / self.entry_price)
                    if unrealized_pnl > 0:
                        reward = 0.0005 * unrealized_pnl  # Small reward for holding winners
                    else:
                        reward = 0.0002 * unrealized_pnl  # Small penalty for holding losers
                    
                    # Penalty for holding too long (encourage active management)
                    holding_time = self.current_step - self.position_entry_step if self.position_entry_step >= 0 else 0
                    if holding_time > 120:  # More than 2 hours
                        reward -= 0.001
                else:
                    # Small reward for staying in cash during potentially bad times
                    # Look at recent price movement as a proxy
                    if self.current_step > self.episode_start + 5:
                        recent_return = (current_price - self.data.iloc[self.current_step - 5]['close']) / self.data.iloc[self.current_step - 5]['close']
                        if recent_return < -0.005:  # If price dropped > 0.5%
                            reward = 0.0005  # Small reward for avoiding loss
            
            # Portfolio-level rewards
            current_portfolio_value = self.balance + (self.position * current_price)
            portfolio_return = (current_portfolio_value - self.config.initial_balance) / self.config.initial_balance
            
            # Reward for maintaining/growing portfolio value
            if portfolio_return > 0:
                reward += 0.0002 * portfolio_return
            
            # Penalty for excessive trading (more than 1 trade per hour on average)
            minutes_elapsed = max(1, self.current_step - self.episode_start)
            trade_rate = self.total_trades / (minutes_elapsed / 60)
            if trade_rate > 1.0:
                reward -= 0.001 * (trade_rate - 1.0)
            
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
        
        # Get next state (or final state if done)
        if done:
            # Return current state as next state when episode is done
            self.current_step -= 1
            next_state = self._get_state()
            self.current_step += 1
        else:
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
            'total_loss': self.total_loss
        }
        
        return next_state, reward, done, info


class DoubleDuelingDQN:
    """Double Dueling DQN Agent with PER and configurable architectures"""
    
    def __init__(self, config: TradingConfig, device: torch.device=DEVICE):
        self.config = config
        self.device = device
        print(f"Using device: {device}")
        print(f"Using architecture: {config.architecture_type}")
        
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
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.config.num_actions)
    
    def update(self) -> Dict[str, float]:
        """Perform one update step"""
        if len(self.memory) < self.config.batch_size:
            return {}
        
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
                # Track metrics if update occurred
                if update_info:
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
    
    def save(self, path: str):
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
    
    def load(self, path: str):
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
              architecture_type: str = "improved"):
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
        from src.models.mark.dqn_v2.data_preprocessor import preprocess_financial_data
        
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
        
        # Validation
        if (episode + 1) % validation_frequency == 0:
            print("\nRunning validation...")
            
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
            agent.save(f"dqn_checkpoint_episode_{episode}.pt")
            print(f"Saved checkpoint at episode {episode}")
    
    # Final test evaluation
    print("\n" + "="*50)
    print("FINAL TEST EVALUATION")
    print("="*50)
    
    test_state = test_env.reset()
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
        
        # Track for plotting - only append valid actions
        if not test_info['invalid_action']:
            test_action_history.append(test_action)
        current_value = test_info['balance'] + (test_info['position'] * test_info['current_price'])
        test_portfolio_values.append(current_value)
        test_price_history.append(test_info['current_price'])
    
    # Calculate final metrics
    test_final_value = test_portfolio_values[-1]
    test_return = (test_final_value - config.initial_balance) / config.initial_balance
    
    # Calculate performance metrics
    returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
    
    # Calculate max drawdown (this calculation is correct)
    peak = np.maximum.accumulate(test_portfolio_values)
    drawdown = (test_portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Calculate a meaningful risk-adjusted return for intraday trading
    if len(returns) > 0:
        # Use the coefficient of variation approach
        # This gives us return per unit of risk in a more interpretable way
        portfolio_volatility = np.std(test_portfolio_values) / np.mean(test_portfolio_values)
        
        if portfolio_volatility > 1e-8:
            # Risk-adjusted return: daily return divided by portfolio volatility
            sharpe = test_return / portfolio_volatility
        else:
            # If no volatility, just use the return itself
            sharpe = test_return * 10  # Scale for better readability
    else:
        sharpe = 0.0
    
    print(f"\nTest Results:")
    print(f"  Initial Balance: ${config.initial_balance:,.2f}")
    print(f"  Final Value: ${test_final_value:,.2f}")
    print(f"  Total Return: {test_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Total Trades: {test_info['total_trades']}")
    print(f"  Winning Trades: {test_info['winning_trades']}")
    print(f"  Losing Trades: {test_info['losing_trades']}")
    print(f"  Invalid Actions: {test_info['invalid_actions']}")
    
    # Return comprehensive results
    return {
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
        'test_results': {
            'final_value': test_final_value,
            'total_return': test_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': test_info['total_trades'],
            'winning_trades': test_info['winning_trades'],
            'losing_trades': test_info['losing_trades'],
            'invalid_actions': test_info['invalid_actions'],
            'action_history': test_action_history,
            'portfolio_values': test_portfolio_values,
            'price_history': test_price_history
        },
        'start_date': start_date,
        'end_date': end_date
    }


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
                    architecture_type="original",
                    hidden_size=512,
                    learning_rate=1e-4
                )
            },
            {
                "name": "Improved Transformer + Multi-scale CNN",
                "config": TradingConfig(
                    architecture_type="improved",
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
                    architecture_type="hybrid",
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
            print(f"  Architecture: {config.architecture_type}")
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
                if config.architecture_type == "original":
                    print("  • Standard 1D CNN with BatchNorm")
                    print("  • ReLU activations")
                    print("  • Max pooling")
                    print("  • Simple dueling streams")
                    
                elif config.architecture_type == "improved":
                    print("  • Multi-scale CNN (3, 5, 7 kernel sizes)")
                    print("  • Transformer blocks with self-attention")
                    print("  • GELU activations (better for financial data)")
                    print("  • GroupNorm instead of BatchNorm")
                    print("  • Positional encoding for time awareness")
                    print("  • Attention pooling")
                    print("  • Residual connections in shared layers")
                    
                elif config.architecture_type == "hybrid":
                    print("  • Multi-scale CNN for local patterns")
                    print("  • Bidirectional LSTM for temporal dependencies")
                    print("  • Attention mechanism for LSTM outputs")
                    print("  • GELU activations")
                    print("  • GroupNorm for stability")
                
                print(f"\nBest Use Cases:")
                if config.architecture_type == "original":
                    print("  ✓ Baseline model")
                    print("  ✓ Quick prototyping") 
                    print("  ✓ Limited computational resources")
                    print("  ✓ Simple pattern recognition")
                    
                elif config.architecture_type == "improved":
                    print("  ✓ Complex temporal relationships")
                    print("  ✓ Long-range dependencies")
                    print("  ✓ Multi-timeframe analysis")
                    print("  ✓ When you have sufficient data")
                    print("  ✓ Production deployment with good hardware")
                    
                elif config.architecture_type == "hybrid":
                    print("  ✓ Best of both worlds (CNN + RNN)")
                    print("  ✓ Sequential pattern recognition")
                    print("  ✓ Trend following strategies")
                    print("  ✓ Medium computational requirements")
                
                print(f"\nExpected Performance Characteristics:")
                if config.architecture_type == "original":
                    print("  • Training Speed: Fast")
                    print("  • Memory Usage: Low")
                    print("  • Pattern Recognition: Basic")
                    print("  • Overfitting Risk: Medium")
                    
                elif config.architecture_type == "improved":
                    print("  • Training Speed: Moderate")
                    print("  • Memory Usage: High")
                    print("  • Pattern Recognition: Advanced")
                    print("  • Overfitting Risk: Low (with proper regularization)")
                    
                elif config.architecture_type == "hybrid":
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
    
    # Uncomment to actually train a model:
    # train_dqn(
    #     data_path=f"{DATA_DIR}/feature_engineered/TSLA.csv",
    #     cutoff=pd.Timestamp('2020-01-01', tz='UTC'),
    #     num_episodes=100,
    #     use_preprocessing=True,
    #     scaling_method='robust',
    #     outlier_method='winsorize'
    # )
