"""
SAC Neural Networks and Replay Buffer
====================================

Contains all network architectures and replay buffer implementations
for SAC trading agents. Supports both original and simplified networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from src.config.config import DEVICE
from src.models.mark.dqn_v2.networks.base import FinancialTransformerBlock
from src.models.jeawan.sac.sac_config import SACConfig


class PrioritizedReplayBufferGPU:
    """Prioritized Experience Replay buffer stored on GPU for SAC"""
    
    def __init__(self, capacity: int, config: SACConfig, device: torch.device = DEVICE):
        self.capacity = capacity
        self.config = config
        self.device = device
        self.position = 0
        self.size = 0
        
        # Store experiences on GPU
        self.states = torch.zeros((capacity, config.window_size, config.num_features), 
                                 dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, config.window_size, config.num_features), 
                                      dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        # PER parameters
        self.alpha = getattr(config, 'per_alpha', 0.6)
        self.beta_start = getattr(config, 'per_beta_start', 0.4)
        self.beta_end = getattr(config, 'per_beta_end', 1.0)
        self.per_epsilon = getattr(config, 'per_epsilon', 0.001)
        
        # Priority storage
        self.priorities = torch.ones(capacity, dtype=torch.float32, device=device) * 0.01
        self.max_priority = 1.0
        
    def push(self, state: torch.Tensor, action: float, reward: float, 
             next_state: torch.Tensor, done: bool):
        """Store experience with maximum priority for new experiences"""
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        
        self.states[self.position] = state
        self.actions[self.position] = torch.tensor([[action]], dtype=torch.float32, device=self.device)
        self.rewards[self.position] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.position] = next_state
        self.dones[self.position] = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        # Set priority to max for new experiences
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int, beta: float) -> Tuple[torch.Tensor, ...]:
        """Sample batch of prioritized experiences"""
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        priorities = torch.clamp(priorities, min=self.per_epsilon)
        
        # Calculate probabilities
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        
        # Handle edge cases
        if probs_sum < 1e-10 or torch.isnan(probs_sum) or torch.isinf(probs_sum):
            probs = torch.ones_like(probs) / self.size
        else:
            probs = probs / probs_sum
            
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones(self.size, device=self.device) / self.size
        
        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
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
        priorities = torch.abs(td_errors) + self.per_epsilon
        priorities = torch.clamp(priorities, min=max(self.per_epsilon, 0.001), max=1e6)
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max().item())
    
    def __len__(self):
        return self.size


# ==========================================
# ORIGINAL NETWORKS (Complex, 900k+ params)
# ==========================================

class Actor(nn.Module):
    """Original Actor network with complex architecture"""
    
    def __init__(self, config: SACConfig):
        super(Actor, self).__init__()
        self.config = config
        
        # Feature embedding for stock data
        self.feature_embedding = nn.Linear(config.num_stock_features, 128)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(config.window_size, 128) * 0.02)
        
        # Multi-scale CNN processing
        self.cnn_branches = nn.ModuleList([
            self._create_cnn_branch(128, kernel_size) 
            for kernel_size in [3, 5, 7]
        ])
        
        # Transformer for temporal modeling
        self.transformer_blocks = nn.ModuleList([
            FinancialTransformerBlock(128, nhead=8, dropout=0.1)
            for _ in range(2)
        ])
        
        # Attention pooling
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
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, config.actor_hidden_size),
            nn.LayerNorm(config.actor_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(config.actor_hidden_size, config.actor_hidden_size),
            nn.LayerNorm(config.actor_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Action head
        self.action_mean = nn.Linear(config.actor_hidden_size, 1)
        self.action_log_std = nn.Linear(config.actor_hidden_size, 1)
        
        self._initialize_weights()
        
    def _create_cnn_branch(self, in_channels: int, kernel_size: int):
        """Create CNN branch for multi-scale processing"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module in [self.action_mean, self.action_log_std]:
                    nn.init.uniform_(module.weight, -0.1, 0.1)
                    nn.init.uniform_(module.bias, -0.05, 0.05)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and log_std"""
        batch_size = x.size(0)
        
        # Split stock and portfolio data
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, 0, self.config.num_stock_features:]  # Only first timestep
        
        # Process stock data
        embedded = self.feature_embedding(stock_data)
        embedded = embedded + self.pos_encoding.unsqueeze(0)
        
        # Multi-scale CNN
        multiscale_features = []
        stock_data_cnn = embedded.transpose(1, 2)
        for cnn_branch in self.cnn_branches:
            features = cnn_branch(stock_data_cnn).squeeze(-1)
            multiscale_features.append(features)
        
        # Transformer processing
        transformer_out = embedded
        for transformer_block in self.transformer_blocks:
            transformer_out = transformer_block(transformer_out)
        
        # Attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled_features, _ = self.attention_pool(query, transformer_out, transformer_out)
        pooled_features = pooled_features.squeeze(1)
        
        # Portfolio processing
        portfolio_features = self.portfolio_branch(portfolio_data)
        
        # Combine features
        combined_features = torch.cat([
            *multiscale_features,
            pooled_features,
            portfolio_features
        ], dim=1)
        
        # Shared processing
        shared_out = self.shared_layers(combined_features)
        
        # Action distribution parameters
        mean = self.action_mean(shared_out)
        log_std = self.action_log_std(shared_out)
        log_std = torch.clamp(log_std, min=-10, max=3)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from Gaussian
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        # Apply tanh to bound action to [-1, 1]
        action = torch.tanh(x_t)
        
        # Calculate log probability with change of variables formula
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-8)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Original Critic network with complex architecture"""
    
    def __init__(self, config: SACConfig):
        super(Critic, self).__init__()
        self.config = config
        
        # Same architecture as Actor but with action input
        self.feature_embedding = nn.Linear(config.num_stock_features, 128)
        self.pos_encoding = nn.Parameter(torch.randn(config.window_size, 128) * 0.02)
        
        self.cnn_branches = nn.ModuleList([
            self._create_cnn_branch(128, kernel_size) 
            for kernel_size in [3, 5, 7]
        ])
        
        self.transformer_blocks = nn.ModuleList([
            FinancialTransformerBlock(128, nhead=8, dropout=0.1)
            for _ in range(2)
        ])
        
        self.attention_pool = nn.MultiheadAttention(128, 4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 128))
        
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
        
        # Combined features + action: multiscale (128*3) + transformer (128) + portfolio (64) + action (1) = 577
        combined_size = 128 * 3 + 128 + 64 + 1
        
        # Q-value network
        self.q_network = nn.Sequential(
            nn.Linear(combined_size, config.critic_hidden_size),
            nn.LayerNorm(config.critic_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(config.critic_hidden_size, config.critic_hidden_size),
            nn.LayerNorm(config.critic_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(config.critic_hidden_size, config.critic_hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.critic_hidden_size // 2, 1)
        )
        
        self._initialize_weights()
        
    def _create_cnn_branch(self, in_channels: int, kernel_size: int):
        """Create CNN branch for multi-scale processing"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass with state and action"""
        batch_size = x.size(0)
        
        # Split stock and portfolio data
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, 0, self.config.num_stock_features:]  # Only first timestep
        
        # Process stock data (same as actor)
        embedded = self.feature_embedding(stock_data)
        embedded = embedded + self.pos_encoding.unsqueeze(0)
        
        # Multi-scale CNN
        multiscale_features = []
        stock_data_cnn = embedded.transpose(1, 2)
        for cnn_branch in self.cnn_branches:
            features = cnn_branch(stock_data_cnn).squeeze(-1)
            multiscale_features.append(features)
        
        # Transformer processing
        transformer_out = embedded
        for transformer_block in self.transformer_blocks:
            transformer_out = transformer_block(transformer_out)
        
        # Attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled_features, _ = self.attention_pool(query, transformer_out, transformer_out)
        pooled_features = pooled_features.squeeze(1)
        
        # Portfolio processing
        portfolio_features = self.portfolio_branch(portfolio_data)
        
        # Combine features with action
        combined_features = torch.cat([
            *multiscale_features,
            pooled_features,
            portfolio_features,
            action
        ], dim=1)
        
        # Q-value estimation
        q_value = self.q_network(combined_features)
        
        return q_value


# ==========================================
# SIMPLIFIED NETWORKS (Efficient, 170k params)
# ==========================================

class SimplifiedActor(nn.Module):
    """Simplified Actor network for continuous action space"""
    
    def __init__(self, config: SACConfig):
        super(SimplifiedActor, self).__init__()
        self.config = config
        
        # Feature embedding (reduced dimension)
        self.feature_embedding = nn.Linear(config.num_stock_features, 64)
        
        # Positional encoding (smaller)
        self.pos_encoding = nn.Parameter(torch.randn(config.window_size, 64) * 0.02)
        
        # Single CNN branch (instead of 3)
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(64, 48, kernel_size=5, padding=2),
            nn.GroupNorm(3, 48),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Transformer (fewer heads, better dimensions)
        self.transformer_block = FinancialTransformerBlock(64, nhead=4, dropout=0.1)
        
        # Attention pooling (smaller)
        self.attention_pool = nn.MultiheadAttention(64, 2, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 64))
        
        # Portfolio temporal processing
        self.portfolio_embedding = nn.Linear(config.num_portfolio_features, 32)
        self.portfolio_temporal = nn.LSTM(32, 24, batch_first=True)
        
        # Combined features: CNN (64) + Transformer (64) + Portfolio LSTM (24) = 152
        combined_size = 64 + 64 + 24
        
        # Simplified shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Action heads
        self.action_mean = nn.Linear(128, 1)
        self.action_log_std = nn.Linear(128, 1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Conservative weight initialization for financial data"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module in [self.action_mean, self.action_log_std]:
                    nn.init.uniform_(module.weight, -0.05, 0.05)
                    nn.init.uniform_(module.bias, -0.01, 0.01)
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with simplified processing"""
        batch_size = x.size(0)
        
        # Split data
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, :, self.config.num_stock_features:]  # USE ALL TIMESTEPS!
        
        # Stock data processing
        embedded = self.feature_embedding(stock_data)
        embedded = embedded + self.pos_encoding.unsqueeze(0)
        
        # Single CNN branch
        stock_cnn = embedded.transpose(1, 2)
        cnn_features = self.cnn_branch(stock_cnn).squeeze(-1)
        
        # Single transformer block
        transformer_out = self.transformer_block(embedded)
        
        # Attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled_features, _ = self.attention_pool(query, transformer_out, transformer_out)
        pooled_features = pooled_features.squeeze(1)
        
        # Portfolio temporal processing
        portfolio_embedded = self.portfolio_embedding(portfolio_data)
        portfolio_lstm_out, (h_n, c_n) = self.portfolio_temporal(portfolio_embedded)
        portfolio_features = h_n[-1]
        
        # Combine all features
        combined_features = torch.cat([
            cnn_features,
            pooled_features,
            portfolio_features
        ], dim=1)
        
        # Shared processing
        shared_out = self.shared_layers(combined_features)
        
        # Action distribution
        mean = self.action_mean(shared_out)
        log_std = self.action_log_std(shared_out)
        log_std = torch.clamp(log_std, min=-10, max=2)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from Gaussian
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        # Apply tanh to bound action
        action = torch.tanh(x_t)
        
        # Log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-8)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class SimplifiedCritic(nn.Module):
    """Simplified Critic network for Q-value estimation"""
    
    def __init__(self, config: SACConfig):
        super(SimplifiedCritic, self).__init__()
        self.config = config
        
        # Same stock processing as actor (smaller)
        self.feature_embedding = nn.Linear(config.num_stock_features, 64)
        self.pos_encoding = nn.Parameter(torch.randn(config.window_size, 64) * 0.02)
        
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(64, 48, kernel_size=5, padding=2),
            nn.GroupNorm(3, 48),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.transformer_block = FinancialTransformerBlock(64, nhead=4, dropout=0.1)
        self.attention_pool = nn.MultiheadAttention(64, 2, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 64))
        
        # Portfolio temporal processing
        self.portfolio_embedding = nn.Linear(config.num_portfolio_features, 32)
        self.portfolio_temporal = nn.LSTM(32, 24, batch_first=True)
        
        # Combined features + action: 64 + 64 + 24 + 1 = 153
        combined_size = 64 + 64 + 24 + 1
        
        # Q-value network (simplified)
        self.q_network = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Conservative weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass with state and action"""
        batch_size = x.size(0)
        
        # Split data
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, :, self.config.num_stock_features:]  # USE ALL TIMESTEPS!
        
        # Stock processing (same as actor)
        embedded = self.feature_embedding(stock_data)
        embedded = embedded + self.pos_encoding.unsqueeze(0)
        
        stock_cnn = embedded.transpose(1, 2)
        cnn_features = self.cnn_branch(stock_cnn).squeeze(-1)
        
        transformer_out = self.transformer_block(embedded)
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled_features, _ = self.attention_pool(query, transformer_out, transformer_out)
        pooled_features = pooled_features.squeeze(1)
        
        # Portfolio processing (temporal)
        portfolio_embedded = self.portfolio_embedding(portfolio_data)
        portfolio_lstm_out, (h_n, c_n) = self.portfolio_temporal(portfolio_embedded)
        portfolio_features = h_n[-1]
        
        # Combine with action
        combined_features = torch.cat([
            cnn_features,
            pooled_features,
            portfolio_features,
            action
        ], dim=1)
        
        # Q-value estimation
        q_value = self.q_network(combined_features)
        
        return q_value


# ==========================================
# NETWORK FACTORY
# ==========================================

def create_networks(config: SACConfig, device: torch.device = DEVICE):
    """Factory function to create networks based on config"""
    if config.network_type.value == 'simplified':
        actor = SimplifiedActor(config).to(device)
        critic_cls = SimplifiedCritic
    else:  # original
        actor = Actor(config).to(device)
        critic_cls = Critic
    
    critic1 = critic_cls(config).to(device)
    critic2 = critic_cls(config).to(device)
    
    # Target critics
    target_critic1 = critic_cls(config).to(device)
    target_critic2 = critic_cls(config).to(device)
    
    # Copy weights to target networks
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    
    return actor, critic1, critic2, target_critic1, target_critic2


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_network_architectures():
    """Compare parameter counts between network types"""
    from src.models.jeawan.sac.sac_config import SACConfig, NetworkType
    
    config = SACConfig()
    
    # Original networks
    config.network_type = NetworkType.ORIGINAL
    orig_actor, orig_critic1, _, _, _ = create_networks(config)
    
    # Simplified networks  
    config.network_type = NetworkType.SIMPLIFIED
    simp_actor, simp_critic1, _, _, _ = create_networks(config)
    
    orig_actor_params = count_parameters(orig_actor)
    orig_critic_params = count_parameters(orig_critic1)
    simp_actor_params = count_parameters(simp_actor)
    simp_critic_params = count_parameters(simp_critic1)
    
    print("="*60)
    print("NETWORK ARCHITECTURE COMPARISON")
    print("="*60)
    print(f"Original Actor:     {orig_actor_params:,} parameters")
    print(f"Simplified Actor:   {simp_actor_params:,} parameters")
    print(f"Reduction:          {((orig_actor_params - simp_actor_params) / orig_actor_params * 100):.1f}%")
    print()
    print(f"Original Critic:    {orig_critic_params:,} parameters")
    print(f"Simplified Critic:  {simp_critic_params:,} parameters")
    print(f"Reduction:          {((orig_critic_params - simp_critic_params) / orig_critic_params * 100):.1f}%")
    print()
    print("SIMPLIFIED NETWORK ADVANTAGES:")
    print("✅ 75-80% fewer parameters → faster training")
    print("✅ Temporal portfolio processing → better patterns")
    print("✅ Less overfitting → better generalization")
    print("✅ Conservative initialization → more stable")


if __name__ == "__main__":
    compare_network_architectures() 