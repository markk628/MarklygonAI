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
from datetime import datetime
from pathlib import Path

from src.config.config import (
    DEVICE,
    DATA_DIR,
    MODELS_DIR,
    WINDOW_SIZE,
    EVALUATE_INTERVAL,
    INITIAL_BALANCE,
    TRANSACTION_FEE_PERCENT,
    BATCH_SIZE,
    REPLAY_BUFFER_SIZE,
    STOCK_FEATURES,
    NUM_EPISODES,
    TRAIN_RATIO,
    VALID_RATIO,
    TRAIN_INTERVAL,
)
from src.utils.utils import create_directory
from src.web.models import app, db, BacktestHistory, ModelType, MarklygonModel

# Import the replay buffer and environment from DQN
from src.models.mark.dqn_v2.dqn import PrioritizedReplayBufferGPU, TradingEnvironment, TradingMode, load_stock_data, save_backtest_results_to_db


@dataclass
class SACConfig:
    """Configuration for the trading environment and SAC"""
    # Environment settings
    initial_balance: float = INITIAL_BALANCE
    transaction_fee_pct: float = TRANSACTION_FEE_PERCENT
    max_position_size: float = 0.7
    
    # SAC settings
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4  # Temperature parameter learning rate
    weight_decay: float = 1e-5
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    
    # Entropy settings
    target_entropy: float = -1.0  # Target entropy for discrete actions
    alpha_auto_tune: bool = True  # Automatic temperature tuning
    
    # PER settings (reuse from DQN)
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_decay: int = 100000
    per_epsilon: float = 0.001
    
    # Training settings
    batch_size: int = BATCH_SIZE
    buffer_size: int = REPLAY_BUFFER_SIZE
    hidden_size: int = 512
    
    # Features
    num_stock_features: int = 39
    num_portfolio_features: int = 8
    num_features: int = num_stock_features + num_portfolio_features 
    window_size: int = WINDOW_SIZE
    
    # Actions: 0=Hold, 1=Buy, 2=Sell
    num_actions: int = 3


class ActorNetwork(nn.Module):
    """Policy network that outputs action probabilities"""
    
    def __init__(self, config: SACConfig):
        super(ActorNetwork, self).__init__()
        self.config = config
        
        # Stock data branch - 1D CNN
        self.stock_data_branch = nn.Sequential(
            nn.Conv1d(in_channels=config.num_stock_features, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),
            
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
        
        # Portfolio branch
        self.portfolio_branch = nn.Sequential(
            nn.Linear(config.num_portfolio_features, 64),
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
        
        # Combined features
        combined_size = 256 + 64
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_actions)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in [self.stock_data_branch, self.portfolio_branch, self.policy_head]:
            for layer in module:
                if isinstance(layer, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action logits"""
        # Split input
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, 0, self.config.num_stock_features:]
        
        # Process branches
        stock_data = stock_data.permute(0, 2, 1)
        stock_features = self.stock_data_branch(stock_data).squeeze(-1)
        portfolio_features = self.portfolio_branch(portfolio_data)
        
        # Combine and get logits
        combined = torch.cat([stock_features, portfolio_features], dim=1)
        logits = self.policy_head(combined)
        
        return logits
    
    def get_action_and_log_prob(self, state: torch.Tensor, deterministic: bool = False):
        """Get action and its log probability"""
        logits = self.forward(state)
        
        if deterministic:
            # For evaluation, take the action with highest probability
            action = torch.argmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(1))
        else:
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(1)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """Q-value network"""
    
    def __init__(self, config: SACConfig):
        super(CriticNetwork, self).__init__()
        self.config = config
        
        # Stock data branch
        self.stock_data_branch = nn.Sequential(
            nn.Conv1d(in_channels=config.num_stock_features, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),
            
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
        
        # Portfolio branch
        self.portfolio_branch = nn.Sequential(
            nn.Linear(config.num_portfolio_features, 64),
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
        
        # Q-value head
        combined_size = 256 + 64
        self.q_head = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.num_actions)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in [self.stock_data_branch, self.portfolio_branch, self.q_head]:
            for layer in module:
                if isinstance(layer, (nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for all actions"""
        # Split input
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, 0, self.config.num_stock_features:]
        
        # Process branches
        stock_data = stock_data.permute(0, 2, 1)
        stock_features = self.stock_data_branch(stock_data).squeeze(-1)
        portfolio_features = self.portfolio_branch(portfolio_data)
        
        # Combine and get Q-values
        combined = torch.cat([stock_features, portfolio_features], dim=1)
        q_values = self.q_head(combined)
        
        return q_values


class SACAgent:
    """Soft Actor-Critic Agent for Stock Trading"""
    
    def __init__(self, config: SACConfig, device: torch.device = DEVICE):
        self.config = config
        self.device = device
        print(f"Using device: {device}")
        
        # Networks
        self.actor = ActorNetwork(config).to(device)
        self.critic1 = CriticNetwork(config).to(device)
        self.critic2 = CriticNetwork(config).to(device)
        self.target_critic1 = CriticNetwork(config).to(device)
        self.target_critic2 = CriticNetwork(config).to(device)
        
        # Initialize targets
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers - AdamW for better regularization
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), 
                                         lr=config.actor_lr, 
                                         weight_decay=config.weight_decay)
        self.critic1_optimizer = optim.AdamW(self.critic1.parameters(), 
                                           lr=config.critic_lr, 
                                           weight_decay=config.weight_decay)
        self.critic2_optimizer = optim.AdamW(self.critic2.parameters(), 
                                           lr=config.critic_lr, 
                                           weight_decay=config.weight_decay)
        
        # Temperature parameter for entropy regularization
        if config.alpha_auto_tune:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=config.alpha_lr)
        else:
            self.log_alpha = torch.log(torch.tensor(0.2, device=device))
        
        # Schedulers
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6, verbose=True)
        self.critic1_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic1_optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6, verbose=True)
        self.critic2_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic2_optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6, verbose=True)
        
        # Replay buffer
        self.memory = PrioritizedReplayBufferGPU(config.buffer_size, config, device)
        
        # Training tracking
        self.steps_done = 0
        self.episodes_done = 0
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Select action"""
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            action, _ = self.actor.get_action_and_log_prob(state, deterministic=deterministic)
            return action.item()
    
    def update(self) -> Dict[str, float]:
        """Perform one update step"""
        if len(self.memory) < self.config.batch_size:
            return {}
        
        try:
            # Calculate beta for importance sampling
            beta = self.config.per_beta_start + (self.config.per_beta_end - self.config.per_beta_start) * \
                   min(1.0, self.steps_done / self.config.per_beta_decay)
            
            # Sample batch
            states, actions, rewards, next_states, dones, indices, weights = \
                self.memory.sample(self.config.batch_size, beta)
            
            # Update critics
            critic_loss, td_errors = self._update_critics(states, actions, rewards, next_states, dones, weights)
            
            # Update actor and temperature
            actor_loss, alpha_loss = self._update_actor_and_alpha(states, weights)
            
            # Soft update target networks
            self._soft_update_targets()
            
            # Update priorities
            self.memory.update_priorities(indices, td_errors.detach())
            
            return {
                'critic_loss': critic_loss,
                'actor_loss': actor_loss,
                'alpha_loss': alpha_loss,
                'alpha': self.alpha.item(),
                'mean_td_error': td_errors.abs().mean().item()
            }
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA Error in update: {e}")
                torch.cuda.empty_cache()
                return {}
            else:
                raise
    
    def _update_critics(self, states, actions, rewards, next_states, dones, weights):
        """Update critic networks"""
        with torch.no_grad():
            # Get next actions and log probs from current policy
            next_actions, next_log_probs = self.actor.get_action_and_log_prob(next_states)
            
            # Get target Q-values
            target_q1 = self.target_critic1(next_states)
            target_q2 = self.target_critic2(next_states)
            target_q = torch.min(target_q1, target_q2)
            
            # Add entropy regularization
            target_q = target_q.gather(1, next_actions.unsqueeze(1)) - self.alpha * next_log_probs
            target_q = rewards.unsqueeze(1) + self.config.gamma * target_q * ~dones.unsqueeze(1)
        
        # Current Q-values
        current_q1 = self.critic1(states).gather(1, actions.unsqueeze(1))
        current_q2 = self.critic2(states).gather(1, actions.unsqueeze(1))
        
        # TD errors for priority updates
        td_errors1 = (current_q1 - target_q).squeeze()
        td_errors2 = (current_q2 - target_q).squeeze()
        td_errors = (td_errors1 + td_errors2) / 2
        
        # Clamp TD errors
        td_errors = torch.clamp(td_errors, min=-10.0, max=10.0)
        td_errors = torch.where(torch.isfinite(td_errors), td_errors, torch.zeros_like(td_errors))
        
        # Weighted losses
        critic1_loss = (weights * F.smooth_l1_loss(current_q1, target_q, reduction='none').squeeze()).mean()
        critic2_loss = (weights * F.smooth_l1_loss(current_q2, target_q, reduction='none').squeeze()).mean()
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        return (critic1_loss.item() + critic2_loss.item()) / 2, td_errors
    
    def _update_actor_and_alpha(self, states, weights):
        """Update actor and temperature parameter"""
        # Get actions and log probs from current policy
        actions, log_probs = self.actor.get_action_and_log_prob(states)
        
        # Get Q-values
        q1 = self.critic1(states)
        q2 = self.critic2(states)
        q = torch.min(q1, q2)
        
        # Actor loss (policy gradient with entropy regularization)
        q_values = q.gather(1, actions.unsqueeze(1))
        actor_loss = (weights * (self.alpha * log_probs - q_values).squeeze()).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update temperature parameter
        alpha_loss = 0
        if self.config.alpha_auto_tune:
            alpha_loss = -(self.log_alpha * (log_probs + self.config.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss = alpha_loss.item()
        
        return actor_loss.item(), alpha_loss
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
    
    def train_episode(self, env: TradingEnvironment) -> Dict[str, float]:
        """Train for one episode"""
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Track update metrics
        update_metrics = {
            'critic_loss': [],
            'actor_loss': [],
            'alpha_loss': [],
            'alpha': [],
            'mean_td_error': []
        }
        
        while True:
            # Select and execute action
            action = self.select_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            self.memory.push(state, action, reward, next_state, done)
            
            # Update counters
            episode_reward += reward
            episode_steps += 1
            self.steps_done += 1
            
            # Perform update every TRAIN_INTERVAL steps
            if self.steps_done % TRAIN_INTERVAL == 0:
                update_info = self.update()
                if update_info:
                    for key in update_metrics:
                        if key in update_info:
                            update_metrics[key].append(update_info[key])
            
            state = next_state
            if done:
                break
        
        self.episodes_done += 1
        
        # Calculate final metrics
        final_value = info['balance'] + (info['position'] * info['current_price'])
        total_return = (final_value - self.config.initial_balance) / self.config.initial_balance
        win_rate = info['winning_trades'] / max(1, info['total_trades'])
        
        # Average update metrics
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
            'win_rate': win_rate,
            'invalid_actions': info['invalid_actions'],
            **avg_update_metrics
        }
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']


def train_sac(data_path: str,
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
              preprocessor_save_path: Optional[str] = None):
    """Main SAC training function"""
    
    print("Loading data...")
    data, start_date, end_date = load_stock_data(data_path, cutoff)
    
    # Split data
    train_end = int(len(data) * train_ratio)
    valid_end = train_end + int(len(data) * valid_ratio)
    
    train_data = data.iloc[:train_end].copy().reset_index(drop=True)
    valid_data = data.iloc[train_end:valid_end].copy().reset_index(drop=True)
    test_data = data.iloc[valid_end:].copy().reset_index(drop=True)
    
    print(f"Data split: Train {len(train_data)}, Validation {len(valid_data)}, Test {len(test_data)}")
    
    # Preprocessing
    preprocessor = None
    if use_preprocessing:
        print(f"\nApplying preprocessing...")
        from src.models.mark.dqn_v2.data_preprocessor import preprocess_financial_data
        
        if preprocessor_save_path is None:
            data_name = Path(data_path).stem
            preprocessor_save_path = f"preprocessor_sac_{data_name}.pkl"
        
        preprocessor, train_data_scaled, valid_data_scaled, test_data_scaled = preprocess_financial_data(
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            scaling_method=scaling_method,
            outlier_method=outlier_method,
            save_preprocessor=True,
            preprocessor_path=preprocessor_save_path
        )
    else:
        train_data_scaled = train_data
        valid_data_scaled = valid_data
        test_data_scaled = test_data
    
    # Initialize
    config = SACConfig()
    train_env = TradingEnvironment(train_data, train_data_scaled, config, mode=TradingMode.TRAIN)
    val_env = TradingEnvironment(valid_data, valid_data_scaled, config, mode=TradingMode.VAL)
    test_env = TradingEnvironment(test_data, test_data_scaled, config, mode=TradingMode.TEST)
    
    agent = SACAgent(config)
    
    # Training tracking
    episode_rewards = []
    episode_returns = []
    validation_returns = []
    best_validation_return = float('-inf')
    patience_counter = 0
    
    print("Starting SAC training...")
    
    for episode in range(num_episodes):
        # Train
        metrics = agent.train_episode(train_env)
        episode_rewards.append(metrics['episode_reward'])
        episode_returns.append(metrics['total_return'])
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        print(f"  Reward: {metrics['episode_reward']:.4f}")
        print(f"  Return: {metrics['total_return']:.2%}")
        print(f"  Final Value: ${metrics['final_value']:,.2f}")
        print(f"  Alpha: {metrics.get('alpha', 0):.4f}")
        
        # Validation
        if (episode + 1) % validation_frequency == 0:
            print("\nRunning validation...")
            
            val_state = val_env.reset()
            val_reward = 0
            val_done = False
            
            while not val_done:
                val_action = agent.select_action(val_state, deterministic=True)
                val_next_state, val_r, val_done, val_info = val_env.step(val_action)
                val_reward += val_r
                val_state = val_next_state
            
            val_final_value = val_info['balance'] + (val_info['position'] * val_info['current_price'])
            val_return = (val_final_value - config.initial_balance) / config.initial_balance
            validation_returns.append(val_return)
            
            print(f"Validation Return: {val_return:.2%}")
            
            # Update schedulers
            agent.actor_scheduler.step(val_return)
            agent.critic1_scheduler.step(val_return)
            agent.critic2_scheduler.step(val_return)
            
            # Early stopping
            if val_return > best_validation_return:
                best_validation_return = val_return
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at episode {episode+1}")
                break
        
        if episode % save_interval == 0 and episode > 0:
            agent.save(f"sac_checkpoint_episode_{episode}.pt")
    
    # Final test
    print("\nFinal test evaluation...")
    test_state = test_env.reset()
    test_reward = 0
    test_done = False
    test_portfolio_values = [config.initial_balance]
    test_price_history = []
    test_action_history = []
    
    while not test_done:
        test_action = agent.select_action(test_state, deterministic=True)
        test_next_state, test_r, test_done, test_info = test_env.step(test_action)
        test_reward += test_r
        test_state = test_next_state
        
        if not test_info['invalid_action']:
            test_action_history.append(test_action)
        current_value = test_info['balance'] + (test_info['position'] * test_info['current_price'])
        test_portfolio_values.append(current_value)
        test_price_history.append(test_info['current_price'])
    
    # Calculate metrics
    test_final_value = test_portfolio_values[-1]
    test_return = (test_final_value - config.initial_balance) / config.initial_balance
    
    returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
    sharpe = np.sqrt(252 * 390) * returns.mean() / (returns.std() + 1e-10)
    
    peak = np.maximum.accumulate(test_portfolio_values)
    drawdown = (test_portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    print(f"\nSAC Test Results:")
    print(f"  Total Return: {test_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Total Trades: {test_info['total_trades']}")
    print(f"  Win Rate: {test_info['winning_trades'] / max(1, test_info['total_trades']):.2%}")
    
    agent.save("sac_final_model.pt")
    
    return {
        'agent': agent,
        'preprocessor': preprocessor,
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'validation_returns': validation_returns,
        'test_results': {
            'final_value': test_final_value,
            'total_return': test_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': test_info['total_trades'],
            'winning_trades': test_info['winning_trades'],
            'losing_trades': test_info['losing_trades'],
            'win_rate': test_info['winning_trades'] / max(1, test_info['total_trades']),
            'invalid_actions': test_info['invalid_actions'],
            'action_history': test_action_history,
            'portfolio_values': test_portfolio_values,
            'price_history': test_price_history
        },
        'start_date': start_date,
        'end_date': end_date
    }


if __name__ == "__main__":
    train_sac(f"{DATA_DIR}/feature_engineered/TSLA.csv", 
              cutoff=pd.Timestamp('2024-05-06 08:00:00', tz='UTC'),
              num_episodes=50,
              use_preprocessing=True)
