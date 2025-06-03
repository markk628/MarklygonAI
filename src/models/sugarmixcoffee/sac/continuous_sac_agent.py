import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path

from src.config.config import DEVICE, NUM_EPISODES, TRAIN_RATIO, VALID_RATIO, EVALUATE_INTERVAL, TRAIN_INTERVAL
from src.models.sugarmixcoffee.sac.continuous_sac import (
    ContinuousSACConfig, ContinuousActorNetwork, ContinuousCriticNetwork,
    ContinuousPrioritizedReplayBuffer, ContinuousTradingEnvironment, TradingMode, load_stock_data
)


class ContinuousSACAgent:
    """Continuous SAC Agent for Stock Trading with Position Sizing"""
    
    def __init__(self, config: ContinuousSACConfig, device: torch.device = DEVICE):
        self.config = config
        self.device = device
        print(f"Using device: {device}")
        
        # Networks
        self.actor = ContinuousActorNetwork(config).to(device)
        self.critic1 = ContinuousCriticNetwork(config).to(device)
        self.critic2 = ContinuousCriticNetwork(config).to(device)
        self.target_critic1 = ContinuousCriticNetwork(config).to(device)
        self.target_critic2 = ContinuousCriticNetwork(config).to(device)
        
        # Initialize targets
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), 
                                         lr=config.actor_lr, 
                                         weight_decay=config.weight_decay)
        self.critic1_optimizer = optim.AdamW(self.critic1.parameters(), 
                                           lr=config.critic_lr, 
                                           weight_decay=config.weight_decay)
        self.critic2_optimizer = optim.AdamW(self.critic2.parameters(), 
                                           lr=config.critic_lr, 
                                           weight_decay=config.weight_decay)
        
        # Temperature parameter
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
        self.memory = ContinuousPrioritizedReplayBuffer(config.buffer_size, config, device)
        
        # Training tracking
        self.steps_done = 0
        self.episodes_done = 0
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """Select continuous action for position sizing"""
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            action = self.actor.sample_action(state, deterministic=deterministic)
            return action.cpu().numpy().flatten()
    
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
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Add entropy regularization
            if next_log_probs is not None:
                target_q = target_q - self.alpha * next_log_probs
            
            target_q = rewards.unsqueeze(1) + self.config.gamma * target_q * ~dones.unsqueeze(1)
        
        # Current Q-values
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
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
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q = torch.min(q1, q2)
        
        # Actor loss (policy gradient with entropy regularization)
        if log_probs is not None:
            actor_loss = (weights * (self.alpha * log_probs - q).squeeze()).mean()
        else:
            actor_loss = (weights * (-q).squeeze()).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update temperature parameter
        alpha_loss = 0
        if self.config.alpha_auto_tune and log_probs is not None:
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
    
    def train_episode(self, env: ContinuousTradingEnvironment) -> Dict[str, float]:
        """Train for one episode"""
        state = env.reset()
        episode_reward = 0.0  # Ensure float type
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
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
            self.memory.push(state, action_tensor, reward, next_state, done)
            
            # Update counters - ensure reward is scalar
            reward_scalar = float(reward) if np.isscalar(reward) else float(reward.item() if hasattr(reward, 'item') else reward)
            episode_reward += reward_scalar
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
        final_value = info['balance'] + info['position_value']
        total_return = (final_value - self.config.initial_balance) / self.config.initial_balance
        win_rate = info['winning_trades'] / max(1, info['total_trades'])
        
        # Average update metrics
        avg_update_metrics = {}
        for key, values in update_metrics.items():
            if values:
                avg_update_metrics[key] = sum(values) / len(values)
        
        return {
            'episode_reward': float(episode_reward),  # Ensure float type
            'episode_steps': episode_steps,
            'total_return': float(total_return),
            'final_value': float(final_value),
            'total_trades': info['total_trades'],
            'win_rate': float(win_rate),
            'invalid_actions': info['invalid_actions'],
            'total_fees_paid': float(info['total_fees_paid']),
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


def train_continuous_sac(data_path: str,
                         cutoff: pd.Timestamp,
                         num_episodes: int = NUM_EPISODES,
                         save_interval: int = 100,
                         validation_frequency: int = EVALUATE_INTERVAL,
                         early_stopping_patience: int = 15,
                         train_ratio: float = TRAIN_RATIO,
                         valid_ratio: float = VALID_RATIO,
                         use_preprocessing: bool = True,
                         scaling_method: str = 'robust',
                         outlier_method: str = 'winsorize',
                         preprocessor_save_path: Optional[str] = None):
    """Main continuous SAC training function"""
    
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
            preprocessor_save_path = f"preprocessor_continuous_sac_{data_name}.pkl"
        
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
    config = ContinuousSACConfig()
    train_env = ContinuousTradingEnvironment(train_data, train_data_scaled, config, mode=TradingMode.TRAIN)
    val_env = ContinuousTradingEnvironment(valid_data, valid_data_scaled, config, mode=TradingMode.VAL)
    test_env = ContinuousTradingEnvironment(test_data, test_data_scaled, config, mode=TradingMode.TEST)
    
    agent = ContinuousSACAgent(config)
    
    # Training tracking
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
    
    print("Starting Continuous SAC training...")
    
    for episode in range(num_episodes):
        # Train
        metrics = agent.train_episode(train_env)
        
        episode_rewards.append(metrics['episode_reward'])
        episode_returns.append(metrics['total_return'])
        episode_trades.append(metrics['total_trades'])
        episode_invalid_actions.append(metrics['invalid_actions'])
        

        print(f"\nEpisode {episode+1}/{num_episodes}")
        print(f"  Reward: {float(metrics['episode_reward']):.4f}")
        print(f"  Return: {float(metrics['total_return']):.2%}")
        print(f"  Final Value: ${float(metrics['final_value']):,.2f}")
        print(f"  Trades: {metrics['total_trades']}, Win Rate: {float(metrics['win_rate']):.2%}, Fees: ${float(metrics['total_fees_paid']):.2f}")
        print(f"  Invalid Actions: {metrics['invalid_actions']}")
        print(f"  Steps: {metrics['episode_steps']}")
        print(f"  Alpha: {float(metrics.get('alpha', 0)):.4f}")
        
        # Validation
        if (episode + 1) % validation_frequency == 0:
            print("\nRunning validation...")
            
            val_state = val_env.reset()
            val_reward = 0.0  # Ensure float type
            val_done = False
            
            while not val_done:
                val_action = agent.select_action(val_state, deterministic=True)
                val_next_state, val_r, val_done, val_info = val_env.step(val_action)
                # Ensure val_r is scalar
                val_r_scalar = float(val_r) if np.isscalar(val_r) else float(val_r.item() if hasattr(val_r, 'item') else val_r)
                val_reward += val_r_scalar
                val_state = val_next_state
            
            val_final_value = float(val_info['balance']) + float(val_info['position_value'])
            val_return = float((val_final_value - config.initial_balance) / config.initial_balance)
            
            validation_rewards.append(val_reward)
            validation_returns.append(val_return)
            validation_trades.append(val_info['total_trades'])
            validation_invalid_actions.append(val_info['invalid_actions'])
            
            print(f"Validation Results:")
            print(f"  Return: {val_return:.2%}")
            print(f"  Final Value: ${val_final_value:,.2f}")
            print(f"  Trades: {val_info['total_trades']}")
            print(f"  Win Rate: {float(val_info['winning_trades'] / max(1, val_info['total_trades'])):.2%}")
            print(f"  Fees: ${float(val_info['total_fees_paid']):.2f}")
            print(f"  Invalid Actions: {val_info['invalid_actions']}")

            # Update schedulers
            agent.actor_scheduler.step(val_return)
            agent.critic1_scheduler.step(val_return)
            agent.critic2_scheduler.step(val_return)
            
            # Early stopping
            if val_return > best_validation_return:
                best_validation_return = val_return
                patience_counter = 0
                
                best_model_state = {
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic1_state_dict': agent.critic1.state_dict(),
                    'critic2_state_dict': agent.critic2.state_dict(),
                    'target_critic1_state_dict': agent.target_critic1.state_dict(),
                    'target_critic2_state_dict': agent.target_critic2.state_dict(),
                    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                    'critic1_optimizer_state_dict': agent.critic1_optimizer.state_dict(),
                    'critic2_optimizer_state_dict': agent.critic2_optimizer.state_dict(),
                    'log_alpha': agent.log_alpha,
                    'steps_done': agent.steps_done,
                    'episodes_done': agent.episodes_done
                }
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at episode {episode+1}")
                print(f"Best validation return: {best_validation_return:.2%}")
                
                # Restore best model
                if best_model_state:
                    agent.actor.load_state_dict(best_model_state['actor_state_dict'])
                    agent.critic1.load_state_dict(best_model_state['critic1_state_dict'])
                    agent.critic2.load_state_dict(best_model_state['critic2_state_dict'])
                    agent.target_critic1.load_state_dict(best_model_state['target_critic1_state_dict'])
                    agent.target_critic2.load_state_dict(best_model_state['target_critic2_state_dict'])
                    agent.actor_optimizer.load_state_dict(best_model_state['actor_optimizer_state_dict'])
                    agent.critic1_optimizer.load_state_dict(best_model_state['critic1_optimizer_state_dict'])
                    agent.critic2_optimizer.load_state_dict(best_model_state['critic2_optimizer_state_dict'])
                    agent.log_alpha = best_model_state['log_alpha']
                    agent.steps_done = best_model_state['steps_done']
                    agent.episodes_done = best_model_state['episodes_done']
        
        
        if episode % save_interval == 0 and episode > 0:
            agent.save(f"continuous_sac_checkpoint_episode_{episode}.pt")
            print(f"Saved checkpoint at episode {episode}")
    
    # Final test
    print("\n" + "="*50)
    print("FINAL TEST EVALUATION")
    print("="*50)
    
    test_state = test_env.reset()
    test_reward = 0.0  # Ensure float type
    test_done = False
    test_portfolio_values = [config.initial_balance]
    test_price_history = []
    test_action_history = []
    test_position_history = []
    
    while not test_done:
        test_action = agent.select_action(test_state, deterministic=True)
        test_next_state, test_r, test_done, test_info = test_env.step(test_action)
        # Ensure test_r is scalar
        test_r_scalar = float(test_r) if np.isscalar(test_r) else float(test_r.item() if hasattr(test_r, 'item') else test_r)
        test_reward += test_r_scalar
        test_state = test_next_state
        
        current_value = float(test_info['balance']) + float(test_info['position_value'])
        test_portfolio_values.append(current_value)
        test_price_history.append(float(test_info['current_price']))
        if not test_info['invalid_action']:
            test_action_history.append(float(test_info['action_value']))
        test_position_history.append(float(test_info['target_position_ratio']))
    
    # Calculate metrics
    test_final_value = test_portfolio_values[-1]
    test_return = float((test_final_value - config.initial_balance) / config.initial_balance)
    
    returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
    sharpe = float(np.sqrt(252 * 390) * returns.mean() / (returns.std() + 1e-10))
    
    peak = np.maximum.accumulate(test_portfolio_values)
    drawdown = (test_portfolio_values - peak) / peak
    max_drawdown = float(np.min(drawdown))
    
    print(f"\nContinuous SAC Test Results:")
    print(f"  Initial Balance: ${config.initial_balance:,.2f}")
    print(f"  Final Value: ${test_final_value:,.2f}")
    print(f"  Total Return: {test_return:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    print(f"  Total Trades: {test_info['total_trades']}")
    print(f"  Win Rate: {float(test_info['winning_trades'] / max(1, test_info['total_trades'])):.2%}")
    print(f"  Total Fees: ${float(test_info['total_fees_paid']):.2f}")
    print(f"  Invalid Actions: {test_info['invalid_actions']}")

    # agent.save("continuous_sac_final_model.pt")
    
    return {
        'agent': agent,
        'preprocessor': preprocessor,
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'episode_trades': episode_trades,
        'episode_invalid_actions': episode_invalid_actions,
        'validation_reward': validation_rewards,
        'validation_returns': validation_returns,
        'validation_trades': validation_trades,
        'validation_invalid_actions': validation_invalid_actions,
        'test_results': {
            'final_value': float(test_final_value),
            'total_return': float(test_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'total_trades': int(test_info['total_trades']),
            'winning_trades': int(test_info['winning_trades']),
            'losing_trades': int(test_info['losing_trades']),
            'win_rate': float(test_info['winning_trades'] / max(1, test_info['total_trades'])),
            'invalid_actions': int(test_info['invalid_actions']),
            'total_fees_paid': float(test_info['total_fees_paid']),
            'action_history': test_action_history,
            'position_history': test_position_history,
            'portfolio_values': test_portfolio_values,
            'price_history': test_price_history
        },
        'start_date': start_date,
        'end_date': end_date
    } 