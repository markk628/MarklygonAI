"""
SAC Agent Implementation
=======================

Contains the main SAC (Soft Actor-Critic) agent class that combines
networks, replay buffer, and training algorithms.

Separated from environments and networks to provide clean modularity.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional

from src.config.config import DEVICE
from src.models.jeawan.sac.sac_config import SACConfig
from src.models.jeawan.sac.sac_networks import (
    create_networks, 
    PrioritizedReplayBufferGPU,
    count_parameters
)


class SAC:
    """Soft Actor-Critic agent for continuous action trading"""
    
    def __init__(self, config: SACConfig, device: torch.device = DEVICE):
        self.config = config
        self.device = device
        
        # Create networks using factory
        self.actor, self.critic1, self.critic2, self.target_critic1, self.target_critic2 = \
            create_networks(config, device)
        
        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic1_optimizer = optim.AdamW(self.critic1.parameters(), lr=config.critic_learning_rate)
        self.critic2_optimizer = optim.AdamW(self.critic2.parameters(), lr=config.critic_learning_rate)
        
        # Learning Rate Schedulers
        self.schedulers = {}
        if config.use_lr_scheduler:
            self._create_schedulers()
        
        # Automatic entropy tuning
        if config.alpha_auto_tune:
            self.target_entropy = config.target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=config.alpha_learning_rate)
            
            # Alpha scheduler (usually not needed)
            if config.use_lr_scheduler and config.schedule_alpha:
                self.schedulers['alpha'] = self._create_scheduler(self.alpha_optimizer, None, 'alpha')
        else:
            self.alpha = config.initial_alpha
        
        # Replay buffer
        self.memory = PrioritizedReplayBufferGPU(config.buffer_size, config, device)
        
        # Training tracking
        self.steps_done = 0
        self.episodes_done = 0
        
        # Print network info
        total_params = sum(p.numel() for p in self.actor.parameters()) + \
                      sum(p.numel() for p in self.critic1.parameters()) + \
                      sum(p.numel() for p in self.critic2.parameters())
        print(f"SAC agent initialized:")
        print(f"  Network type: {config.network_type.value}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Auto-tune alpha: {config.alpha_auto_tune}")
        print(f"  Target entropy: {config.target_entropy}")
        if config.use_lr_scheduler:
            print(f"  LR Scheduler: {config.scheduler_type} (schedulers: {list(self.schedulers.keys())})")
    
    def _create_schedulers(self):
        """Create learning rate schedulers for optimizers"""
        config = self.config
        
        if config.schedule_actor:
            self.schedulers['actor'] = self._create_scheduler(
                self.actor_optimizer, 
                config.actor_scheduler_type,
                'actor'
            )
        
        if config.schedule_critics:
            self.schedulers['critic1'] = self._create_scheduler(
                self.critic1_optimizer, 
                config.critic_scheduler_type,
                'critic'
            )
            self.schedulers['critic2'] = self._create_scheduler(
                self.critic2_optimizer, 
                config.critic_scheduler_type,
                'critic'
            )
    
    def _create_scheduler(self, optimizer, scheduler_type: str = None, optimizer_type: str = 'default'):
        """Create a specific scheduler for an optimizer
        
        Args:
            optimizer: The optimizer to create scheduler for
            scheduler_type: Type of scheduler ('plateau', 'exponential', etc.)
            optimizer_type: Type of optimizer ('actor', 'critic', 'alpha') for parameter selection
        """
        config = self.config
        
        # Use specified scheduler type or fall back to default
        if scheduler_type is None:
            scheduler_type = config.scheduler_type
        
        # Get optimizer-specific parameters
        if optimizer_type == 'actor':
            patience = config.actor_scheduler_patience
            factor = config.actor_scheduler_factor
            gamma = config.actor_scheduler_gamma
        elif optimizer_type == 'critic':
            patience = config.critic_scheduler_patience
            factor = config.critic_scheduler_factor
            gamma = config.critic_scheduler_gamma
        elif optimizer_type == 'alpha':
            patience = config.alpha_scheduler_patience
            factor = config.alpha_scheduler_factor
            gamma = config.alpha_scheduler_gamma
        else:
            # Default fallback
            patience = config.scheduler_patience
            factor = config.scheduler_factor
            gamma = config.scheduler_gamma
        
        if scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.scheduler_mode,
                factor=factor,
                patience=patience,
                min_lr=config.scheduler_min_lr,
                threshold=config.scheduler_threshold,
                verbose=False  # Disable initialization messages, we provide better logging during validation
            )
        elif scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=gamma,
                verbose=False  # Disable initialization messages, we provide better logging during validation
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.scheduler_step_size,
                gamma=factor,
                verbose=False  # Disable initialization messages, we provide better logging during validation
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.scheduler_T_max,
                eta_min=config.scheduler_eta_min,
                verbose=False  # Disable initialization messages, we provide better logging during validation
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step_schedulers(self, metric: Optional[float] = None):
        """Step learning rate schedulers
        
        Args:
            metric: Metric for plateau scheduler (e.g., validation return)
        """
        if not self.config.use_lr_scheduler:
            return
        
        for name, scheduler in self.schedulers.items():
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if metric is not None:
                    scheduler.step(metric)
            else:
                scheduler.step()
    
    def get_current_learning_rates(self) -> Dict[str, float]:
        """Get current learning rates for all optimizers"""
        lrs = {}
        
        lrs['actor'] = self.actor_optimizer.param_groups[0]['lr']
        lrs['critic1'] = self.critic1_optimizer.param_groups[0]['lr']
        lrs['critic2'] = self.critic2_optimizer.param_groups[0]['lr']
        
        if self.config.alpha_auto_tune:
            lrs['alpha'] = self.alpha_optimizer.param_groups[0]['lr']
        
        return lrs
    
    @property
    def alpha(self):
        """Get current alpha value"""
        if self.config.alpha_auto_tune:
            return self.log_alpha.exp()
        else:
            return self._alpha
    
    @alpha.setter 
    def alpha(self, value):
        """Set alpha value (only when not auto-tuning)"""
        if not self.config.alpha_auto_tune:
            self._alpha = value
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> float:
        """Select action from policy"""
        self.actor.eval()
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            if deterministic:
                # Use mean action for evaluation
                mean, _ = self.actor.forward(state)
                action = torch.tanh(mean)
            else:
                # Sample from policy
                action, _ = self.actor.sample(state)
            
            action = action.cpu().item()
        self.actor.train()
        return action
    
    def update(self) -> Dict[str, float]:
        """Perform SAC update with prioritized experience replay"""
        if len(self.memory) < self.config.batch_size:
            return {}
        
        # Calculate current beta for importance sampling
        beta_annealing_steps = getattr(self.config, 'per_beta_annealing_steps', 100000)
        beta = self.config.per_beta_start + (self.config.per_beta_end - self.config.per_beta_start) * \
               min(1.0, self.steps_done / beta_annealing_steps)
        
        try:
            # Sample batch with prioritized sampling
            states, actions, rewards, next_states, dones, indices, weights = \
                self.memory.sample(self.config.batch_size, beta)
            
            # Update critics
            with torch.no_grad():
                # Sample actions for next states
                next_actions, next_log_probs = self.actor.sample(next_states)
                
                # Target Q-values using minimum of two critics
                target_q1 = self.target_critic1(next_states, next_actions)
                target_q2 = self.target_critic2(next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
                
                target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1).float()) * self.config.gamma * target_q
            
            # Current Q-values
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            
            # Calculate TD errors for priority updates
            td_errors_1 = (current_q1 - target_q).squeeze()
            td_errors_2 = (current_q2 - target_q).squeeze()
            
            # Clamp TD errors to prevent extreme values
            td_errors_1 = torch.clamp(td_errors_1, min=-10.0, max=10.0)
            td_errors_2 = torch.clamp(td_errors_2, min=-10.0, max=10.0)
            
            # Replace any NaN/inf values with zero
            td_errors_1 = torch.where(torch.isfinite(td_errors_1), td_errors_1, torch.zeros_like(td_errors_1))
            td_errors_2 = torch.where(torch.isfinite(td_errors_2), td_errors_2, torch.zeros_like(td_errors_2))
            
            # Combined TD error for priority update (average of both critics)
            td_errors = (torch.abs(td_errors_1) + torch.abs(td_errors_2)) / 2.0
            
            # Update priorities in replay buffer
            self.memory.update_priorities(indices, td_errors.detach())
            
            # Weighted critic losses using importance sampling weights
            critic1_loss = (weights * F.mse_loss(current_q1, target_q, reduction='none').squeeze()).mean()
            critic2_loss = (weights * F.mse_loss(current_q2, target_q, reduction='none').squeeze()).mean()
            
            # Update critics
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
            self.critic1_optimizer.step()
            
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
            self.critic2_optimizer.step()
            
            # Update actor
            new_actions, log_probs = self.actor.sample(states)
            q1_new = self.critic1(states, new_actions)
            q2_new = self.critic2(states, new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            # Weighted actor loss using importance sampling weights
            actor_loss = (weights * (self.alpha * log_probs - q_new).squeeze()).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Update alpha (if auto-tuning)
            alpha_loss = 0
            if self.config.alpha_auto_tune:
                # Weighted alpha loss using importance sampling weights
                alpha_loss = -(weights * (self.log_alpha * (log_probs + self.target_entropy).detach()).squeeze()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)
            
            return {
                'critic1_loss': critic1_loss.item(),
                'critic2_loss': critic2_loss.item(),
                'actor_loss': actor_loss.item(),
                'alpha_loss': alpha_loss.item() if self.config.alpha_auto_tune else 0,
                'alpha': self.alpha.item(),
                'mean_q1': current_q1.mean().item(),
                'mean_q2': current_q2.mean().item(),
                'beta': beta,
                'mean_td_error': td_errors.mean().item(),
                'learning_rates': self.get_current_learning_rates()
            }
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA Error in SAC update: {e}")
                print(f"Memory size: {len(self.memory)}")
                print(f"Steps done: {self.steps_done}")
                # Try to recover by clearing CUDA cache
                torch.cuda.empty_cache()
                return {}
            else:
                raise
    
    def _soft_update(self, target, source):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )
    
    def train_episode(self, env) -> Dict[str, float]:
        """Train for one episode using any trading environment"""
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Track update metrics
        update_metrics = {
            'critic1_loss': [],
            'critic2_loss': [],
            'actor_loss': [],
            'alpha_loss': [],
            'alpha': [],
            'mean_q1': [],
            'mean_q2': []
        }
        
        while True:
            # Select action
            action = self.select_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            self.memory.push(state, action, reward, next_state, done)
            
            # Update counters
            episode_reward += reward
            episode_steps += 1
            self.steps_done += 1
            
            # Update every step (if enough samples)
            if self.steps_done % self.config.update_frequency == 0 and len(self.memory) >= self.config.batch_size:
                update_info = self.update()
                # Track metrics
                for key, value in update_info.items():
                    if key in update_metrics:
                        update_metrics[key].append(value)
            
            state = next_state
            
            if done:
                break
        
        self.episodes_done += 1
        
        # Calculate return
        final_value = info['balance'] + (info.get('position', 0) * info['current_price'])
        total_return = (final_value - self.config.initial_balance) / self.config.initial_balance
        
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
            'winning_trades': info['winning_trades'],
            'losing_trades': info['losing_trades'],
            'invalid_actions': info['invalid_actions'],
            **avg_update_metrics
        }
    
    def evaluate_episode(self, env, deterministic: bool = True) -> Dict[str, float]:
        """Evaluate agent on one episode (no training)"""
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # Select action (deterministic for evaluation)
            action = self.select_action(state, deterministic=deterministic)
            next_state, reward, done, info = env.step(action)
            
            # Update counters (no training)
            episode_reward += reward
            episode_steps += 1
            
            state = next_state
            
            if done:
                break
        
        # Calculate return
        final_value = info['balance'] + (info.get('position', 0) * info['current_price'])
        total_return = (final_value - self.config.initial_balance) / self.config.initial_balance
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'total_return': total_return,
            'final_value': final_value,
            'total_trades': info['total_trades'],
            'winning_trades': info['winning_trades'],
            'losing_trades': info['losing_trades'],
            'invalid_actions': info['invalid_actions']
        }
    
    def save(self, path: str, portfolio_normalizer=None):
        """Save model checkpoint"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'config': self.config
        }
        
        # Add alpha-related states if auto-tuning
        if self.config.alpha_auto_tune:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        # Add scheduler states if using schedulers
        if self.config.use_lr_scheduler and self.schedulers:
            scheduler_states = {}
            for name, scheduler in self.schedulers.items():
                scheduler_states[name] = scheduler.state_dict()
            checkpoint['scheduler_states'] = scheduler_states
        
        torch.save(checkpoint, path)
        
        # Save normalizer if provided
        if portfolio_normalizer is not None:
            normalizer_path = path.replace('.pt', '_normalizer.pkl')
            portfolio_normalizer.save(normalizer_path)
    
    def load(self, path: str, portfolio_normalizer=None):
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
        
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        
        # Load alpha-related states if available
        if self.config.alpha_auto_tune and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        # Load scheduler states if available
        if (self.config.use_lr_scheduler and 
            'scheduler_states' in checkpoint and 
            self.schedulers):
            scheduler_states = checkpoint['scheduler_states']
            for name, scheduler in self.schedulers.items():
                if name in scheduler_states:
                    scheduler.load_state_dict(scheduler_states[name])
        
        # Load normalizer if provided
        if portfolio_normalizer is not None:
            normalizer_path = path.replace('.pt', '_normalizer.pkl')
            portfolio_normalizer.load(normalizer_path)
    
    def get_network_summary(self) -> Dict[str, int]:
        """Get summary of network parameters"""
        return {
            'actor_params': count_parameters(self.actor),
            'critic1_params': count_parameters(self.critic1),
            'critic2_params': count_parameters(self.critic2),
            'total_params': (count_parameters(self.actor) + 
                           count_parameters(self.critic1) + 
                           count_parameters(self.critic2)),
            'network_type': self.config.network_type.value,
            'buffer_size': len(self.memory),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done
        }


def create_sac_agent(config: SACConfig, device: torch.device = DEVICE) -> SAC:
    """Factory function to create SAC agent"""
    return SAC(config, device) 