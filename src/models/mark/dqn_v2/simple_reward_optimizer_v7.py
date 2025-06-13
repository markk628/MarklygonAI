"""
DQN v7 Enhanced Hyperparameter Optimizer
========================================

Specialized optimizer for DQN v7 with 7-action space, action masking, and enhanced features.
Optimizes both traditional RL parameters AND new v7-specific parameters.

Key v7 Features Being Optimized:
- 7-action space performance
- Action masking effectiveness  
- Full PER optimization (alpha, beta schedule, epsilon)
- Exploration bonus mechanisms
- Anti-overtrading measures
- Enhanced state representation (30 features)

Usage:
    python simple_reward_optimizer_v7.py --data-path "data/feature_engineered_v2/TSLA.csv" --cutoff "2021-05-06"
"""

import optuna
import numpy as np
import pandas as pd
import torch
import argparse
from datetime import datetime
from typing import Dict

# Import DQN v7 components
from src.models.mark.dqn_v2.dqn_v7 import (
    EnhancedDoubleDuelingDQN, 
    EnhancedTradingEnvironment, 
    TradingMode, 
    load_stock_data,
    EnhancedTradingConfig
)
from src.models.mark.dqn_v2.config import ArchitectureType
from src.models.mark.dqn_v2.data_preprocessor_v2 import preprocess_financial_data
from src.config.config import DEVICE


def test_enhanced_parameters_with_warmup(
                               # Traditional RL parameters
                               invalid_penalty: float,
                               portfolio_scaling: float, 
                               learning_rate: float,
                               epsilon_decay: int,
                               batch_size: int,
                               gamma: float,
                               alpha: float,
                               tau: float,
                               epsilon_start: float,
                               epsilon_end: float,
                               update_frequency: int,
                               min_profit_threshold: float,
                               beta_start: float,
                               beta_end: float,
                               per_epsilon: float,
                               # NEW: DQN v7 specific parameters
                               exploration_bonus: float,
                               position_change_bonus: float,
                               max_exploration_per_episode: float,
                               min_trade_ratio: float,
                               min_steps_between_trades: int,
                               transaction_cost_multiplier: float,
                               min_position_change_pct: float,
                               # Data
                               train_data: pd.DataFrame,
                               scaled_train_data: pd.DataFrame,
                               num_episodes: int = 100) -> Dict[str, float]:
    """Test DQN v7 parameter configuration with proper warmup handling"""
    
    # Create enhanced config with test parameters
    config = EnhancedTradingConfig()
    
    # Traditional RL parameters
    config.invalid_penalty = invalid_penalty
    config.portfolio_scaling = portfolio_scaling
    config.learning_rate = learning_rate
    config.epsilon_decay = epsilon_decay
    config.batch_size = batch_size
    config.gamma = gamma
    config.alpha = alpha
    # PER importance sampling parameters
    config.tau = tau
    config.epsilon_start = epsilon_start
    config.epsilon_end = epsilon_end
    config.update_frequency = update_frequency
    config.min_profit_threshold = min_profit_threshold
    config.beta_start = beta_start
    config.beta_end = beta_end
    config.per_epsilon = per_epsilon
    
    # NEW: DQN v7 specific parameters
    config.exploration_bonus = exploration_bonus
    config.position_change_bonus = position_change_bonus
    config.max_exploration_per_episode = max_exploration_per_episode
    config.min_trade_ratio = min_trade_ratio
    config.min_steps_between_trades = min_steps_between_trades
    config.transaction_cost_multiplier = transaction_cost_multiplier
    config.min_position_change_pct = min_position_change_pct
    
    try:
        # Create enhanced environment and agent
        env = EnhancedTradingEnvironment(train_data, scaled_train_data, config, mode=TradingMode.TRAIN)
        agent = EnhancedDoubleDuelingDQN(config)
        
        # 🔧 PHASE 1: WARMUP (if portfolio normalizer exists and needs warmup)
        warmup_episodes = 0
        if (hasattr(env, 'portfolio_normalizer') and 
            env.portfolio_normalizer is not None and 
            not env.portfolio_normalizer.is_fitted):
            
            warmup_episodes = env.portfolio_normalizer.warmup_episodes
            print(f"      🔥 v7 Warmup Phase: {warmup_episodes} episodes (not tracked)")
            
            for warmup_ep in range(warmup_episodes):
                state = env.reset()
                episode_portfolio_states = []
                
                while True:
                    # Random action from valid actions (v7 feature)
                    valid_actions = env.get_valid_actions()
                    import random
                    action = random.choice(valid_actions) if valid_actions else 0
                    next_state, reward, done, info = env.step(action)
                    
                    # Collect portfolio states
                    if hasattr(env, 'episode_portfolio_states'):
                        episode_portfolio_states.extend(env.episode_portfolio_states)
                    
                    state = next_state
                    if done:
                        break
                
                if episode_portfolio_states:
                    env.portfolio_normalizer.collect_warmup_data(episode_portfolio_states)
                
                env.portfolio_normalizer.increment_episode()
                
                if (warmup_ep + 1) % 10 == 0:
                    print(f"      📈 v7 Episode {warmup_ep + 1}: normalizer.episode_count = {env.portfolio_normalizer.episode_count}")
                
                if env.portfolio_normalizer.is_fitted:
                    actual_warmup = warmup_ep + 1
                    print(f"      ✅ v7 Normalizer fitted after {actual_warmup} episodes")
                    break
            
            if env.portfolio_normalizer.is_fitted:
                print(f"      🚀 Starting v7 evaluation phase...")
            else:
                print(f"      ❌ ERROR: v7 Normalizer not fitted after warmup!")
                return {
                    'avg_return': -0.1, 
                    'win_rate': 0.0, 
                    'avg_invalid_actions': 5,  # v7 should have much lower invalid actions
                    'avg_trades': 0,
                    'avg_exploration_bonus': 0,
                    'avg_valid_actions': 0,
                    'action_usage_diversity': 0,
                    'sharpe_ratio': -1.0,
                    'warmup_episodes': warmup_episodes,
                    'score': -10
                }
        else:
            print(f"      ⚡ v7 No warmup needed, starting evaluation...")
        
        # 🎯 PHASE 2: EVALUATION (track v7-specific performance after warmup)
        returns = []
        portfolio_values = []
        invalid_actions_list = []
        trades_list = []
        exploration_bonuses = []
        valid_actions_counts = []
        action_usage_arrays = []
        
        for episode in range(num_episodes):
            if episode % 20 == 0:
                print(f"      v7 Eval Episode {episode}/{num_episodes}...")
            
            metrics = agent.train_episode(env)
            returns.append(metrics['total_return'])
            portfolio_values.append(metrics['final_value'])
            invalid_actions_list.append(metrics['invalid_actions'])
            trades_list.append(metrics['episode_trades'])  # v7 tracks episode trades
            exploration_bonuses.append(metrics.get('episode_exploration_bonus', 0))
            valid_actions_counts.append(metrics.get('valid_actions_count', 1))
            action_usage_arrays.append(metrics.get('action_usage', [0] * 7))
        
        # Calculate v7-specific performance metrics
        avg_return = np.mean(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        avg_invalid = np.mean(invalid_actions_list)
        avg_trades = np.mean(trades_list)
        avg_exploration_bonus = np.mean(exploration_bonuses)
        avg_valid_actions = np.mean(valid_actions_counts)
        
        # Calculate action usage diversity (how well it uses all 7 actions)
        total_action_usage = np.sum(action_usage_arrays, axis=0)
        total_actions = np.sum(total_action_usage)
        if total_actions > 0:
            action_probs = total_action_usage / total_actions
            # Use entropy to measure action diversity (higher = more diverse)
            action_entropy = -np.sum([p * np.log(p + 1e-8) for p in action_probs if p > 0])
            action_usage_diversity = action_entropy / np.log(7)  # Normalize by max entropy
        else:
            action_usage_diversity = 0.0
        
        # Calculate Sharpe ratio (same as v5)
        if len(returns) > 0:
            portfolio_volatility = np.std(portfolio_values) / np.mean(portfolio_values)
            if portfolio_volatility > 1e-8:
                sharpe_ratio = avg_return / portfolio_volatility
            else:
                sharpe_ratio = avg_return * 10
        else:
            sharpe_ratio = 0.0
        
        # ENHANCED v7 scoring function
        score = (
            avg_return * 5.0 +                       # Primary: actual returns (higher weight for v7)
            sharpe_ratio * 2.5 +                     # Risk-adjusted return
            win_rate * 3.0 +                         # Consistency reward (higher for v7)
            -abs(avg_trades - 12) * 0.015 +          # Target ~12 trades/day for 7-action space
            -avg_invalid * 0.5 +                     # Penalty for invalid actions (should be low in v7)
            action_usage_diversity * 1.0 +           # Reward for using diverse actions
            min(avg_exploration_bonus, 0.005) * 200  # Moderate bonus for exploration (capped)
        )
        # Note: v7 action masking should make invalid actions much lower
        
        return {
            'avg_return': avg_return,
            'win_rate': win_rate,
            'avg_invalid_actions': avg_invalid,
            'avg_trades': avg_trades,
            'avg_exploration_bonus': avg_exploration_bonus,
            'avg_valid_actions': avg_valid_actions,
            'action_usage_diversity': action_usage_diversity,
            'sharpe_ratio': sharpe_ratio,
            'warmup_episodes': warmup_episodes,
            'score': score
        }
        
    except Exception as e:
        print(f"      ❌ Error testing v7 config: {e}")
        return {
            'avg_return': -0.1, 
            'win_rate': 0.0, 
            'avg_invalid_actions': 5,
            'avg_trades': 0,
            'avg_exploration_bonus': 0,
            'avg_valid_actions': 0,
            'action_usage_diversity': 0,
            'sharpe_ratio': -1.0,
            'warmup_episodes': 0,
            'score': -10
        }


def run_enhanced_v7_optimization(data_path: str, cutoff: pd.Timestamp, n_trials: int = 150, 
                                n_episodes: int = 150, scaling_method: str = 'robust', 
                                outlier_method: str = 'winsorize'):
    """Run DQN v7 optimization with enhanced 7-action space parameters"""
    
    print("🚀 DQN v7 Enhanced Parameter Optimization (7-Action Space)")
    print("=" * 60)
    print(f"   Trials: {n_trials}")
    print(f"   Episodes per trial: {n_episodes} (post-warmup)")
    print(f"   Action Space: 7 actions (HOLD, BUY_S/M/L, SELL_S/M/L)")
    print(f"   Action Masking: ✅ Enabled (should reduce invalid actions)")
    print(f"   Enhanced Features: 30 portfolio features vs 13 in v5")
    print(f"   Architecture: 512 units (fixed, proven optimal)")
    print(f"   🎯 NEW: Optimizing exploration bonuses & anti-overtrading")
    print(f"   🎯 NEW: Optimizing action diversity & trade frequency")
    print(f"   🎯 NEW: Full PER optimization (alpha, beta_start/end, epsilon)")
    
    # Load and prepare data
    data, start_date, end_date = load_stock_data(data_path, cutoff)
    train_end = int(len(data) * 0.7)
    train_data = data.iloc[:train_end].copy().reset_index(drop=True)
    
    # Apply preprocessing
    print(f"   Applying preprocessing ({scaling_method} scaling, {outlier_method} outliers)...")
    import time
    start_time = time.time()
    _, scaled_train_data, _, _ = preprocess_financial_data(
        train_data=train_data,
        scaling_method=scaling_method,
        outlier_method=outlier_method,
        save_preprocessor=False
    )
    preprocess_time = time.time() - start_time
    print(f"   Training data: {len(train_data)} rows (preprocessed in {preprocess_time:.1f}s)")
    
    best_score = -999
    best_params = {}
    
    def objective(trial):
        nonlocal best_score, best_params
        
        # Traditional RL parameters (adjusted ranges for v7)
        invalid_penalty = trial.suggest_float('invalid_penalty', 0.05, 0.4)  # Lower range due to action masking
        portfolio_scaling = trial.suggest_float('portfolio_scaling', 0.005, 0.15)  # Adjusted for 7-action rewards
        learning_rate = trial.suggest_float('learning_rate', 3e-5, 5e-4, log=True)
        epsilon_decay = trial.suggest_int('epsilon_decay', 1000, 8000)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        gamma = trial.suggest_float('gamma', 0.975, 0.9995)
        alpha = trial.suggest_float('alpha', 0.5, 0.8)
        # PER importance sampling parameters
        tau = trial.suggest_float('tau', 0.001, 0.02, log=True)
        epsilon_start = trial.suggest_float('epsilon_start', 0.8, 1.0)
        epsilon_end = trial.suggest_float('epsilon_end', 0.001, 0.025)
        update_frequency = trial.suggest_categorical('update_frequency', [1, 2, 4, 8])
        min_profit_threshold = trial.suggest_float('min_profit_threshold', 0.001, 0.025)
        beta_start = trial.suggest_float('beta_start', 0.2, 0.6)
        beta_end = trial.suggest_float('beta_end', 0.8, 1.0)
        per_epsilon = trial.suggest_float('per_epsilon', 0.0001, 0.01, log=True)
        
        # NEW: DQN v7 specific parameters
        exploration_bonus = trial.suggest_float('exploration_bonus', 0.00005, 0.001, log=True)
        position_change_bonus = trial.suggest_float('position_change_bonus', 0.0001, 0.0008, log=True)
        max_exploration_per_episode = trial.suggest_float('max_exploration_per_episode', 0.003, 0.03)
        min_trade_ratio = trial.suggest_float('min_trade_ratio', 0.02, 0.12)
        min_steps_between_trades = trial.suggest_int('min_steps_between_trades', 3, 20)
        transaction_cost_multiplier = trial.suggest_float('transaction_cost_multiplier', 1.0, 2.5)
        min_position_change_pct = trial.suggest_float('min_position_change_pct', 0.008, 0.06)
        
        print(f"\n🔍 v7 Trial {trial.number + 1}/{n_trials} - Enhanced Optimizer")
        print(f"   Traditional: invalid={invalid_penalty:.3f}, portfolio={portfolio_scaling:.3f}")
        print(f"   Learning: lr={learning_rate:.2e}, eps_decay={epsilon_decay}")
        print(f"   Training: batch={batch_size}, gamma={gamma:.3f}, alpha={alpha:.3f}")
        print(f"   PER: beta_start={beta_start:.2f}, beta_end={beta_end:.2f}, epsilon={per_epsilon:.4f}")
        print(f"   Updates: tau={tau:.4f}, freq={update_frequency}")
        print(f"   Exploration: start={epsilon_start:.2f}, end={epsilon_end:.3f}")
        print(f"   🆕 v7 Bonuses: explore={exploration_bonus:.4f}, pos_change={position_change_bonus:.4f}")
        print(f"   🆕 v7 Trading: trade_ratio={min_trade_ratio:.3f}, steps_between={min_steps_between_trades}")
        print(f"   🆕 v7 Costs: cost_mult={transaction_cost_multiplier:.2f}, min_change={min_position_change_pct:.3f}")
        
        # Test configuration
        results = test_enhanced_parameters_with_warmup(
            # Traditional parameters
            invalid_penalty, portfolio_scaling, learning_rate, epsilon_decay,
            batch_size, gamma, alpha, tau, epsilon_start, epsilon_end,
            update_frequency, min_profit_threshold,
            beta_start, beta_end, per_epsilon,
            # NEW: v7 parameters
            exploration_bonus, position_change_bonus, max_exploration_per_episode,
            min_trade_ratio, min_steps_between_trades, transaction_cost_multiplier,
            min_position_change_pct,
            # Data
            train_data, scaled_train_data, num_episodes=n_episodes
        )
        
        score = results['score']
        warmup = results['warmup_episodes']
        warmup_str = f" (warmup: {warmup})" if warmup > 0 else ""
        
        print(f"   → Return: {results['avg_return']:.2%}, Win: {results['win_rate']:.1%}, Sharpe: {results['sharpe_ratio']:.2f}")
        print(f"   → Invalid: {results['avg_invalid_actions']:.1f}, Trades: {results['avg_trades']:.1f}")
        print(f"   → v7 Exploration: {results['avg_exploration_bonus']:.4f}, Valid Actions: {results['avg_valid_actions']:.1f}")
        print(f"   → v7 Action Diversity: {results['action_usage_diversity']:.3f}")
        print(f"   → Score: {score:.3f}{warmup_str}")
        
        # Track best parameters
        if score > best_score:
            best_score = score
            best_params = {
                # Traditional parameters
                'invalid_penalty': invalid_penalty,
                'portfolio_scaling': portfolio_scaling,
                'learning_rate': learning_rate,
                'epsilon_decay': epsilon_decay,
                'batch_size': batch_size,
                'gamma': gamma,
                'alpha': alpha,
                'beta_start': beta_start,
                'beta_end': beta_end,
                'per_epsilon': per_epsilon,
                'tau': tau,
                'epsilon_start': epsilon_start,
                'epsilon_end': epsilon_end,
                'update_frequency': update_frequency,
                'min_profit_threshold': min_profit_threshold,
                # NEW: v7 parameters
                'exploration_bonus': exploration_bonus,
                'position_change_bonus': position_change_bonus,
                'max_exploration_per_episode': max_exploration_per_episode,
                'min_trade_ratio': min_trade_ratio,
                'min_steps_between_trades': min_steps_between_trades,
                'transaction_cost_multiplier': transaction_cost_multiplier,
                'min_position_change_pct': min_position_change_pct,
                'performance': results
            }
            print(f"   🏆 NEW v7 BEST!")
        
        return score
    
    # Run optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)
    
    return best_params


def save_v7_results(best_params: Dict, output_file: str = "DQN_V7_OPTIMIZED_PARAMETERS.txt", 
                   scaling_method: str = 'robust', outlier_method: str = 'winsorize'):
    """Save DQN v7 optimization results"""
    
    params = best_params
    perf = best_params['performance']
    
    content = f"""
================================================================================
DQN v7 ENHANCED OPTIMIZED PARAMETERS - 7-ACTION SPACE OPTIMIZATION
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Optimizer: DQN v7 Enhanced version with 7-action space and action masking
Preprocessing: {scaling_method} scaling, {outlier_method} outliers

v7 PERFORMANCE ACHIEVED (POST-WARMUP):
   Average Return: {perf['avg_return']:.2%} per episode
   Win Rate: {perf['win_rate']:.1%} of episodes  
   Sharpe Ratio: {perf['sharpe_ratio']:.2f} (risk-adjusted return)
   Invalid Actions: {perf['avg_invalid_actions']:.1f} per episode (v7 action masking)
   Avg Trades/Day: {perf['avg_trades']:.1f} (optimized for 7-action space)
   Exploration Bonus: {perf['avg_exploration_bonus']:.4f} per episode
   Avg Valid Actions: {perf['avg_valid_actions']:.1f} (action masking effectiveness)
   Action Diversity: {perf['action_usage_diversity']:.3f} (0-1, how well it uses all 7 actions)
   Warmup Episodes: {perf['warmup_episodes']} (not counted in metrics)
   Score: {perf['score']:.3f}

v7 IMPROVEMENTS OVER v5:
   7-action space optimization (HOLD, BUY_S/M/L, SELL_S/M/L)
   Action masking integration (invalid actions should be <5/day)
   Full PER optimization (alpha, beta schedule, epsilon)
   Exploration bonus tuning for active trading
   Anti-overtrading measures optimization
   Enhanced state features (30 vs 13 portfolio features)
   Action diversity measurement and optimization

================================================================================
COPY-PASTE INSTRUCTIONS FOR DQN v7
================================================================================

STEP 1: Open src/models/mark/dqn_v2/dqn_v7.py and find the EnhancedTradingConfig class

STEP 2: Replace the __init__ method parameters with optimized values:

FIND the EnhancedTradingConfig.__init__ method and REPLACE these lines:

    # Traditional RL parameters
    self.invalid_penalty = 0.497
    self.portfolio_scaling = 0.010
    self.learning_rate = 2.37e-04
    self.epsilon_decay = 2924
    self.batch_size = 32
    self.gamma = 0.99
    self.alpha = 0.6
    self.beta_start = 0.4
    self.beta_end = 1.0
    self.per_epsilon = 0.001
    self.tau = 0.005
    self.epsilon_start = 1.0
    self.epsilon_end = 0.01
    self.update_frequency = 4
    self.min_profit_threshold = 0.015

REPLACE WITH:

    # Optimized traditional RL parameters
    self.invalid_penalty = {params['invalid_penalty']:.4f}
    self.portfolio_scaling = {params['portfolio_scaling']:.4f}
    self.learning_rate = {params['learning_rate']:.2e}
    self.epsilon_decay = {params['epsilon_decay']}
    self.batch_size = {params['batch_size']}
    self.gamma = {params['gamma']:.4f}
    self.alpha = {params['alpha']:.4f}
    self.beta_start = {params['beta_start']:.3f}
    self.beta_end = {params['beta_end']:.3f}
    self.per_epsilon = {params['per_epsilon']:.5f}
    self.tau = {params['tau']:.5f}
    self.epsilon_start = {params['epsilon_start']:.4f}
    self.epsilon_end = {params['epsilon_end']:.5f}
    self.update_frequency = {params['update_frequency']}
    self.min_profit_threshold = {params['min_profit_threshold']:.5f}

STEP 3: Replace v7-specific parameters:

FIND these v7-specific lines:

    # Exploration bonuses
    self.exploration_bonus = 0.0001
    self.position_change_bonus = 0.0002
    self.max_exploration_per_episode = 0.01
    
    # Action masking parameters
    self.min_trade_ratio = 0.05
    
    # Anti-overtrading measures
    self.min_steps_between_trades = 10
    self.transaction_cost_multiplier = 1.5
    self.min_position_change_pct = 0.02

REPLACE WITH:

    # Optimized exploration bonuses
    self.exploration_bonus = {params['exploration_bonus']:.6f}
    self.position_change_bonus = {params['position_change_bonus']:.6f}
    self.max_exploration_per_episode = {params['max_exploration_per_episode']:.5f}
    
    # Optimized action masking parameters
    self.min_trade_ratio = {params['min_trade_ratio']:.5f}
    
    # Optimized anti-overtrading measures
    self.min_steps_between_trades = {params['min_steps_between_trades']}
    self.transaction_cost_multiplier = {params['transaction_cost_multiplier']:.3f}
    self.min_position_change_pct = {params['min_position_change_pct']:.5f}

================================================================================
SUMMARY OF v7 OPTIMIZED VALUES
================================================================================

TRADITIONAL RL PARAMETERS (v7 OPTIMIZED):
Portfolio Scaling: {params['portfolio_scaling']:.4f}     (v5: 0.010)
Invalid Penalty: {params['invalid_penalty']:.4f}        (v5: 0.497, less important in v7)
Learning Rate: {params['learning_rate']:.2e}       (v5: 2.37e-04)
Epsilon Decay: {params['epsilon_decay']}            (v5: 2924)
Batch Size: {params['batch_size']}                 (v5: 32)
Gamma: {params['gamma']:.4f}                       (v5: 0.99)
Alpha (PER Priority): {params['alpha']:.4f}        (v5: 0.6)
Beta Start (PER): {params['beta_start']:.3f}       (v5: 0.4)
Beta End (PER): {params['beta_end']:.3f}           (v5: 1.0)
PER Epsilon: {params['per_epsilon']:.5f}           (v5: 0.001)
Tau: {params['tau']:.5f}                           (v5: 0.005)
Epsilon Start: {params['epsilon_start']:.4f}       (v5: 1.0)
Epsilon End: {params['epsilon_end']:.5f}           (v5: 0.01)
Hidden Size: {params['hidden_size']}               (FIXED - proven optimal)
Update Frequency: {params['update_frequency']}     (v5: 4)
Min Profit Threshold: {params['min_profit_threshold']:.5f} (v5: 0.015)

NEW v7-SPECIFIC PARAMETERS (OPTIMIZED):
Exploration Bonus: {params['exploration_bonus']:.6f}          (default: 0.0001)
Position Change Bonus: {params['position_change_bonus']:.6f}   (default: 0.0002)
Max Exploration/Episode: {params['max_exploration_per_episode']:.5f}    (default: 0.01)
Min Trade Ratio: {params['min_trade_ratio']:.5f}              (default: 0.05)
Min Steps Between Trades: {params['min_steps_between_trades']}           (default: 10)
Transaction Cost Multiplier: {params['transaction_cost_multiplier']:.3f}  (default: 1.5)
Min Position Change %: {params['min_position_change_pct']:.5f}   (default: 0.02)

EXPECTED v7 IMPROVEMENTS:
• Action masking reduces invalid actions to near zero
• Optimized PER parameters for better sample efficiency and stability  
• Optimized exploration bonuses for active trading
• Better anti-overtrading balance
• Enhanced action diversity across 7-action space
• Improved risk-adjusted returns
• More consistent trading patterns

v7 TECHNICAL ADVANTAGES:
• 7-action space provides finer control
• Action masking eliminates most invalid actions
• Enhanced state features (30 vs 13) for better decisions
• Full PER optimization (priority, importance sampling, stability)
• Exploration bonuses encourage active trading
• Anti-overtrading measures prevent exploitation
• Action diversity metrics ensure balanced strategy

================================================================================
v7 OPTIMIZATION NOTES:
================================================================================

1. v7 Action Space Benefits:
   - HOLD: Strategic waiting
   - BUY_S/M/L: Graduated position building (25%/50%/75%)
   - SELL_S/M/L: Graduated position reduction (25%/50%/100%)
   - Better risk management through position sizing

2. Action Masking Impact:
   - Invalid actions should drop from ~260/day to <5/day
   - Agent focuses on profitable decisions vs avoiding penalties
   - Faster training convergence

3. Full PER Optimization:
   - Alpha: Controls priority strength (0.5-0.8, higher = more prioritization)
   - Beta Start/End: Importance sampling correction schedule (0.2-0.6 → 0.8-1.0)
   - PER Epsilon: Small constant for priority stability (0.0001-0.01)
   - Optimizes sample efficiency and training stability

4. Exploration Optimization:
   - Balances active trading vs overtrading
   - Encourages position size changes
   - Caps total exploration bonus to prevent exploitation

5. If v7 performance >>> v5:
   - Validates 7-action space approach
   - Action masking significantly improves training
   - Enhanced features provide better decision-making

6. If performance similar to v5:
   - May indicate diminishing returns from complexity
   - Consider simpler action spaces for production
   - Focus on the most effective v7 features

================================================================================
"""
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\n💾 DQN v7 optimization results saved to: {output_file}")
    print("🚀 This should significantly improve v7 performance!")


def main():
    parser = argparse.ArgumentParser(description='DQN v7 Enhanced optimization with 7-action space')
    parser.add_argument('--data-path', required=True, help='Path to CSV data file')
    parser.add_argument('--cutoff', default='2021-05-06', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--trials', type=int, default=150, help='Number of trials (default: 150)')
    parser.add_argument('--episodes', type=int, default=150, help='Episodes per trial post-warmup (default: 150)')
    parser.add_argument('--scaling', default='robust', choices=['robust', 'standard', 'minmax', 'none'], 
                       help='Data scaling method (default: robust)')
    parser.add_argument('--outliers', default='winsorize', choices=['winsorize', 'clip', 'none'],
                       help='Outlier handling method (default: winsorize)')
    parser.add_argument('--output', default='DQN_V7_OPTIMIZED_PARAMETERS.txt', help='Output file name')
    
    args = parser.parse_args()
    
    print(f"🎮 Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"🧠 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    print(f"\n🚀 Starting DQN v7 Enhanced optimization...")
    print(f"   This version optimizes 7-action space + action masking!")
    
    # Run v7 optimization
    best_params = run_enhanced_v7_optimization(
        args.data_path, pd.Timestamp(args.cutoff), args.trials,
        n_episodes=args.episodes, scaling_method=args.scaling, outlier_method=args.outliers
    )
    
    # Show results
    print("\n" + "="*60)
    print("🏆 DQN v7 ENHANCED OPTIMIZATION COMPLETE!")
    print("="*60)
    
    perf = best_params['performance']
    print(f"Best v7 Performance (Post-Warmup):")
    print(f"  Return: {perf['avg_return']:.2%}")
    print(f"  Win Rate: {perf['win_rate']:.1%}")
    print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"  Invalid Actions: {perf['avg_invalid_actions']:.1f} (action masking)")
    print(f"  Avg Trades: {perf['avg_trades']:.1f}")
    print(f"  Exploration Bonus: {perf['avg_exploration_bonus']:.4f}")
    print(f"  Action Diversity: {perf['action_usage_diversity']:.3f}")
    print(f"  Warmup Episodes: {perf['warmup_episodes']}")
    print(f"  Score: {perf['score']:.3f}")
    
    print(f"\nOptimal v7 Parameters:")
    for key, value in best_params.items():
        if key != 'performance':
            if isinstance(value, float):
                print(f"  {key} = {value:.4f}")
            else:
                print(f"  {key} = {value}")
    
    # v7 Performance analysis
    print(f"\n🚀 DQN v7 Analysis:")
    if perf['avg_return'] > 0:
        print(f"✅ Positive returns achieved! v7 working well.")
    if perf['sharpe_ratio'] > 1.0:
        print(f"✅ Excellent risk-adjusted performance! Sharpe ratio > 1.0")
    elif perf['sharpe_ratio'] > 0.5:
        print(f"✅ Good risk-adjusted performance! Sharpe ratio > 0.5")
    if perf['avg_invalid_actions'] < 5:
        print(f"✅ Action masking working! Very low invalid actions.")
    if 8 <= perf['avg_trades'] <= 20:
        print(f"✅ Good trading frequency for 7-action space!")
    if perf['action_usage_diversity'] > 0.6:
        print(f"✅ Good action diversity! Using multiple actions effectively.")
    if perf['avg_exploration_bonus'] > 0.001:
        print(f"✅ Active exploration bonuses promoting trading!")
    
    # Save results
    save_v7_results(best_params, args.output, args.scaling, args.outliers)
    
    print(f"\n🎯 Now apply these optimized parameters to DQN v7!")
    print(f"\n✅ DQN v7 optimization complete! 🚀🎮")


if __name__ == "__main__":
    main() 


'''
→ Return: -0.63%, Win: 36.7%, Sharpe: -0.29
   → Invalid: 0.0, Trades: 19.1
   → v7 Exploration: 0.0185, Valid Actions: 2.3
   → v7 Action Diversity: 0.141
   → Score: 1.373 (warmup: 50)
   🏆 NEW v7 BEST!
[I 2025-06-13 07:28:09,798] Trial 26 finished with value: 1.3727331512598318 and parameters: {'invalid_penalty': 0.12098424998368312, 'portfolio_scaling': 0.04484302295620472, 'learning_rate': 0.00021766506642734341, 'epsilon_decay': 4676, 'batch_size': 32, 'gamma': 0.9771033411147915, 'alpha': 0.550412414151707, 'tau': 0.0130720145688393, 'epsilon_start': 0.9776414845085334, 'epsilon_end': 0.011846778895697132, 'update_frequency': 2, 'min_profit_threshold': 0.009935316689040505, 'beta_start': 0.4799425310108217, 'beta_end': 0.884781915735828, 'per_epsilon': 0.0006319935698127862, 'exploration_bonus': 0.0007545451542850604, 'position_change_bonus': 0.00025850149832639573, 'max_exploration_per_episode': 0.021592806316304235, 'min_trade_ratio': 0.03864282626355591, 'min_steps_between_trades': 14, 'transaction_cost_multiplier': 1.168330018667315, 'min_position_change_pct': 0.05519495195657326}. Best is trial 26 with value: 1.3727331512598318.
'''