"""
Fixed Simple DQN v5 Reward Parameter Optimizer
==============================================

FIXED VERSION that properly handles portfolio normalization warmup phase.
This ensures fair comparison between old hardcoded normalization and new adaptive normalization.

Key improvements:
- Separates warmup phase from evaluation phase
- Only tracks performance during post-warmup episodes
- Ensures consistent baseline across all trials
- Gives adaptive normalization a fair chance

Usage:
    python simple_reward_optimizer_fixed.py --data-path "data/feature_engineered_v2/TSLA.csv" --cutoff "2021-05-06"
"""

import optuna
import numpy as np
import pandas as pd
import torch
import argparse
from datetime import datetime
from typing import Dict

# Import DQN components
from src.models.mark.dqn_v2.dqn_v5 import DoubleDuelingDQN, TradingEnvironment, TradingMode, load_stock_data
from src.models.mark.dqn_v2.config import TradingConfig
from src.models.mark.dqn_v2.data_preprocessor_v2 import preprocess_financial_data
from src.config.config import DEVICE


def test_parameters_with_warmup(invalid_penalty: float, 
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
                               train_data: pd.DataFrame,
                               scaled_train_data: pd.DataFrame,
                               num_episodes: int = 100) -> Dict[str, float]:
    """Test a parameter configuration with proper warmup handling"""
    
    # Create config with test parameters
    config = TradingConfig()
    config.invalid_penalty = invalid_penalty
    config.portfolio_scaling = portfolio_scaling
    config.learning_rate = learning_rate
    config.epsilon_decay = epsilon_decay
    config.batch_size = batch_size
    config.gamma = gamma
    config.alpha = alpha
    config.tau = tau
    config.epsilon_start = epsilon_start
    config.epsilon_end = epsilon_end
    config.hidden_size = 512  # Fixed based on domain knowledge
    config.update_frequency = update_frequency
    config.min_profit_threshold = min_profit_threshold
    
    try:
        # Create environment and agent
        env = TradingEnvironment(train_data, scaled_train_data, config, mode=TradingMode.TRAIN)
        agent = DoubleDuelingDQN(config)
        
        # 🔧 PHASE 1: WARMUP (if portfolio normalizer exists and needs warmup)
        warmup_episodes = 0
        if (hasattr(env, 'portfolio_normalizer') and 
            env.portfolio_normalizer is not None and 
            not env.portfolio_normalizer.is_fitted):
            
            warmup_episodes = env.portfolio_normalizer.warmup_episodes
            print(f"      🔥 Warmup Phase: {warmup_episodes} episodes (not tracked)")
            print(f"      📊 Initial episode count: {env.portfolio_normalizer.episode_count}")
            
            for warmup_ep in range(warmup_episodes):
                state = env.reset()
                episode_portfolio_states = []
                
                while True:
                    import random
                    action = random.randrange(config.num_actions)
                    next_state, reward, done, info = env.step(action)
                    
                    # Collect portfolio states (following dqn_v5.py pattern)
                    if hasattr(env, 'episode_portfolio_states'):
                        episode_portfolio_states.extend(env.episode_portfolio_states)
                    
                    state = next_state
                    if done:
                        break
                
                if episode_portfolio_states:
                    env.portfolio_normalizer.collect_warmup_data(episode_portfolio_states)
                
                # CRITICAL: Increment episode counter for normalizer
                env.portfolio_normalizer.increment_episode()
                
                # Debug: Show episode count progress and data collection
                if (warmup_ep + 1) % 10 == 0:
                    print(f"      📈 Episode {warmup_ep + 1}: normalizer.episode_count = {env.portfolio_normalizer.episode_count}, states_collected = {len(episode_portfolio_states)}")
                
                # Check if normalizer is fitted after each episode
                if env.portfolio_normalizer.is_fitted:
                    actual_warmup = warmup_ep + 1
                    print(f"      ✅ Normalizer fitted after {actual_warmup} episodes")
                    break
            
            # Verify normalizer is now fitted
            if env.portfolio_normalizer.is_fitted:
                print(f"      🚀 Starting evaluation phase...")
            else:
                print(f"      ❌ ERROR: Normalizer not fitted after warmup!")
                return {
                    'avg_return': -0.1, 
                    'win_rate': 0.0, 
                    'avg_invalid_actions': 100,
                    'avg_trades': 0,
                    'warmup_episodes': warmup_episodes,
                    'score': -10
                }
        else:
            print(f"      ⚡ No warmup needed, starting evaluation...")
        
        # 🎯 PHASE 2: EVALUATION (track performance after warmup)
        returns = []
        invalid_actions_list = []
        trades_list = []
        
        for episode in range(num_episodes):
            if episode % 20 == 0:  # Less frequent progress updates
                print(f"      Eval Episode {episode}/{num_episodes}...")
            
            metrics = agent.train_episode(env)
            returns.append(metrics['total_return'])
            invalid_actions_list.append(metrics['invalid_actions'])
            # Track trades if available
            if 'total_trades' in metrics:
                trades_list.append(metrics['total_trades'])
        
        # Calculate performance metrics (only from evaluation phase)
        avg_return = np.mean(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        avg_invalid = np.mean(invalid_actions_list)
        avg_trades = np.mean(trades_list) if trades_list else 0
        
        # Enhanced scoring function
        score = (
            avg_return * 4.0 +                    # Primary: actual returns
            win_rate * 2.0 +                      # Secondary: consistency  
            -(avg_invalid / 100) * 3.0 +          # Strong penalty: invalid actions
            -abs(avg_trades - 15) * 0.02          # Minor penalty: target ~15 trades/day
        )
        
        return {
            'avg_return': avg_return,
            'win_rate': win_rate,
            'avg_invalid_actions': avg_invalid,
            'avg_trades': avg_trades,
            'warmup_episodes': warmup_episodes,
            'score': score
        }
        
    except Exception as e:
        print(f"      ❌ Error testing config: {e}")
        return {
            'avg_return': -0.1, 
            'win_rate': 0.0, 
            'avg_invalid_actions': 100,
            'avg_trades': 0,
            'warmup_episodes': 0,
            'score': -10
        }


def run_fixed_optimization(data_path: str, cutoff: pd.Timestamp, n_trials: int = 100, 
                          n_episodes: int = 150, scaling_method: str = 'robust', 
                          outlier_method: str = 'winsorize'):
    """Run the fixed optimization with proper warmup handling"""
    
    print("🔧 FIXED DQN v5 Parameter Optimization")
    print("=" * 50)
    print(f"   Trials: {n_trials}")
    print(f"   Episodes per trial: {n_episodes} (post-warmup)")
    print(f"   Warmup: Handled separately (not counted in performance)")
    print(f"   Total episodes: {n_trials * n_episodes:,} (+ warmup)")
    print(f"   Architecture: 512 units (fixed, proven optimal)")
    print(f"   🎯 FAIR COMPARISON: New normalization gets proper warmup")
    
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
        
        # Sample parameters (same ranges as before)
        invalid_penalty = trial.suggest_float('invalid_penalty', 0.1, 0.8)
        portfolio_scaling = trial.suggest_float('portfolio_scaling', 0.005, 0.2)
        learning_rate = trial.suggest_float('learning_rate', 5e-5, 3e-4, log=True)
        epsilon_decay = trial.suggest_int('epsilon_decay', 800, 6000)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
        gamma = trial.suggest_float('gamma', 0.97, 0.9995)
        alpha = trial.suggest_float('alpha', 0.5, 0.75)
        tau = trial.suggest_float('tau', 0.002, 0.015, log=True)
        epsilon_start = trial.suggest_float('epsilon_start', 0.7, 1.0)
        epsilon_end = trial.suggest_float('epsilon_end', 0.002, 0.03)
        update_frequency = trial.suggest_categorical('update_frequency', [1, 2, 4, 8])
        min_profit_threshold = trial.suggest_float('min_profit_threshold', 0.002, 0.03)
        
        print(f"\n🔍 Trial {trial.number + 1}/{n_trials} - FIXED Optimizer")
        print(f"   Penalties: invalid={invalid_penalty:.3f}, portfolio={portfolio_scaling:.3f}")
        print(f"   Learning: lr={learning_rate:.2e}, eps_decay={epsilon_decay}")
        print(f"   Training: batch={batch_size}, gamma={gamma:.3f}, alpha={alpha:.3f}")
        print(f"   Updates: tau={tau:.4f}, freq={update_frequency}")
        print(f"   Exploration: start={epsilon_start:.2f}, end={epsilon_end:.3f}")
        print(f"   Trading: profit_thresh={min_profit_threshold:.3f}")
        print(f"   Architecture: 512 units (FIXED)")
        
        # Test configuration with proper warmup handling
        results = test_parameters_with_warmup(
            invalid_penalty, portfolio_scaling, learning_rate, epsilon_decay,
            batch_size, gamma, alpha, tau, epsilon_start, epsilon_end,
            update_frequency, min_profit_threshold,
            train_data, scaled_train_data, num_episodes=n_episodes
        )
        
        score = results['score']
        warmup = results['warmup_episodes']
        warmup_str = f" (warmup: {warmup})" if warmup > 0 else ""
        
        print(f"   → Return: {results['avg_return']:.2%}, Win: {results['win_rate']:.1%}")
        print(f"   → Invalid: {results['avg_invalid_actions']:.0f}, Trades: {results['avg_trades']:.1f}")
        print(f"   → Score: {score:.3f}{warmup_str}")
        
        # Track best parameters
        if score > best_score:
            best_score = score
            best_params = {
                'invalid_penalty': invalid_penalty,
                'portfolio_scaling': portfolio_scaling,
                'learning_rate': learning_rate,
                'epsilon_decay': epsilon_decay,
                'batch_size': batch_size,
                'gamma': gamma,
                'alpha': alpha,
                'tau': tau,
                'epsilon_start': epsilon_start,
                'epsilon_end': epsilon_end,
                'hidden_size': 512,
                'update_frequency': update_frequency,
                'min_profit_threshold': min_profit_threshold,
                'performance': results
            }
            print(f"   🏆 NEW BEST (warmup-adjusted)!")
        
        return score
    
    # Run optimization
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)
    
    return best_params


def save_fixed_results(best_params: Dict, output_file: str = "FIXED_OPTIMIZED_PARAMETERS.txt", 
                      scaling_method: str = 'robust', outlier_method: str = 'winsorize'):
    """Save fixed optimization results"""
    
    params = best_params
    perf = best_params['performance']
    
    content = f"""
================================================================================
DQN v5 FIXED OPTIMIZED PARAMETERS - WARMUP-AWARE OPTIMIZATION
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Optimizer: FIXED version with proper warmup handling
Preprocessing: {scaling_method} scaling, {outlier_method} outliers

🎯 PERFORMANCE ACHIEVED (POST-WARMUP):
   Average Return: {perf['avg_return']:.2%} per episode
   Win Rate: {perf['win_rate']:.1%} of episodes  
   Invalid Actions: {perf['avg_invalid_actions']:.0f} per episode
   Avg Trades/Day: {perf['avg_trades']:.1f}
   Warmup Episodes: {perf['warmup_episodes']} (not counted in metrics)
   Score: {perf['score']:.3f}

🔧 IMPROVEMENTS OVER ORIGINAL OPTIMIZER:
   ✅ Warmup phase handled separately
   ✅ Performance only measured post-warmup
   ✅ Fair comparison for adaptive normalization
   ✅ Consistent baseline across trials

================================================================================
📋 COPY-PASTE INSTRUCTIONS FOR DQN v5
================================================================================

STEP 1: Open src/models/mark/dqn_v2/dqn_v5.py and find the step() function

STEP 2: Replace portfolio scaling:

FIND:
    portfolio_scaling = getattr(self.config, 'portfolio_scaling', 0.010)

REPLACE WITH:
    portfolio_scaling = getattr(self.config, 'portfolio_scaling', {params['portfolio_scaling']:.4f})

STEP 3: Replace invalid penalty:

FIND:
    invalid_penalty = getattr(self.config, 'invalid_penalty', 0.497)

REPLACE WITH:
    invalid_penalty = getattr(self.config, 'invalid_penalty', {params['invalid_penalty']:.4f})

STEP 4: Open src/models/mark/dqn_v2/config.py and update TradingConfig defaults:

FIND and REPLACE these lines in TradingConfig.__init__():

    portfolio_scaling: float = 0.010
    invalid_penalty: float = 0.497
    learning_rate: float = 2.37e-04
    epsilon_decay: int = 2924
    batch_size: int = 32
    gamma: float = 0.99
    alpha: float = 0.6
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    hidden_size: int = 512
    update_frequency: int = 4
    min_profit_threshold: float = 0.015

REPLACE WITH:

    portfolio_scaling: float = {params['portfolio_scaling']:.4f}
    invalid_penalty: float = {params['invalid_penalty']:.4f}
    learning_rate: float = {params['learning_rate']:.2e}
    epsilon_decay: int = {params['epsilon_decay']}
    batch_size: int = {params['batch_size']}
    gamma: float = {params['gamma']:.4f}
    alpha: float = {params['alpha']:.4f}
    tau: float = {params['tau']:.5f}
    epsilon_start: float = {params['epsilon_start']:.4f}
    epsilon_end: float = {params['epsilon_end']:.5f}
    hidden_size: int = {params['hidden_size']}
    update_frequency: int = {params['update_frequency']}
    min_profit_threshold: float = {params['min_profit_threshold']:.5f}

================================================================================
📊 SUMMARY OF FIXED OPTIMIZED VALUES
================================================================================

🎯 REWARD PARAMETERS:
✅ Portfolio Scaling: {params['portfolio_scaling']:.4f}     (was 0.010)
✅ Invalid Penalty: {params['invalid_penalty']:.4f}        (was 0.497)

🧠 CORE RL PARAMETERS:
✅ Learning Rate: {params['learning_rate']:.2e}       (was 2.37e-04)
✅ Epsilon Decay: {params['epsilon_decay']}            (was 2924)
✅ Batch Size: {params['batch_size']}                 (was 32)
✅ Gamma: {params['gamma']:.4f}                       (was 0.99)
✅ Alpha (PER): {params['alpha']:.4f}                 (was 0.6)
✅ Tau: {params['tau']:.5f}                           (was 0.005)

🔍 EXPLORATION PARAMETERS:
✅ Epsilon Start: {params['epsilon_start']:.4f}       (was 1.0)
✅ Epsilon End: {params['epsilon_end']:.5f}           (was 0.01)

🏗️ ARCHITECTURE & TRAINING:
✅ Hidden Size: {params['hidden_size']}               (FIXED - proven optimal)
✅ Update Frequency: {params['update_frequency']}     (was 4)
✅ Min Profit Threshold: {params['min_profit_threshold']:.5f} (was 0.015)

🎯 EXPECTED IMPROVEMENTS:
• More accurate optimization (warmup not contaminating results)
• Better parameter values for adaptive normalization
• Fairer comparison vs old hardcoded approach
• More reliable performance metrics
• Optimized specifically for post-warmup performance

🔧 TECHNICAL IMPROVEMENTS:
• Warmup phase detection and handling
• Separate evaluation phase tracking
• Enhanced scoring function with trade frequency
• Consistent baseline across all trials
• Adaptive normalization gets fair chance

================================================================================
⚠️  IMPORTANT NOTES:
================================================================================

1. This optimization gives adaptive normalization a FAIR chance by:
   - Not penalizing warmup episodes
   - Only measuring performance post-warmup
   - Ensuring consistent baselines

2. If this performs better than the old approach, it validates that:
   - Adaptive normalization IS superior
   - Previous poor results were due to unfair comparison
   - The warmup contamination was hiding the true performance

3. If this still performs worse, then:
   - The old hardcoded values might have been "accidentally optimal"
   - Consider hybrid approach or different normalization strategies

================================================================================
"""
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\n💾 Fixed optimization results saved to: {output_file}")
    print("📋 This version gives adaptive normalization a fair chance!")


def main():
    parser = argparse.ArgumentParser(description='Fixed DQN v5 optimization with proper warmup')
    parser.add_argument('--data-path', required=True, help='Path to CSV data file')
    parser.add_argument('--cutoff', default='2021-05-06', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials (default: 100)')
    parser.add_argument('--episodes', type=int, default=150, help='Episodes per trial post-warmup (default: 150)')
    parser.add_argument('--scaling', default='robust', choices=['robust', 'standard', 'minmax', 'none'], 
                       help='Data scaling method (default: robust)')
    parser.add_argument('--outliers', default='winsorize', choices=['winsorize', 'clip', 'none'],
                       help='Outlier handling method (default: winsorize)')
    parser.add_argument('--output', default='FIXED_OPTIMIZED_PARAMETERS.txt', help='Output file name')
    
    args = parser.parse_args()
    
    print(f"🎮 Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"🧠 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    print(f"\n🔧 Starting FIXED optimization...")
    print(f"   This version properly handles warmup for fair comparison!")
    
    # Run fixed optimization
    best_params = run_fixed_optimization(
        args.data_path, pd.Timestamp(args.cutoff), args.trials,
        n_episodes=args.episodes, scaling_method=args.scaling, outlier_method=args.outliers
    )
    
    # Show results
    print("\n" + "="*60)
    print("🏆 FIXED OPTIMIZATION COMPLETE!")
    print("="*60)
    
    perf = best_params['performance']
    print(f"Best Performance (Post-Warmup):")
    print(f"  Return: {perf['avg_return']:.2%}")
    print(f"  Win Rate: {perf['win_rate']:.1%}")
    print(f"  Invalid Actions: {perf['avg_invalid_actions']:.0f}")
    print(f"  Avg Trades: {perf['avg_trades']:.1f}")
    print(f"  Warmup Episodes: {perf['warmup_episodes']}")
    print(f"  Score: {perf['score']:.3f}")
    
    print(f"\nOptimal Parameters:")
    for key, value in best_params.items():
        if key != 'performance':
            if isinstance(value, float):
                print(f"  {key} = {value:.4f}")
            else:
                print(f"  {key} = {value}")
    
    # Performance analysis
    if perf['avg_return'] > 0:
        print(f"\n✅ Positive returns achieved! Adaptive normalization working well.")
    if perf['avg_invalid_actions'] < 5:
        print(f"✅ Low invalid actions! Good parameter optimization.")
    if 10 <= perf['avg_trades'] <= 25:
        print(f"✅ Reasonable trading frequency!")
    
    # Save results
    save_fixed_results(best_params, args.output, args.scaling, args.outliers)
    
    print(f"\n🎯 Now run this to compare fairly with old approach!")
    print(f"\n✅ Fixed optimization complete! 🔧🎮")


if __name__ == "__main__":
    main() 