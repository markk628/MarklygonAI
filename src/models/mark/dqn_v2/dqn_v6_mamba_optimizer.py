"""
DQN v6 Comprehensive Mamba Optimizer
===================================

Comprehensive hyperparameter optimizer for DQN v6's Mamba architecture.
Optimizes all applicable hyperparameters like the DQN v5 optimizer.
NOTE: update_frequency removed - Mamba updates every step, uses tau for target updates.

Usage:
    python dqn_v6_mamba_optimizer.py --data-path "data/feature_engineered_v2/TSLA.csv" --cutoff "2021-05-06" --trials 100
"""

import optuna
import numpy as np
import pandas as pd
import torch
import argparse
from datetime import datetime
from typing import Dict

# Import DQN v6 components (UPDATED IMPORTS)
from src.models.mark.dqn_v2.dqn_v6 import MambaDQN, MambaEnvironment, MambaTradingConfig, MAMBA_FEATURES
from src.models.mark.dqn_v2.data_preprocessor_v2 import preprocess_financial_data
from src.config.config import DEVICE

# Features that must exist in the raw data (excludes dynamically computed ones)
RAW_DATA_FEATURES = [
    'close',                    # Current price (for trading calculations)
    'return_1m',               # 1-minute price momentum  
    'return_5m',               # 5-minute price momentum
    'return_15m',              # 15-minute price momentum
    'volume_ratio_5m',         # Volume relative to recent average
    'volatility_5m',           # Recent volatility measure
    'rsi_14m',                 # RSI for momentum detection
    'macd',                    # MACD for trend analysis
    'hour_sin',                # Time of day (cyclical)
    # NOTE: 'close_normalized' excluded - computed dynamically by MambaEnvironment
]


def test_mamba_parameters(invalid_penalty: float,
                         portfolio_scaling: float, 
                         learning_rate: float,
                         epsilon_decay: int,
                         batch_size: int,
                         gamma: float,
                         alpha: float,
                         beta_start: float,
                         beta_end: float,
                         tau: float,
                         epsilon_start: float,
                         epsilon_end: float,
                         min_profit_threshold: float,
                         d_model: int,
                         n_layers: int,
                         temporal_window: int,
                         train_data: pd.DataFrame,
                         scaled_train_data: pd.DataFrame,
                         num_episodes: int = 50) -> Dict[str, float]:
    """Test comprehensive Mamba parameter configuration (15 parameters)"""
    
    # Create Mamba config with test parameters
    config = MambaTradingConfig(temporal_window=temporal_window)
    
    # Apply core RL parameters
    config.learning_rate = learning_rate
    config.epsilon_decay = epsilon_decay
    config.batch_size = batch_size
    config.gamma = gamma
    config.alpha = alpha
    config.beta_start = beta_start
    config.beta_end = beta_end
    config.tau = tau
    config.epsilon_start = epsilon_start
    config.epsilon_end = epsilon_end
    config.min_profit_threshold = min_profit_threshold
    
    # Apply Mamba-specific architecture parameters
    config.d_model = d_model
    config.n_layers = n_layers
    
    # Add reward parameters (these don't exist in config by default)
    config.invalid_penalty = invalid_penalty
    config.portfolio_scaling = portfolio_scaling
    
    try:
        # Apply preprocessing for better performance
        print(f"    Applying data preprocessing...")
        _, scaled_data, _, _ = preprocess_financial_data(
            train_data=train_data,
            scaling_method='robust',
            outlier_method='winsorize', 
            save_preprocessor=False
        )
        
        # Create environment and agent with preprocessed data
        env = MambaEnvironment(train_data, scaled_data, config)
        agent = MambaDQN(config)
        
        # Track results
        returns = []
        portfolio_values = []
        invalid_actions_list = []
        trades_list = []
        rewards_list = []
        
        print(f"    Training {num_episodes} episodes with Mamba...")
        
        for episode in range(num_episodes):
            if episode % 10 == 0:  # Progress every 10 episodes
                print(f"      Episode {episode}/{num_episodes}...")
            
            # Train episode
            results = agent.train_episode(env)
            
            returns.append(results['episode_return'])
            portfolio_values.append(results['final_value'])  # Track portfolio values for Sharpe calculation
            invalid_actions_list.append(results['invalid_actions'])
            trades_list.append(results['total_trades'])
            rewards_list.append(results['episode_reward'])
        
        # Calculate performance metrics
        avg_return = np.mean(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        avg_invalid = np.mean(invalid_actions_list)
        avg_trades = np.mean(trades_list)
        avg_reward = np.mean(rewards_list)
        
        # Calculate Sharpe ratio (risk-adjusted return)
        if len(returns) > 0:
            portfolio_volatility = np.std(portfolio_values) / np.mean(portfolio_values)
            if portfolio_volatility > 1e-8:
                sharpe_ratio = avg_return / portfolio_volatility
            else:
                sharpe_ratio = avg_return * 10
        else:
            sharpe_ratio = 0.0
        
        # Enhanced scoring function with Sharpe ratio (PURE FINANCIAL FOCUS)
        score = (
            avg_return * 4.0 +                    # Primary: actual returns
            sharpe_ratio * 2.0 +                  # Risk-adjusted return metric
            win_rate * 2.5 +                      # Consistency reward
            -abs(avg_trades - 20) * 0.05          # Minor penalty: target ~20 trades/day
        )
        # Note: Invalid actions removed from scoring - they're handled by reward penalties during training
        
        return {
            'avg_return': avg_return,
            'win_rate': win_rate,
            'avg_invalid_actions': avg_invalid,  # Still track for diagnostics
            'avg_trades': avg_trades,
            'avg_reward': avg_reward,
            'sharpe_ratio': sharpe_ratio,
            'score': score
        }
        
    except Exception as e:
        print(f"    Error testing Mamba config: {e}")
        # Calculate error Sharpe ratio same way as main method
        error_portfolio_values = [config.initial_balance * 0.9] * 10  # Simulate losing portfolio
        error_portfolio_volatility = np.std(error_portfolio_values) / np.mean(error_portfolio_values)
        if error_portfolio_volatility > 1e-8:
            error_sharpe = -0.1 / error_portfolio_volatility 
        else:
            error_sharpe = -0.1 * 10
        
        return {
            'avg_return': -0.1,
            'win_rate': 0.0, 
            'avg_invalid_actions': 100,  # Still track for diagnostics
            'avg_trades': 0,
            'avg_reward': -50,
            'sharpe_ratio': error_sharpe,
            'score': -10  # Error score unchanged - pure penalty
        }


def run_comprehensive_mamba_optimization(data_path: str, cutoff: pd.Timestamp, n_trials: int = 100,
                                        num_episodes: int = 50, scaling_method: str = 'robust', 
                                        outlier_method: str = 'winsorize') -> Dict:
    """Run comprehensive Mamba optimization"""
    
    print("🔥 DQN v6 Comprehensive Mamba Hyperparameter Optimization")
    print(f"   Architecture: Mamba SSM (State Space Model)")
    print(f"   Trials: {n_trials}")
    print(f"   Episodes per trial: {num_episodes}")
    print(f"   Preprocessing: {scaling_method} scaling, {outlier_method} outliers")
    print(f"   Total training episodes: {n_trials * num_episodes:,}")
    print(f"   📊 RISK-ADJUSTED: Scoring includes Sharpe ratio")
    print(f"   💰 PURE FINANCIAL: Invalid actions removed from scoring (handled by training rewards)")
    print(f"   NOTE: update_frequency excluded - Mamba updates every step, uses tau for targets")
    
    # Load and prepare data
    print("   Loading data...")
    data = pd.read_csv(data_path)
    print(f"   Data shape: {data.shape}")
    
    # Validate required features for DQN v6 (exclude dynamically computed ones)
    missing_features = [f for f in RAW_DATA_FEATURES if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing required raw data features for DQN v6: {missing_features}")
    
    # Use portion of data for optimization
    data_timestamps = pd.to_datetime(data['timestamp'])
    if data_timestamps.dt.tz is not None:
        cutoff = cutoff.tz_localize('UTC') if cutoff.tz is None else cutoff.tz_convert('UTC')
    
    cutoff_data = data[data_timestamps >= cutoff].reset_index(drop=True)
    train_end = int(len(cutoff_data) * 0.7)
    train_data = cutoff_data.iloc[:train_end].copy().reset_index(drop=True)
    
    print(f"   Training data: {len(train_data)} rows")
    print(f"   Features available: {[f for f in MAMBA_FEATURES if f in data.columns]}")
    
    best_score = -999
    best_params = {}
    
    def objective(trial):
        nonlocal best_score, best_params
        
        # 🎯 COMPREHENSIVE PARAMETER SAMPLING (15 parameters total - no update_frequency)
        
        # Reward parameters (2)
        invalid_penalty = trial.suggest_float('invalid_penalty', 0.001, 0.1)
        portfolio_scaling = trial.suggest_float('portfolio_scaling', 0.5, 10.0)
        
        # Core RL parameters (7) - removed update_frequency since Mamba updates every step
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        epsilon_decay = trial.suggest_int('epsilon_decay', 1000, 4000)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        gamma = trial.suggest_float('gamma', 0.97, 0.9995)
        alpha = trial.suggest_float('alpha', 0.5, 0.8)  # PER exponent
        beta_start = trial.suggest_float('beta_start', 0.3, 0.5)  # PER importance sampling
        beta_end = trial.suggest_float('beta_end', 0.9, 1.0)
        tau = trial.suggest_float('tau', 0.002, 0.02, log=True)  # Soft target update
        epsilon_start = trial.suggest_float('epsilon_start', 0.7, 1.0)
        epsilon_end = trial.suggest_float('epsilon_end', 0.005, 0.05)
        min_profit_threshold = trial.suggest_float('min_profit_threshold', 0.001, 0.03)
        
        # Mamba-specific architecture parameters (3)
        d_model = trial.suggest_categorical('d_model', [32, 64, 96, 128])  # Model dimension
        n_layers = trial.suggest_int('n_layers', 1, 4)  # Number of SSM layers
        temporal_window = trial.suggest_int('temporal_window', 15, 45)  # Lookback window
        
        print(f"\n🧠 Trial {trial.number + 1}/{n_trials} - Comprehensive Mamba (15 params)")
        print(f"   Rewards: invalid_penalty={invalid_penalty:.4f}, portfolio_scaling={portfolio_scaling:.2f}")
        print(f"   RL Core: lr={learning_rate:.2e}, eps_decay={epsilon_decay}, batch={batch_size}")
        print(f"   RL More: gamma={gamma:.3f}, alpha={alpha:.3f}, tau={tau:.4f}")
        print(f"   Exploration: eps_start={epsilon_start:.2f}, eps_end={epsilon_end:.3f}")
        print(f"   PER: beta_start={beta_start:.2f}, beta_end={beta_end:.2f}")
        print(f"   Trading: min_profit={min_profit_threshold:.4f}")
        print(f"   Mamba: d_model={d_model}, n_layers={n_layers}, temporal_window={temporal_window}")
        print(f"   NOTE: No update_frequency - Mamba updates every step")
        
        # Test configuration
        results = test_mamba_parameters(
            invalid_penalty, portfolio_scaling, learning_rate, epsilon_decay,
            batch_size, gamma, alpha, beta_start, beta_end, tau,
            epsilon_start, epsilon_end, min_profit_threshold,
            d_model, n_layers, temporal_window,
            train_data, train_data, num_episodes  # Use original data for both
        )
        
        score = results['score']
        print(f"   → Return: {results['avg_return']:.2%}, Win: {results['win_rate']:.1%}, Sharpe: {results['sharpe_ratio']:.2f}")
        print(f"   → Trades: {results['avg_trades']:.1f}, Invalid: {results['avg_invalid_actions']:.0f}")
        print(f"   → Score: {score:.3f}")
        
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
                'beta_start': beta_start,
                'beta_end': beta_end,
                'tau': tau,
                'epsilon_start': epsilon_start,
                'epsilon_end': epsilon_end,
                'min_profit_threshold': min_profit_threshold,
                'd_model': d_model,
                'n_layers': n_layers,
                'temporal_window': temporal_window,
                'performance': results
            }
            print(f"   🏆 NEW BEST COMPREHENSIVE MAMBA CONFIG!")
        
        return score
    
    # Run optimization with TPE sampler
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)
    
    return best_params


def save_comprehensive_mamba_results(best_params: Dict, output_file: str = "COMPREHENSIVE_MAMBA_OPTIMIZED_PARAMETERS.txt",
                                    scaling_method: str = 'robust', outlier_method: str = 'winsorize'):
    """Save comprehensive Mamba optimization results to text file"""
    
    params = best_params
    perf = best_params['performance']
    
    content = f"""
================================================================================
DQN v6 COMPREHENSIVE MAMBA OPTIMIZED PARAMETERS - READY TO COPY-PASTE
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Architecture: Mamba SSM (State Space Model)
Preprocessing: {scaling_method} scaling + {outlier_method} outliers
Parameters Optimized: 15 (comprehensive, excluding update_frequency)

🎯 PERFORMANCE ACHIEVED:
   Average Return: {perf['avg_return']:.2%} per episode
   Win Rate: {perf['win_rate']:.1%} of episodes  
   Sharpe Ratio: {perf['sharpe_ratio']:.2f} (risk-adjusted return)
   Trading Frequency: {perf['avg_trades']:.1f} trades per day
   Invalid Actions: {perf['avg_invalid_actions']:.0f} per episode
   Score: {perf['score']:.3f}

================================================================================
📋 COPY-PASTE INSTRUCTIONS FOR DQN v6
================================================================================

STEP 1: Open dqn_v6.py and find the MambaEnvironment.step() function

STEP 2: Modify the reward calculations:

FIND this section in step():
    if action not in valid_actions:
        invalid_action = True
        reward = -0.01  # Small penalty for invalid action

REPLACE WITH:
    if action not in valid_actions:
        invalid_action = True
        reward = -{params['invalid_penalty']:.4f}  # Optimized invalid penalty

STEP 3: Find the portfolio change reward section:

FIND:
    portfolio_change = current_portfolio_value - self.last_portfolio_value
    reward = portfolio_change / self.config.initial_balance * 100

REPLACE WITH:
    portfolio_change = current_portfolio_value - self.last_portfolio_value
    reward = portfolio_change / self.config.initial_balance * {params['portfolio_scaling']:.2f}

STEP 4: Find the P&L reward section:

FIND:
    reward = profit / self.config.initial_balance * 100

REPLACE WITH:
    reward = profit / self.config.initial_balance * {params['portfolio_scaling']:.2f}

STEP 5: Update MambaTradingConfig defaults:

FIND in MambaTradingConfig.__init__():
    self.learning_rate = 1e-3
    self.batch_size = 64
    self.alpha = 0.6
    self.beta_start = 0.4
    self.beta_end = 1.0
    self.epsilon_start = 1.0
    self.epsilon_end = 0.01
    self.epsilon_decay = 2000
    self.tau = 0.005
    self.gamma = 0.99
    self.min_profit_threshold = 0.01
    self.d_model = 64
    self.n_layers = 2

REPLACE WITH:
    self.learning_rate = {params['learning_rate']:.2e}
    self.batch_size = {params['batch_size']}
    self.alpha = {params['alpha']:.3f}
    self.beta_start = {params['beta_start']:.3f}
    self.beta_end = {params['beta_end']:.3f}
    self.epsilon_start = {params['epsilon_start']:.3f}
    self.epsilon_end = {params['epsilon_end']:.4f}
    self.epsilon_decay = {params['epsilon_decay']}
    self.tau = {params['tau']:.4f}
    self.gamma = {params['gamma']:.4f}
    self.min_profit_threshold = {params['min_profit_threshold']:.4f}
    self.d_model = {params['d_model']}
    self.n_layers = {params['n_layers']}

STEP 6: Update temporal_window when calling MambaTradingConfig:

FIND:
    config = MambaTradingConfig(temporal_window=30)

REPLACE WITH:
    config = MambaTradingConfig(temporal_window={params['temporal_window']})

================================================================================
📊 SUMMARY OF COMPREHENSIVE OPTIMIZED VALUES
================================================================================

🎯 REWARD PARAMETERS (2):
✅ Invalid Penalty: {params['invalid_penalty']:.4f}        (was 0.01)
✅ Portfolio Scaling: {params['portfolio_scaling']:.2f}         (was 100)

🧠 CORE RL PARAMETERS (7): # NOTE: update_frequency excluded - Mamba updates every step
✅ Learning Rate: {params['learning_rate']:.2e}          (was 1e-3)
✅ Epsilon Decay: {params['epsilon_decay']}              (was 2000)
✅ Batch Size: {params['batch_size']}                   (was 64)
✅ Gamma: {params['gamma']:.4f}                        (was 0.99)
✅ Alpha (PER): {params['alpha']:.3f}                   (was 0.6)
✅ Beta Start: {params['beta_start']:.3f}               (was 0.4)
✅ Beta End: {params['beta_end']:.3f}                   (was 1.0)
✅ Tau: {params['tau']:.4f}                            (was 0.005)
✅ Epsilon Start: {params['epsilon_start']:.3f}         (was 1.0)
✅ Epsilon End: {params['epsilon_end']:.4f}             (was 0.01)
✅ Min Profit Threshold: {params['min_profit_threshold']:.4f}   (was 0.01)

🏗️ MAMBA ARCHITECTURE PARAMETERS (3):
✅ D Model: {params['d_model']}                         (was 64)
✅ N Layers: {params['n_layers']}                       (was 2)
✅ Temporal Window: {params['temporal_window']}         (was 30)

🎯 EXPECTED IMPROVEMENTS:
• Optimized Mamba architecture for better temporal pattern recognition
• Balanced exploration-exploitation for better convergence
• Tuned PER parameters for better experience replay
• Optimized reward scaling for better learning signals
• Architecture tuned for trading-specific temporal patterns
• Risk-adjusted optimization using Sharpe ratio
• Pure financial focus (invalid actions excluded from scoring)

🧠 MAMBA-SPECIFIC OPTIMIZATIONS:
• d_model and n_layers tuned for financial time series
• temporal_window optimized for minute-level trading patterns
• tau optimized for stable Mamba training (target network updates)
• All parameters comprehensively optimized together
• No update_frequency needed - Mamba updates every step by design

Total Parameters Optimized: 15 (comprehensive, Mamba-specific)

⚠️ NOTE: update_frequency excluded because Mamba DQN updates every step,
         unlike DQN v5 which uses update_frequency to control training frequency.
         Target network updates handled by tau (soft) and target_update_frequency (hard).

================================================================================
"""
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\n💾 Comprehensive Mamba optimization results saved to: {output_file}")
    print("📋 Open the file and copy-paste the values into dqn_v6.py!")


def main():
    parser = argparse.ArgumentParser(description='DQN v6 comprehensive Mamba optimization')
    parser.add_argument('--data-path', required=True, help='Path to CSV data file')
    parser.add_argument('--cutoff', default='2021-05-06', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials (default: 100)')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per trial (default: 50)')
    parser.add_argument('--scaling', default='robust', choices=['robust', 'standard', 'minmax', 'none'], 
                       help='Data scaling method (default: robust)')
    parser.add_argument('--outliers', default='winsorize', choices=['winsorize', 'clip', 'none'],
                       help='Outlier handling method (default: winsorize)')
    parser.add_argument('--output', default='COMPREHENSIVE_MAMBA_OPTIMIZED_PARAMETERS.txt', help='Output file name')
    
    args = parser.parse_args()
    
    # Validate CUDA availability for Mamba (computationally intensive)
    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA not available. Comprehensive Mamba optimization will be slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    print(f"🎮 Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"🧠 Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Run comprehensive optimization
    print("\nStarting comprehensive Mamba optimization...")
    best_params = run_comprehensive_mamba_optimization(
        args.data_path, 
        pd.Timestamp(args.cutoff), 
        n_trials=args.trials,
        num_episodes=args.episodes,
        scaling_method=args.scaling,
        outlier_method=args.outliers
    )
    
    # Show results
    print("\n" + "="*60)
    print("🏆 COMPREHENSIVE MAMBA OPTIMIZATION COMPLETE!")
    print("="*60)
    
    perf = best_params['performance']
    print(f"Best Comprehensive Mamba Performance:")
    print(f"  Return: {perf['avg_return']:.2%}")
    print(f"  Win Rate: {perf['win_rate']:.1%}")
    print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"  Trading Frequency: {perf['avg_trades']:.1f} trades/day")
    print(f"  Invalid Actions: {perf['avg_invalid_actions']:.0f}")
    
    print(f"\nOptimal Comprehensive Mamba Parameters (15 total):")
    print(f"  🎯 Reward: invalid_penalty={best_params['invalid_penalty']:.4f}, portfolio_scaling={best_params['portfolio_scaling']:.2f}")
    print(f"  🧠 RL Core: lr={best_params['learning_rate']:.2e}, eps_decay={best_params['epsilon_decay']}, batch={best_params['batch_size']}")
    print(f"  🧠 RL More: gamma={best_params['gamma']:.3f}, alpha={best_params['alpha']:.3f}, tau={best_params['tau']:.4f}")
    print(f"  🔍 Explore: eps_start={best_params['epsilon_start']:.2f}, eps_end={best_params['epsilon_end']:.3f}")
    print(f"  📚 PER: beta_start={best_params['beta_start']:.2f}, beta_end={best_params['beta_end']:.2f}")
    print(f"  💰 Trading: min_profit={best_params['min_profit_threshold']:.4f}")
    print(f"  🏗️ Mamba: d_model={best_params['d_model']}, n_layers={best_params['n_layers']}, temporal_window={best_params['temporal_window']}")
    print(f"  ⚡ NOTE: No update_frequency - Mamba updates every step, tau handles target updates")
    
    # Trading frequency analysis
    if perf['avg_trades'] < 10:
        print(f"\n⚠️  Low trading frequency ({perf['avg_trades']:.1f})")
        print("   Consider running more trials or adjusting parameter ranges")
    elif perf['avg_trades'] > 50:
        print(f"\n⚠️  High trading frequency ({perf['avg_trades']:.1f})")
        print("   May be overtrading - consider transaction costs")
    else:
        print(f"\n✅ Good trading frequency ({perf['avg_trades']:.1f})")
        print("   Balanced between holding and active trading")
    
    # Save results
    save_comprehensive_mamba_results(best_params, args.output, args.scaling, args.outliers)
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Apply the 15 optimized parameters to dqn_v6.py")
    print(f"   2. Run full Mamba training with comprehensive optimization")
    print(f"   3. Compare against previous Mamba and DQN v5 results")
    print(f"   4. The comprehensive optimization should perform much better!")
    print(f"\n✅ Comprehensive Mamba optimization complete! 🧠🎮")
    print(f"   ⚡ Architecture optimized: Mamba updates every step, tau handles target updates!")


if __name__ == "__main__":
    main() 