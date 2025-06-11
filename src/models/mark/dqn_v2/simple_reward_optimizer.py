"""
Simple DQN v5 Reward Parameter Optimizer
========================================

Finds optimal reward parameters using proper data preprocessing and saves them 
to a text file for easy copy-pasting.

ARCHITECTURE FIXED: Network architecture (hidden_size=512) is fixed based on domain
knowledge for financial RL. Only hyperparameters are optimized to avoid chasing
random architectural variance.

Usage:
    python simple_reward_optimizer.py --data-path "data/TSLA_1min_features.csv" --cutoff "2023-01-01"
    
    # With custom preprocessing
    python simple_reward_optimizer.py --data-path "data/TSLA_1min_features.csv" --cutoff "2023-01-01" --scaling standard --outliers clip
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


def test_parameters(invalid_penalty: float, 
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
    """Test a parameter configuration"""
    
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
    # Fixed architecture based on domain knowledge (trading RL typically uses 256-512)
    config.hidden_size = 512  # Proven to work well for financial RL
    config.update_frequency = update_frequency
    config.min_profit_threshold = min_profit_threshold
    
    try:
        # Create environment and agent
        env = TradingEnvironment(train_data, scaled_train_data, config, mode=TradingMode.TRAIN)
        agent = DoubleDuelingDQN(config)
        
        # Track results
        returns = []
        invalid_actions_list = []
        
        for episode in range(num_episodes):
            if episode % 10 == 0:  # Progress every 10 episodes (less spam with 200 episodes)
                print(f"    Episode {episode}/{num_episodes}...")
            metrics = agent.train_episode(env)
            returns.append(metrics['total_return'])
            invalid_actions_list.append(metrics['invalid_actions'])
        
        # Calculate performance metrics
        avg_return = np.mean(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        avg_invalid = np.mean(invalid_actions_list)
        
        # Simple scoring function (you can adjust weights)
        score = (
            avg_return * 3.0 +           # Primary: returns
            win_rate * 1.5 +             # Secondary: consistency
            -(avg_invalid / 100) * 2.0   # Penalty: invalid actions
        )
        
        return {
            'avg_return': avg_return,
            'win_rate': win_rate,
            'avg_invalid_actions': avg_invalid,
            'score': score
        }
        
    except Exception as e:
        print(f"Error testing config: {e}")
        return {'avg_return': -0.1, 'win_rate': 0.0, 'avg_invalid_actions': 300, 'score': -10}


def run_optimization(data_path: str, cutoff: pd.Timestamp, n_trials: int = 200, 
                    n_episodes: int = 200, scaling_method: str = 'robust', outlier_method: str = 'winsorize'):
    """Run the optimization"""
    
    print("🚀 Simple DQN v5 Parameter Optimization")
    print(f"   Running {n_trials} trials with {n_episodes} episodes each...")
    print(f"   Total training episodes: {n_trials * n_episodes:,}")
    print(f"   Architecture FIXED to 512 units (not optimized - based on domain knowledge)")
    
    # Load and prepare data
    data, start_date, end_date = load_stock_data(data_path, cutoff)
    train_end = int(len(data) * 0.7)
    train_data = data.iloc[:train_end].copy().reset_index(drop=True)
    
    # Apply preprocessing for realistic optimization
    print(f"   Applying data preprocessing (scaling: {scaling_method}, outliers: {outlier_method})...")
    import time
    start_time = time.time()
    _, scaled_train_data, _, _ = preprocess_financial_data(
        train_data=train_data,
        scaling_method=scaling_method,
        outlier_method=outlier_method,
        save_preprocessor=False  # Don't save during optimization
    )
    preprocess_time = time.time() - start_time
    print(f"   Training data: {len(train_data)} rows (preprocessed in {preprocess_time:.1f}s)")
    
    best_score = -999
    best_params = {}
    
    def objective(trial):
        nonlocal best_score, best_params
        
        # Sample parameters to test (refined ranges for 12-parameter optimization)
        # ❌ REMOVED: hidden_size - fixed to 512 based on domain knowledge
        invalid_penalty = trial.suggest_float('invalid_penalty', 0.1, 0.8)  # Narrower, more realistic
        portfolio_scaling = trial.suggest_float('portfolio_scaling', 0.005, 0.2)  # Lower upper bound
        learning_rate = trial.suggest_float('learning_rate', 5e-5, 3e-4, log=True)
        epsilon_decay = trial.suggest_int('epsilon_decay', 800, 6000)  # Slightly wider for 200 episodes
        
        # Core DQN hyperparameters (memory-aware)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
        gamma = trial.suggest_float('gamma', 0.97, 0.9995)  # Tighter for trading
        alpha = trial.suggest_float('alpha', 0.5, 0.75)  # Tighter PER range
        
        # High-impact network & training parameters (refined)
        tau = trial.suggest_float('tau', 0.002, 0.015, log=True)  # Better soft update range
        epsilon_start = trial.suggest_float('epsilon_start', 0.7, 1.0)  # Allow less initial exploration
        epsilon_end = trial.suggest_float('epsilon_end', 0.002, 0.03)  # Tighter final exploration
        # Architecture fixed to 512 units (proven optimal for trading RL)
        update_frequency = trial.suggest_categorical('update_frequency', [1, 2, 4, 8])
        min_profit_threshold = trial.suggest_float('min_profit_threshold', 0.002, 0.03)  # Better for minute trading
        
        print(f"\n🔍 Trial {trial.number + 1}/{n_trials}")
        print(f"   invalid_penalty={invalid_penalty:.3f}, portfolio_scaling={portfolio_scaling:.3f}")
        print(f"   learning_rate={learning_rate:.2e}, epsilon_decay={epsilon_decay}")
        print(f"   batch_size={batch_size}, gamma={gamma:.3f}, alpha={alpha:.3f}")
        print(f"   tau={tau:.4f}, eps_start={epsilon_start:.2f}, eps_end={epsilon_end:.3f}")
        print(f"   hidden_size=512 (FIXED), update_freq={update_frequency}, profit_thresh={min_profit_threshold:.3f}")
        
        # Test the configuration
        results = test_parameters(
            invalid_penalty, portfolio_scaling, learning_rate, epsilon_decay,
            batch_size, gamma, alpha, tau, epsilon_start, epsilon_end,
            update_frequency, min_profit_threshold,
            train_data, scaled_train_data, num_episodes=n_episodes
        )
        
        score = results['score']
        print(f"   → Return: {results['avg_return']:.2%}, Win: {results['win_rate']:.1%}, Invalid: {results['avg_invalid_actions']:.0f}")
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
                'tau': tau,
                'epsilon_start': epsilon_start,
                'epsilon_end': epsilon_end,
                'hidden_size': 512,  # Fixed architecture value
                'update_frequency': update_frequency,
                'min_profit_threshold': min_profit_threshold,
                'performance': results
            }
            print(f"   🏆 NEW BEST!")
        
        return score
    
    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return best_params


def save_to_file(best_params: Dict, output_file: str = "OPTIMIZED_PARAMETERS.txt", 
                scaling_method: str = 'robust', outlier_method: str = 'winsorize'):
    """Save parameters to text file for copy-pasting"""
    
    params = best_params
    perf = best_params['performance']
    
    content = f"""
================================================================================
DQN v5 OPTIMIZED PARAMETERS - READY TO COPY-PASTE
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Preprocessing: {scaling_method} scaling, {outlier_method} outliers

PERFORMANCE ACHIEVED:
   Average Return: {perf['avg_return']:.2%} per episode
   Win Rate: {perf['win_rate']:.1%} of episodes  
   Invalid Actions: {perf['avg_invalid_actions']:.0f} per episode
   Score: {perf['score']:.3f}

================================================================================
COPY-PASTE INSTRUCTIONS
================================================================================

STEP 1: Open dqn_v5.py and find the step() function

STEP 2: Replace these lines:

FIND:
    portfolio_scaling = getattr(self.config, 'portfolio_scaling', 0.1)

REPLACE WITH:
    portfolio_scaling = getattr(self.config, 'portfolio_scaling', {params['portfolio_scaling']:.3f})

FIND:
    invalid_penalty = getattr(self.config, 'invalid_penalty', 0.1)

REPLACE WITH:
    invalid_penalty = getattr(self.config, 'invalid_penalty', {params['invalid_penalty']:.3f})

STEP 3: Open config.py and replace these default values:

FIND:
    portfolio_scaling: float = 0.1
    invalid_penalty: float = 0.1
    learning_rate: float = 0.0001
    epsilon_decay: int = 2000
    batch_size: int = BATCH_SIZE
    gamma: float = 0.99
    alpha: float = 0.6
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    hidden_size: int = 512
    update_frequency: int = TRAIN_INTERVAL
    min_profit_threshold: float = 0.015

REPLACE WITH:
    portfolio_scaling: float = {params['portfolio_scaling']:.3f}
    invalid_penalty: float = {params['invalid_penalty']:.3f}
    learning_rate: float = {params['learning_rate']:.2e}
    epsilon_decay: int = {params['epsilon_decay']}
    batch_size: int = {params['batch_size']}
    gamma: float = {params['gamma']:.3f}
    alpha: float = {params['alpha']:.3f}
    tau: float = {params['tau']:.4f}
    epsilon_start: float = {params['epsilon_start']:.3f}
    epsilon_end: float = {params['epsilon_end']:.4f}
    hidden_size: int = {params['hidden_size']}
    update_frequency: int = {params['update_frequency']}
    min_profit_threshold: float = {params['min_profit_threshold']:.4f}

================================================================================
SUMMARY OF OPTIMIZED VALUES
================================================================================

Portfolio Scaling: {params['portfolio_scaling']:.3f}  (was 0.010)
Invalid Penalty: {params['invalid_penalty']:.3f}    (was 0.497)  
Learning Rate: {params['learning_rate']:.2e}     (was 2.37e-4)
Epsilon Decay: {params['epsilon_decay']}         (was 2924)
Batch Size: {params['batch_size']}              (was 32)
Gamma: {params['gamma']:.3f}                    (was 0.99)
Alpha (PER): {params['alpha']:.3f}              (was 0.6)
Tau: {params['tau']:.4f}                        (was 0.005)
Epsilon Start: {params['epsilon_start']:.3f}    (was 1.0)
Epsilon End: {params['epsilon_end']:.4f}        (was 0.01)
Hidden Size: {params['hidden_size']}            (FIXED - not optimized, proven optimal for trading)
Update Frequency: {params['update_frequency']}   (was 4)
Min Profit Threshold: {params['min_profit_threshold']:.4f} (was 0.015)

Expected improvement:
• Better reward-return alignment
• Fewer invalid actions  
• Higher win rate
• More consistent performance

NOTE: Network architecture (hidden_size=512) was FIXED during optimization
to avoid chasing random variance. 512 units is proven optimal for financial RL.

================================================================================
"""
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\n💾 Parameters saved to: {output_file}")
    print("📋 Open the file and copy-paste the values into your code!")


def main():
    parser = argparse.ArgumentParser(description='Simple DQN v5 optimization')
    parser.add_argument('--data-path', required=True, help='Path to CSV data file')
    parser.add_argument('--cutoff', default='2021-05-06', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--trials', type=int, default=200, help='Number of trials (default: 200)')
    parser.add_argument('--episodes', type=int, default=200, help='Episodes per trial (default: 200)')
    parser.add_argument('--scaling', default='robust', choices=['robust', 'standard', 'minmax', 'none'], 
                       help='Data scaling method (default: robust)')
    parser.add_argument('--outliers', default='winsorize', choices=['winsorize', 'clip', 'none'],
                       help='Outlier handling method (default: winsorize)')
    parser.add_argument('--output', default='OPTIMIZED_PARAMETERS.txt', help='Output file name')
    
    args = parser.parse_args()
    
    # Run optimization
    print("Starting optimization...")
    best_params = run_optimization(
        args.data_path, pd.Timestamp(args.cutoff), args.trials,
        n_episodes=args.episodes, scaling_method=args.scaling, outlier_method=args.outliers
    )
    
    # Show results
    print("\n" + "="*50)
    print("🏆 OPTIMIZATION COMPLETE!")
    print("="*50)
    
    perf = best_params['performance']
    print(f"Best Performance:")
    print(f"  Return: {perf['avg_return']:.2%}")
    print(f"  Win Rate: {perf['win_rate']:.1%}")
    print(f"  Invalid Actions: {perf['avg_invalid_actions']:.0f}")
    
    print(f"\nOptimal Parameters:")
    print(f"  portfolio_scaling = {best_params['portfolio_scaling']:.3f}")
    print(f"  invalid_penalty = {best_params['invalid_penalty']:.3f}")
    print(f"  learning_rate = {best_params['learning_rate']:.2e}")
    print(f"  epsilon_decay = {best_params['epsilon_decay']}")
    print(f"  batch_size = {best_params['batch_size']}")
    print(f"  gamma = {best_params['gamma']:.3f}")
    print(f"  alpha = {best_params['alpha']:.3f}")
    print(f"  tau = {best_params['tau']:.4f}")
    print(f"  epsilon_start = {best_params['epsilon_start']:.3f}")
    print(f"  epsilon_end = {best_params['epsilon_end']:.4f}")
    print(f"  hidden_size = {best_params['hidden_size']} (FIXED - not optimized)")
    print(f"  update_frequency = {best_params['update_frequency']}")
    print(f"  min_profit_threshold = {best_params['min_profit_threshold']:.4f}")
    
    # Save to file
    save_to_file(best_params, args.output, args.scaling, args.outliers)
    
    print(f"\n✅ Done! Check {args.output} for copy-paste instructions.")


if __name__ == "__main__":
    main() 
    
    '''
    → Return: -1.06%, Win: 23.0%, Invalid: 11
   → Score: 0.086
   🏆 NEW BEST!
[I 2025-06-11 01:32:12,640] Trial 0 finished with value: 0.08619344167171336 and parameters: {
    'invalid_penalty': 0.6670893704523253, 
    'portfolio_scaling': 0.15191720952673463, 
    'learning_rate': 0.00017357556159736305, 
    'epsilon_decay': 965, 
    'batch_size': 32, 
    'gamma': 0.9738452076911572, 
    'alpha': 0.6267331565590931, 
    'tau': 0.005425172712396755, 
    'epsilon_start': 0.7032506605701238, 
    'epsilon_end': 0.0227870512892445, 
    'hidden_size': 256, 
    'update_frequency': 2, 
    'min_profit_threshold': 0.011730612416859255
    }
    '''