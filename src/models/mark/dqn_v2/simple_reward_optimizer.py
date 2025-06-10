"""
Simple DQN v5 Reward Parameter Optimizer
========================================

Finds optimal reward parameters using proper data preprocessing and saves them 
to a text file for easy copy-pasting.

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
                   train_data: pd.DataFrame,
                   scaled_train_data: pd.DataFrame,
                   num_episodes: int = 10) -> Dict[str, float]:
    """Test a parameter configuration"""
    
    # Create config with test parameters
    config = TradingConfig()
    config.invalid_penalty = invalid_penalty
    config.portfolio_scaling = portfolio_scaling
    config.learning_rate = learning_rate
    config.epsilon_decay = epsilon_decay
    
    try:
        # Create environment and agent
        env = TradingEnvironment(train_data, scaled_train_data, config, mode=TradingMode.TRAIN)
        agent = DoubleDuelingDQN(config)
        
        # Track results
        returns = []
        invalid_actions_list = []
        
        for episode in range(num_episodes):
            if episode % 5 == 0:  # Progress every 5 episodes
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


def run_optimization(data_path: str, cutoff: pd.Timestamp, n_trials: int = 25, 
                    scaling_method: str = 'robust', outlier_method: str = 'winsorize'):
    """Run the optimization"""
    
    print("🚀 Simple DQN v5 Parameter Optimization")
    print(f"   Running {n_trials} trials...")
    
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
        
        # Sample parameters to test
        invalid_penalty = trial.suggest_float('invalid_penalty', 0.05, 0.5)
        portfolio_scaling = trial.suggest_float('portfolio_scaling', 0.01, 0.2)
        learning_rate = trial.suggest_float('learning_rate', 5e-5, 3e-4, log=True)
        epsilon_decay = trial.suggest_int('epsilon_decay', 1000, 4000)
        
        print(f"\n🔍 Trial {trial.number + 1}/{n_trials}")
        print(f"   invalid_penalty={invalid_penalty:.3f}, portfolio_scaling={portfolio_scaling:.3f}")
        print(f"   learning_rate={learning_rate:.2e}, epsilon_decay={epsilon_decay}")
        
        # Test the configuration
        results = test_parameters(
            invalid_penalty, portfolio_scaling, learning_rate, epsilon_decay,
            train_data, scaled_train_data, num_episodes=10
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

🎯 PERFORMANCE ACHIEVED:
   Average Return: {perf['avg_return']:.2%} per episode
   Win Rate: {perf['win_rate']:.1%} of episodes  
   Invalid Actions: {perf['avg_invalid_actions']:.0f} per episode
   Score: {perf['score']:.3f}

================================================================================
📋 COPY-PASTE INSTRUCTIONS
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

REPLACE WITH:
    portfolio_scaling: float = {params['portfolio_scaling']:.3f}
    invalid_penalty: float = {params['invalid_penalty']:.3f}
    learning_rate: float = {params['learning_rate']:.2e}
    epsilon_decay: int = {params['epsilon_decay']}

================================================================================
📊 SUMMARY OF OPTIMIZED VALUES
================================================================================

✅ Portfolio Scaling: {params['portfolio_scaling']:.3f}  (was 0.1)
✅ Invalid Penalty: {params['invalid_penalty']:.3f}    (was 0.1)  
✅ Learning Rate: {params['learning_rate']:.2e}     (was 1e-4)
✅ Epsilon Decay: {params['epsilon_decay']}         (was 2000)

Expected improvement:
• Better reward-return alignment
• Fewer invalid actions  
• Higher win rate
• More consistent performance

================================================================================
"""
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\n💾 Parameters saved to: {output_file}")
    print("📋 Open the file and copy-paste the values into your code!")


def main():
    parser = argparse.ArgumentParser(description='Simple DQN v5 optimization')
    parser.add_argument('--data-path', required=True, help='Path to CSV data file')
    parser.add_argument('--cutoff', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--trials', type=int, default=25, help='Number of trials (default: 25)')
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
        scaling_method=args.scaling, outlier_method=args.outliers
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
    
    # Save to file
    save_to_file(best_params, args.output, args.scaling, args.outliers)
    
    print(f"\n✅ Done! Check {args.output} for copy-paste instructions.")


if __name__ == "__main__":
    main() 