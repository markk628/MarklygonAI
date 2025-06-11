"""
SAC Hyperparameter Optimizer
============================

Comprehensive hyperparameter optimization for SAC trading agent using Optuna.
Optimizes learning rates, SAC parameters, PER settings, and reward system.
Network architecture is kept fixed for focused algorithmic optimization.

Usage:
    python sac_optimizer.py --data-path "data/feature_engineering_v2/AAPL.csv" --cutoff "2023-01-01"
    
    # With custom preprocessing
    python sac_optimizer.py --data-path "data/feature_engineering_v2/AAPL.csv" --cutoff "2023-01-01" --scaling standard --outliers clip
"""

import optuna
import numpy as np
import pandas as pd
import torch
import argparse
from datetime import datetime
from typing import Dict

# Import SAC components
from src.models.jeawan.sac.sac import SAC, SACTradingEnvironment, TradingMode, SACConfig, load_stock_data
from src.models.mark.dqn_v2.data_preprocessor_v2 import preprocess_financial_data
from src.config.config import DEVICE


def test_sac_parameters(
    # Learning rates
    actor_learning_rate: float,
    critic_learning_rate: float,
    alpha_learning_rate: float,
    
    # SAC parameters
    initial_alpha: float,
    target_entropy_multiplier: float,
    tau: float,
    gamma: float,
    
    # PER parameters
    per_alpha: float,
    per_beta_start: float,
    per_beta_end: float,
    per_beta_annealing_steps: int,
    
    # Training parameters
    batch_size: int,
    update_frequency: int,
    max_position_size: float,
    
    # Reward parameters
    portfolio_scaling: float,
    invalid_penalty: float,
    min_trade_amount: float,
    
    # Training data
    train_data: pd.DataFrame,
    scaled_train_data: pd.DataFrame,
    num_episodes: int = 100
) -> Dict[str, float]:
    """Test SAC configuration with core hyperparameters (no architecture tuning)"""
    
    # Create config with test parameters
    config = SACConfig()
    
    # Fixed architecture parameters (original values)
    config.actor_hidden_size = 512
    config.critic_hidden_size = 512
    config.dropout_rate = 0.1
    config.transformer_heads = 8
    config.num_transformer_blocks = 2
    config.gradient_clip_norm = 1.0
    
    # Optimizable parameters
    config.actor_learning_rate = actor_learning_rate
    config.critic_learning_rate = critic_learning_rate
    config.alpha_learning_rate = alpha_learning_rate
    config.initial_alpha = initial_alpha
    config.target_entropy = -1.0 * target_entropy_multiplier
    config.tau = tau
    config.gamma = gamma
    config.per_alpha = per_alpha
    config.per_beta_start = per_beta_start
    config.per_beta_end = per_beta_end
    config.per_beta_annealing_steps = per_beta_annealing_steps
    config.batch_size = batch_size
    config.update_frequency = update_frequency
    config.max_position_size = max_position_size
    config.portfolio_scaling = portfolio_scaling
    config.invalid_penalty = invalid_penalty
    config.min_trade_amount = min_trade_amount
    
    try:
        # Create environment and agent
        env = SACTradingEnvironment(train_data, scaled_train_data, config, mode=TradingMode.TRAIN)
        agent = SAC(config)
        
        # Track results
        returns = []
        invalid_actions_list = []
        sharpe_ratios = []
        total_trades_list = []
        winning_trades_list = []
        
        for episode in range(num_episodes):
            if episode % 5 == 0:  # Progress every 5 episodes
                print(f"    Episode {episode}/{num_episodes}...")
            
            metrics = agent.train_episode(env)
            returns.append(metrics['total_return'])
            invalid_actions_list.append(metrics['invalid_actions'])
            total_trades_list.append(metrics['total_trades'])
            winning_trades_list.append(metrics['winning_trades'])
            
            # Calculate Sharpe ratio approximation
            if metrics['total_return'] != 0:
                # Simple volatility approximation
                volatility = abs(metrics['total_return']) * 0.1  # Rough estimate
                sharpe = metrics['total_return'] / (volatility + 1e-6)
            else:
                sharpe = 0.0
            sharpe_ratios.append(sharpe)
        
        # Calculate performance metrics
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        avg_invalid = np.mean(invalid_actions_list)
        avg_sharpe = np.mean(sharpe_ratios)
        avg_trades = np.mean(total_trades_list)
        avg_winning_trades = np.mean(winning_trades_list)
        
        # Trading activity score (prefer moderate trading)
        activity_score = 1.0
        if avg_trades < 0.5:  # Too passive
            activity_score = 0.7
        elif avg_trades > 5:  # Too active
            activity_score = 0.8
        
        # Win rate bonus
        win_rate_bonus = min(win_rate * 2, 1.5)  # Cap at 1.5x bonus
        
        # Consistency bonus (lower std is better)
        consistency_bonus = 1.0 / (1.0 + std_return * 10)
        
        # Composite scoring function
        score = (
            avg_return * 5.0 +                    # Primary: average returns
            avg_sharpe * 1.0 +                    # Secondary: risk-adjusted returns
            win_rate_bonus * 1.5 +                # Bonus: consistency
            activity_score * 0.5 +                # Bonus: reasonable trading frequency
            consistency_bonus * 0.5 +             # Bonus: lower volatility
            -(avg_invalid / 50) * 1.0 +           # Penalty: invalid actions
            (avg_winning_trades / max(avg_trades, 1)) * 0.5  # Bonus: win rate on trades
        )
        
        return {
            'avg_return': avg_return,
            'std_return': std_return,
            'win_rate': win_rate,
            'avg_invalid_actions': avg_invalid,
            'avg_sharpe': avg_sharpe,
            'avg_trades': avg_trades,
            'avg_winning_trades': avg_winning_trades,
            'score': score
        }
        
    except Exception as e:
        print(f"Error testing SAC config: {e}")
        return {
            'avg_return': -0.2, 
            'std_return': 0.5, 
            'win_rate': 0.0, 
            'avg_invalid_actions': 100, 
            'avg_sharpe': -2.0,
            'avg_trades': 0,
            'avg_winning_trades': 0,
            'score': -10
        }


def run_sac_optimization(data_path: str, cutoff: pd.Timestamp, n_trials: int = 40,
                        scaling_method: str = 'robust', outlier_method: str = 'winsorize'):
    """Run comprehensive SAC hyperparameter optimization"""
    
    print("🚀 SAC Focused Hyperparameter Optimization")
    print(f"   Running {n_trials} trials (algorithmic parameters only)...")
    print(f"   Architecture fixed: Actor/Critic 512, dropout 0.1, 8 heads, 2 blocks")
    
    # Load and prepare data
    data, start_date, end_date = load_stock_data(data_path, cutoff)
    train_end = int(len(data) * 0.7)
    train_data = data.iloc[:train_end].copy().reset_index(drop=True)
    
    # Apply preprocessing
    print(f"   Applying data preprocessing (scaling: {scaling_method}, outliers: {outlier_method})...")
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
        
        # Learning rates
        actor_lr = trial.suggest_float('actor_learning_rate', 1e-5, 1e-3, log=True)
        critic_lr = trial.suggest_float('critic_learning_rate', 1e-5, 1e-3, log=True)
        alpha_lr = trial.suggest_float('alpha_learning_rate', 1e-5, 1e-3, log=True)
        
        # SAC parameters
        initial_alpha = trial.suggest_float('initial_alpha', 0.01, 0.5)
        target_entropy_multiplier = trial.suggest_float('target_entropy_multiplier', 0.5, 2.0)
        tau = trial.suggest_float('tau', 0.001, 0.02)
        gamma = trial.suggest_float('gamma', 0.95, 0.999)
        
        # PER parameters
        per_alpha = trial.suggest_float('per_alpha', 0.4, 0.8)
        per_beta_start = trial.suggest_float('per_beta_start', 0.3, 0.5)
        per_beta_end = trial.suggest_float('per_beta_end', 0.9, 1.0)
        per_beta_annealing_steps = trial.suggest_int('per_beta_annealing_steps', 100, 500)
        
        # Training parameters
        batch_size = trial.suggest_int('batch_size', 32, 128)
        update_frequency = trial.suggest_int('update_frequency', 1, 10)
        max_position_size = trial.suggest_float('max_position_size', 0.5, 2.0)
        
        # Reward system parameters
        portfolio_scaling = trial.suggest_float('portfolio_scaling', 0.01, 0.3)
        invalid_penalty = trial.suggest_float('invalid_penalty', 0.05, 0.5)
        min_trade_amount = trial.suggest_float('min_trade_amount', 0.005, 0.05)
        
        print(f"\n🔍 Trial {trial.number + 1}/{n_trials}")
        print(f"   Learning rates: Actor {actor_lr:.2e}, Critic {critic_lr:.2e}, Alpha {alpha_lr:.2e}")
        print(f"   Training: batch={batch_size}, update_freq={update_frequency}")
        print(f"   SAC params: α={initial_alpha:.3f}, τ={tau:.3f}, γ={gamma:.3f}")
        print(f"   PER params: α={per_alpha:.3f}, β_start={per_beta_start:.3f}, β_end={per_beta_end:.3f}, steps={per_beta_annealing_steps}")
        print(f"   Trading: position_size={max_position_size:.2f}, portfolio={portfolio_scaling:.3f}, invalid={invalid_penalty:.3f}")
        print(f"   Min trade amount: {min_trade_amount:.3f}")
        print(f"   [Architecture: Fixed - Actor/Critic 512, dropout 0.1, 8 heads, 2 blocks]")
        
        # Test the configuration
        results = test_sac_parameters(
            actor_learning_rate=actor_lr,
            critic_learning_rate=critic_lr,
            alpha_learning_rate=alpha_lr,
            initial_alpha=initial_alpha,
            target_entropy_multiplier=target_entropy_multiplier,
            tau=tau,
            gamma=gamma,
            per_alpha=per_alpha,
            per_beta_start=per_beta_start,
            per_beta_end=per_beta_end,
            per_beta_annealing_steps=per_beta_annealing_steps,
            batch_size=batch_size,
            update_frequency=update_frequency,
            max_position_size=max_position_size,
            portfolio_scaling=portfolio_scaling,
            invalid_penalty=invalid_penalty,
            min_trade_amount=min_trade_amount,
            train_data=train_data,
            scaled_train_data=scaled_train_data,
            num_episodes=100
        )
        
        score = results['score']
        print(f"   → Return: {results['avg_return']:.2%} ± {results['std_return']:.2%}")
        print(f"   → Win rate: {results['win_rate']:.1%}, Sharpe: {results['avg_sharpe']:.2f}")
        print(f"   → Trades: {results['avg_trades']:.1f}, Invalid: {results['avg_invalid_actions']:.0f}")
        print(f"   → Score: {score:.3f}")
        
        # Track best parameters
        if score > best_score:
            best_score = score
            best_params = {
                'actor_learning_rate': actor_lr,
                'critic_learning_rate': critic_lr,
                'alpha_learning_rate': alpha_lr,
                'initial_alpha': initial_alpha,
                'target_entropy_multiplier': target_entropy_multiplier,
                'tau': tau,
                'gamma': gamma,
                'per_alpha': per_alpha,
                'per_beta_start': per_beta_start,
                'per_beta_end': per_beta_end,
                'per_beta_annealing_steps': per_beta_annealing_steps,
                'batch_size': batch_size,
                'update_frequency': update_frequency,
                'max_position_size': max_position_size,
                'portfolio_scaling': portfolio_scaling,
                'invalid_penalty': invalid_penalty,
                'min_trade_amount': min_trade_amount,
                'performance': results
            }
            print(f"   🏆 NEW BEST!")
        
        return score
    
    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return best_params


def save_sac_params_to_file(best_params: Dict, output_file: str = "SAC_OPTIMIZED_PARAMETERS.txt",
                           scaling_method: str = 'robust', outlier_method: str = 'winsorize'):
    """Save SAC parameters to text file for copy-pasting"""
    
    params = best_params
    perf = best_params['performance']
    
    content = f"""
================================================================================
SAC OPTIMIZED PARAMETERS - READY TO COPY-PASTE
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Preprocessing: {scaling_method} scaling, {outlier_method} outliers

🎯 PERFORMANCE ACHIEVED:
   Average Return: {perf['avg_return']:.2%} ± {perf['std_return']:.2%} per episode
   Win Rate: {perf['win_rate']:.1%} of episodes
   Average Sharpe: {perf['avg_sharpe']:.2f}
   Average Trades: {perf['avg_trades']:.1f} per episode
   Invalid Actions: {perf['avg_invalid_actions']:.0f} per episode
   Score: {perf['score']:.3f}

================================================================================
📋 COPY-PASTE INSTRUCTIONS FOR SAC
================================================================================

STEP 1: Open sac.py and find the SACConfig class

STEP 2: Replace the following values in SACConfig:

FIND:
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    batch_size: int = BATCH_SIZE
    update_frequency: int = 1
    max_position_size: float = MAX_POSITION_SIZE
    initial_alpha: float = 0.1
    target_entropy_multiplier: float = 1.0
    tau: float = 0.005
    gamma: float = 0.99
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_annealing_steps: int = 100
    portfolio_scaling: float = 0.1
    invalid_penalty: float = 0.1
    min_trade_amount: float = 0.01

REPLACE WITH:
    actor_learning_rate: float = {params['actor_learning_rate']:.2e}
    critic_learning_rate: float = {params['critic_learning_rate']:.2e}
    alpha_learning_rate: float = {params['alpha_learning_rate']:.2e}
    batch_size: int = {params['batch_size']}
    update_frequency: int = {params['update_frequency']}
    max_position_size: float = {params['max_position_size']:.2f}
    initial_alpha: float = {params['initial_alpha']:.3f}
    target_entropy_multiplier: float = {params['target_entropy_multiplier']:.3f}
    tau: float = {params['tau']:.3f}
    gamma: float = {params['gamma']:.3f}
    per_alpha: float = {params['per_alpha']:.3f}
    per_beta_start: float = {params['per_beta_start']:.3f}
    per_beta_end: float = {params['per_beta_end']:.3f}
    per_beta_annealing_steps: int = {params['per_beta_annealing_steps']}
    portfolio_scaling: float = {params['portfolio_scaling']:.3f}
    invalid_penalty: float = {params['invalid_penalty']:.3f}
    min_trade_amount: float = {params['min_trade_amount']:.3f}

================================================================================
📊 SUMMARY OF OPTIMIZED SAC VALUES
================================================================================

🎓 LEARNING RATES:
✅ Actor LR: {params['actor_learning_rate']:.2e}      (was 3e-4)
✅ Critic LR: {params['critic_learning_rate']:.2e}     (was 3e-4) 
✅ Alpha LR: {params['alpha_learning_rate']:.2e}       (was 3e-4)

⚙️ TRAINING PARAMETERS:
✅ Batch Size: {params['batch_size']}                  (was BATCH_SIZE)
✅ Update Frequency: {params['update_frequency']}          (was 1)
✅ Max Position Size: {params['max_position_size']:.2f}      (was MAX_POSITION_SIZE)

🤖 SAC PARAMETERS:
✅ Initial Alpha: {params['initial_alpha']:.3f}           (was 0.1)
✅ Target Entropy Multiplier: {params['target_entropy_multiplier']:.3f}  (was 1.0)
✅ Tau (soft update): {params['tau']:.3f}       (was 0.005)
✅ Gamma (discount): {params['gamma']:.3f}        (was 0.99)

📊 PER PARAMETERS:
✅ PER Alpha: {params['per_alpha']:.3f}              (was 0.6)
✅ PER Beta Start: {params['per_beta_start']:.3f}         (was 0.4)
✅ PER Beta End: {params['per_beta_end']:.3f}           (was 1.0)
✅ Beta Annealing Steps: {params['per_beta_annealing_steps']}     (was 100)

💰 REWARD SYSTEM:
✅ Portfolio Scaling: {params['portfolio_scaling']:.3f}      (was 0.1)
✅ Invalid Penalty: {params['invalid_penalty']:.3f}        (was 0.1)

📈 TRADING BEHAVIOR:
✅ Min Trade Amount: {params['min_trade_amount']:.3f}        (was 0.01)

🏗️ NETWORK ARCHITECTURE (FIXED):
   Actor Hidden Size: 512
   Critic Hidden Size: 512
   Dropout Rate: 0.1
   Transformer Heads: 8
   Transformer Blocks: 2
   Gradient Clip Norm: 1.0

Expected improvements:
• Better exploration-exploitation balance with optimized alpha
• More stable learning with tuned critic/actor learning rates  
• Improved risk-return profile with optimized reward scaling
• More efficient trading with optimal minimum trade thresholds
• Enhanced convergence with tuned soft update rates

================================================================================
🚀 NEXT STEPS
================================================================================

1. Apply the parameter changes above to your SAC implementation
2. Run training with: train_sac(data_path, cutoff, num_episodes=500)
3. Monitor training for improved stability and performance
4. Consider running more optimization trials for further refinement

Performance should improve in:
- Higher average returns per episode
- More consistent performance (lower volatility)
- Better risk-adjusted returns (Sharpe ratio)
- Fewer invalid actions and more strategic trading

================================================================================
"""
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"\n💾 SAC parameters saved to: {output_file}")
    print("📋 Open the file and copy-paste the values into your SAC code!")


def main():
    parser = argparse.ArgumentParser(description='SAC comprehensive hyperparameter optimization')
    parser.add_argument('--data-path', required=True, help='Path to CSV data file')
    parser.add_argument('--cutoff', default='2021-05-06', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--trials', type=int, default=200, help='Number of trials (default: 200)')
    parser.add_argument('--scaling', default='robust', choices=['robust', 'standard', 'minmax', 'none'],
                       help='Data scaling method (default: robust)')
    parser.add_argument('--outliers', default='winsorize', choices=['winsorize', 'clip', 'none'],
                       help='Outlier handling method (default: winsorize)')
    parser.add_argument('--output', default='SAC_OPTIMIZED_PARAMETERS.txt', help='Output file name')
    
    args = parser.parse_args()
    
    # Run optimization
    print("Starting SAC hyperparameter optimization...")
    best_params = run_sac_optimization(
        args.data_path, pd.Timestamp(args.cutoff), args.trials,
        scaling_method=args.scaling, outlier_method=args.outliers
    )
    
    # Show results
    print("\n" + "="*60)
    print("🏆 SAC OPTIMIZATION COMPLETE!")
    print("="*60)
    
    perf = best_params['performance']
    print(f"Best Performance:")
    print(f"  Return: {perf['avg_return']:.2%} ± {perf['std_return']:.2%}")
    print(f"  Win Rate: {perf['win_rate']:.1%}")
    print(f"  Sharpe Ratio: {perf['avg_sharpe']:.2f}")
    print(f"  Average Trades: {perf['avg_trades']:.1f}")
    print(f"  Invalid Actions: {perf['avg_invalid_actions']:.0f}")
    
    print(f"\nOptimal SAC Parameters:")
    print(f"  Learning rates: Actor {best_params['actor_learning_rate']:.2e}, Critic {best_params['critic_learning_rate']:.2e}")
    print(f"  SAC params: α={best_params['initial_alpha']:.3f}, τ={best_params['tau']:.3f}, γ={best_params['gamma']:.3f}")
    print(f"  Training: batch={best_params['batch_size']}, update_freq={best_params['update_frequency']}")
    print(f"  Reward scaling: portfolio={best_params['portfolio_scaling']:.3f}, invalid={best_params['invalid_penalty']:.3f}")
    print(f"  Architecture: Fixed (Actor/Critic 512, dropout 0.1, 8 heads, 2 blocks)")
    
    # Save to file
    save_sac_params_to_file(best_params, args.output, args.scaling, args.outliers)
    
    print(f"\n✅ Done! Check {args.output} for copy-paste instructions.")


if __name__ == "__main__":
    main() 