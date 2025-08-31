"""
SAC Hyperparameter Optimizer
============================

Comprehensive hyperparameter optimization for SAC trading agent using Optuna.
Optimizes learning rates, SAC parameters, PER settings, and reward system.
Network architecture is kept fixed for focused algorithmic optimization.

Usage:
    python sac_optimizer.py --data-path "data/feature_engineering_v2/AAPL.csv" --cutoff "2023-01-01"
    
    # With custom environment type
    python sac_optimizer.py --data-path "data/feature_engineering_v2/AAPL.csv" --cutoff "2023-01-01" --environment basic
    python sac_optimizer.py --data-path "data/feature_engineering_v2/AAPL.csv" --cutoff "2023-01-01" --environment weighted_avg
    python sac_optimizer.py --data-path "data/feature_engineering_v2/AAPL.csv" --cutoff "2023-01-01" --environment lot_based
    
    # With custom preprocessing
    python sac_optimizer.py --data-path "data/feature_engineering_v2/AAPL.csv" --cutoff "2023-01-01" --scaling standard --outliers clip --environment weighted_avg
"""

import optuna
import numpy as np
import pandas as pd
import torch
import argparse
from datetime import datetime
from typing import Dict

# Import SAC components
from src.models.jeawan.sac.sac import SAC, load_stock_data
from src.models.jeawan.sac.sac_config import SACConfig, TradingMode, EnvironmentType, NetworkType
from src.models.jeawan.sac.sac_environments import create_environment
from src.models.mark.dqn_v2.data_preprocessor_v2 import preprocess_financial_data
from src.config.config import DEVICE, STOCK_FEATURES_V2


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
    
    # Environment configuration
    environment_type: EnvironmentType = EnvironmentType.WEIGHTED_AVERAGE,
    use_action_guidance: bool = True,
    
    # Network architecture
    network_type: NetworkType = NetworkType.ORIGINAL,
    
    num_episodes: int = 300
) -> Dict[str, float]:
    """Test SAC configuration with core hyperparameters (no architecture tuning)"""
    
    # Create config with test parameters
    config = SACConfig()
    
    # Environment configuration
    config.environment_type = environment_type
    config.use_action_guidance = use_action_guidance
    config.action_guidance_strength = 0.5
    config.soft_invalid_penalty = 0.01
    config.min_action_threshold = 0.05
    
    # Network architecture configuration
    config.network_type = network_type
    
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
        # Create environment and agent using new factory system
        env = create_environment(train_data, scaled_train_data, config, mode=TradingMode.TRAIN)
        agent = SAC(config)
        
        # DEBUG: Check data types
        print(f"    🔍 DEBUG: Train data shape: {train_data.shape}")
        print(f"    🔍 DEBUG: Scaled data shape: {scaled_train_data.shape}")
        print(f"    🔍 DEBUG: Train data dtypes: {train_data.dtypes.value_counts()}")
        print(f"    🔍 DEBUG: Scaled data dtypes: {scaled_train_data.dtypes.value_counts()}")
        
        # Check for any non-numeric columns that might cause issues
        for col in scaled_train_data.columns:
            if scaled_train_data[col].dtype == 'object' or 'timestamp' in col.lower():
                print(f"    ⚠️ WARNING: Non-numeric column detected: {col} (dtype: {scaled_train_data[col].dtype})")
                
        # Phase 1: Portfolio State Warmup (if needed) - IDENTICAL TO MAIN TRAINING
        if env.portfolio_normalizer is not None and not env.portfolio_normalizer.is_fitted:
            warmup_episodes = env.portfolio_normalizer.warmup_episodes
            print(f"    📊 PORTFOLIO NORMALIZATION WARMUP PHASE ({environment_type.value.upper()} ENV, {network_type.value.upper()} NET)")
            print(f"    Collecting portfolio states for {warmup_episodes} episodes...")
            print("    (No training will occur during this phase)")
            
            for warmup_ep in range(warmup_episodes):
                try:
                    state = env.reset()
                    episode_portfolio_states = []
                    
                    while True:
                        # Random action during warmup (pure exploration for SAC)
                        action = np.random.uniform(-1, 1)  # SAC continuous action space
                        next_state, reward, done, info = env.step(action)
                        
                        # Collect portfolio states
                        if hasattr(env, 'episode_portfolio_states'):
                            episode_portfolio_states.extend(env.episode_portfolio_states)
                        
                        state = next_state
                        if done:
                            break
                    
                    # Add collected states to normalizer
                    if episode_portfolio_states:
                        env.portfolio_normalizer.collect_warmup_data(episode_portfolio_states)
                    env.portfolio_normalizer.increment_episode()
                    
                    # Progress update
                    if (warmup_ep + 1) % 10 == 0 or warmup_ep == warmup_episodes - 1:
                        progress = (warmup_ep + 1) / warmup_episodes
                        print(f"      Warmup progress: {warmup_ep + 1}/{warmup_episodes} ({progress*100:.1f}%)")
                        
                except Exception as warmup_error:
                    print(f"    ❌ Error during warmup episode {warmup_ep}: {warmup_error}")
                    raise warmup_error
            
            print(f"    ✅ Portfolio state collection complete!")
            print(f"    🧠 Fitting portfolio normalizer...")
            print(f"    ✅ Ready to start SAC optimization with normalized portfolio features!")
        
        # Track results
        returns = []
        invalid_actions_list = []
        sharpe_ratios = []
        total_trades_list = []
        winning_trades_list = []
        
        for episode in range(num_episodes):
            if episode % 5 == 0:  # Progress every 5 episodes
                print(f"    Episode {episode}/{num_episodes}... ({environment_type.value}, {network_type.value})")
            
            try:
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
                
            except Exception as episode_error:
                print(f"    ❌ Error during episode {episode}: {episode_error}")
                import traceback
                print(f"    📋 Traceback: {traceback.format_exc()}")
                # Return early with error results
                return {
                    'avg_return': -0.2,
                    'std_return': 0.5,
                    'win_rate': 0.0,
                    'avg_invalid_actions': 100,
                    'avg_sharpe': -2.0,
                    'avg_trades': 0,
                    'avg_winning_trades': 0,
                    'environment_type': environment_type.value,
                    'network_type': network_type.value,
                    'score': -10,
                    'error': str(episode_error)
                }
        
        # Calculate performance metrics
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        avg_invalid = np.mean(invalid_actions_list)
        avg_sharpe = np.mean(sharpe_ratios)
        avg_trades = np.mean(total_trades_list)
        avg_winning_trades = np.mean(winning_trades_list)
        
        # Environment-specific scoring adjustments
        if environment_type == EnvironmentType.BASIC:
            # Basic environment - expect fewer trades but higher consistency
            optimal_trades_range = (2, 8)  # Conservative trading
            activity_bonus = 1.0 if optimal_trades_range[0] <= avg_trades <= optimal_trades_range[1] else 0.8
        elif environment_type == EnvironmentType.WEIGHTED_AVERAGE:
            # Weighted average - moderate trading activity
            optimal_trades_range = (3, 12)  # Balanced approach
            activity_bonus = 1.0 if optimal_trades_range[0] <= avg_trades <= optimal_trades_range[1] else 0.9
        else:  # LOT_BASED
            # Lot-based - can handle more frequent trading
            optimal_trades_range = (5, 20)  # More active trading
            activity_bonus = 1.0 if optimal_trades_range[0] <= avg_trades <= optimal_trades_range[1] else 0.85
        
        # Network-specific adjustments
        network_bonus = 1.0
        if network_type == NetworkType.SIMPLIFIED:
            # Simplified networks might have slightly different performance characteristics
            # but should be competitive with faster training
            network_bonus = 1.0  # No penalty - simplified networks can be just as good
        # ORIGINAL networks get no bonus/penalty (baseline)
        
        # Action masking effectiveness check (for monitoring only)
        if avg_invalid > 50:
            print(f"    ⚠️ WARNING: High invalid actions ({avg_invalid:.0f}) despite action masking!")
        elif avg_invalid > 20:
            print(f"    ℹ️ INFO: Moderate invalid actions ({avg_invalid:.0f}) - action masking working partially")
        # With good action masking, expect avg_invalid <= 20
        
        # Win rate bonus
        win_rate_bonus = min(win_rate * 2, 1.5)  # Cap at 1.5x bonus
        
        # Consistency bonus (lower std is better)
        consistency_bonus = 1.0 / (1.0 + std_return * 10)
        
        # Composite scoring function focused on trading performance
        # (No invalid action penalty needed with action masking)
        score = (
            avg_return * 5.0 +                    # Primary: average returns
            avg_sharpe * 1.0 +                    # Secondary: risk-adjusted returns
            win_rate_bonus * 1.5 +                # Bonus: consistency
            activity_bonus * 0.5 +                # Bonus: optimal trading frequency for environment
            consistency_bonus * 0.5 +             # Bonus: lower volatility
            network_bonus * 0.2 +                 # Bonus: network architecture efficiency
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
            'environment_type': environment_type.value,
            'network_type': network_type.value,
            'score': score
        }
        
    except Exception as e:
        print(f"Error testing SAC config on {environment_type.value} with {network_type.value}: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return {
            'avg_return': -0.2, 
            'std_return': 0.5, 
            'win_rate': 0.0, 
            'avg_invalid_actions': 100, 
            'avg_sharpe': -2.0,
            'avg_trades': 0,
            'avg_winning_trades': 0,
            'environment_type': environment_type.value,
            'network_type': network_type.value,
            'score': -10,
            'error': str(e)
        }


def run_sac_optimization(data_path: str, cutoff: pd.Timestamp, n_trials: int = 40,
                        scaling_method: str = 'robust', outlier_method: str = 'winsorize',
                        environment_type: EnvironmentType = EnvironmentType.WEIGHTED_AVERAGE,
                        use_action_guidance: bool = True,
                        network_type: NetworkType = NetworkType.ORIGINAL):
    """Run comprehensive SAC hyperparameter optimization for chosen environment and network"""
    
    env_name = environment_type.value.replace('_', ' ').title()
    net_name = network_type.value.title()
    print("🚀 SAC Focused Hyperparameter Optimization")
    print(f"   🏗️ Environment: {env_name}")
    print(f"   🧠 Network: {net_name} Architecture")
    print(f"   🎯 Action Masking: {'✅ ENABLED' if use_action_guidance else '❌ DISABLED'}")
    print(f"   Running {n_trials} trials (algorithmic parameters only)...")
    print(f"   Architecture fixed: Actor/Critic 512, dropout 0.1, 8 heads, 2 blocks")
    
    # Load and prepare data
    data, start_date, end_date = load_stock_data(data_path, cutoff, STOCK_FEATURES_V2)
    train_end = int(len(data) * 0.7)
    train_data = data.iloc[:train_end].copy().reset_index(drop=True)
    
    # DEBUG: Verify timestamp column is removed
    print(f"   📊 Loaded data shape: {data.shape}")
    print(f"   📋 Columns: {len(data.columns)} features")
    if 'timestamp' in data.columns:
        print(f"   ⚠️ WARNING: timestamp column still present!")
    else:
        print(f"   ✅ timestamp column properly filtered out")
    print(f"   📈 Train data: {len(train_data)} rows")
    
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
    
    # DEBUG: Verify preprocessed data doesn't have timestamp
    print(f"   📊 Scaled data shape: {scaled_train_data.shape}")
    if 'timestamp' in scaled_train_data.columns:
        print(f"   ⚠️ WARNING: timestamp column in scaled data!")
    else:
        print(f"   ✅ scaled data timestamp-free")
    
    best_score = -999
    best_params = {}
    
    def objective(trial):
        nonlocal best_score, best_params
        
        # Learning rates
        actor_lr = trial.suggest_float('actor_learning_rate', 1e-5, 1e-3, log=True)
        critic_lr = trial.suggest_float('critic_learning_rate', 1e-5, 1e-3, log=True)
        alpha_lr = trial.suggest_float('alpha_learning_rate', 1e-5, 1e-3, log=True)
        
        # SAC parameters
        initial_alpha = trial.suggest_float('initial_alpha', 0.2, 1.0)
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
        portfolio_scaling = trial.suggest_float('portfolio_scaling', 2.0, 10.0)
        invalid_penalty = trial.suggest_float('invalid_penalty', 0.0001, 0.005)
        min_trade_amount = trial.suggest_float('min_trade_amount', 0.0001, 0.005)
        
        print(f"\n🔍 Trial {trial.number + 1}/{n_trials} - {env_name} Environment, {net_name} Network")
        print(f"   Learning rates: Actor {actor_lr:.2e}, Critic {critic_lr:.2e}, Alpha {alpha_lr:.2e}")
        print(f"   Training: batch={batch_size}, update_freq={update_frequency}")
        print(f"   SAC params: α={initial_alpha:.3f}, τ={tau:.3f}, γ={gamma:.3f}")
        print(f"   PER params: α={per_alpha:.3f}, β_start={per_beta_start:.3f}, β_end={per_beta_end:.3f}, steps={per_beta_annealing_steps}")
        print(f"   Trading: position_size={max_position_size:.2f}, portfolio={portfolio_scaling:.3f}, invalid={invalid_penalty:.3f}")
        print(f"   Min trade amount: {min_trade_amount:.3f}")
        print(f"   Environment: {environment_type.value} with action masking: {use_action_guidance}")
        print(f"   Network: {network_type.value} architecture")
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
            environment_type=environment_type,
            use_action_guidance=use_action_guidance,
            network_type=network_type,
            num_episodes=300
        )
        
        score = results['score']
        print(f"   → Return: {results['avg_return']:.2%} ± {results['std_return']:.2%}")
        print(f"   → Win rate: {results['win_rate']:.1%}, Sharpe: {results['avg_sharpe']:.2f}")
        print(f"   → Trades: {results['avg_trades']:.1f}, Invalid: {results['avg_invalid_actions']:.0f}")
        print(f"   → Environment: {results['environment_type']}, Network: {results['network_type']}")
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
                'environment_type': environment_type,
                'use_action_guidance': use_action_guidance,
                'network_type': network_type,
                'performance': results
            }
            print(f"   🏆 NEW BEST for {env_name} + {net_name}!")
        
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
    env_type = best_params.get('environment_type', EnvironmentType.WEIGHTED_AVERAGE)
    env_name = env_type.value.replace('_', ' ').title() if isinstance(env_type, EnvironmentType) else str(env_type).replace('_', ' ').title()
    net_type = best_params.get('network_type', NetworkType.ORIGINAL)
    net_name = net_type.value.title() if isinstance(net_type, NetworkType) else str(net_type).title()
    action_guidance = best_params.get('use_action_guidance', True)
    
    content = f"""
================================================================================
SAC OPTIMIZED PARAMETERS - READY TO COPY-PASTE
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Environment: {env_name}
Network: {net_name} Architecture
Action Masking: {'✅ ENABLED' if action_guidance else '❌ DISABLED'}
Preprocessing: {scaling_method} scaling, {outlier_method} outliers

🎯 PERFORMANCE ACHIEVED:
   Average Return: {perf['avg_return']:.2%} ± {perf['std_return']:.2%} per episode
   Win Rate: {perf['win_rate']:.1%} of episodes
   Average Sharpe: {perf['avg_sharpe']:.2f}
   Average Trades: {perf['avg_trades']:.1f} per episode
   Invalid Actions: {perf['avg_invalid_actions']:.0f} per episode
   Environment: {perf.get('environment_type', 'unknown')}
   Network: {perf.get('network_type', 'unknown')}
   Score: {perf['score']:.3f}

================================================================================
📋 COPY-PASTE INSTRUCTIONS FOR SAC
================================================================================

STEP 1: Open sac.py and find the SACConfig class

STEP 2: Replace the following values in SACConfig:

FIND:
    environment_type: EnvironmentType = EnvironmentType.BASIC
    network_type: NetworkType = NetworkType.ORIGINAL
    use_action_guidance: bool = True
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
    environment_type: EnvironmentType = EnvironmentType.{env_type.name if isinstance(env_type, EnvironmentType) else env_type}
    network_type: NetworkType = NetworkType.{net_type.name if isinstance(net_type, NetworkType) else net_type}
    use_action_guidance: bool = {action_guidance}
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
📊 SUMMARY OF OPTIMIZED SAC VALUES FOR {env_name.upper()} + {net_name.upper()}
================================================================================

🏗️ CONFIGURATION:
✅ Environment Type: {env_name}
✅ Network Architecture: {net_name}
✅ Action Masking: {'✅ ENABLED' if action_guidance else '❌ DISABLED'}

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

🏗️ NETWORK ARCHITECTURE ({net_name.upper()}):
   {'Actor: Multi-scale CNN + Transformer + Complex Portfolio (900k+ params)' if net_name == 'Original' else 'Actor: Single CNN + Transformer + LSTM Portfolio (170k params)'}
   {'Critic: Multi-scale CNN + Transformer + Complex Portfolio (900k+ params)' if net_name == 'Original' else 'Critic: Single CNN + Transformer + LSTM Portfolio (170k params)'}
   {'Best for: Maximum model capacity and performance' if net_name == 'Original' else 'Best for: Faster training and reduced overfitting'}

Expected improvements for {env_name} + {net_name}:
• Better exploration-exploitation balance with optimized alpha
• More stable learning with tuned critic/actor learning rates  
• Improved risk-return profile with optimized reward scaling
• More efficient trading with optimal minimum trade thresholds
• Enhanced convergence with tuned soft update rates
• {'Minimal invalid actions with action masking enabled' if action_guidance else 'Traditional penalty-based invalid action handling'}
• {'Maximum model capacity for complex patterns' if net_name == 'Original' else 'Faster training with competitive performance'}

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
- {'Dramatically fewer invalid actions (action masking)' if action_guidance else 'Standard invalid action handling'}
- More strategic trading patterns
- {'Maximum learning capacity' if net_name == 'Original' else 'Efficient learning with faster convergence'}

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
    parser.add_argument('--environment', default='weighted_avg', 
                       choices=['basic', 'weighted_avg', 'lot_based'],
                       help='Trading environment type (default: weighted_avg)')
    parser.add_argument('--network', default='original',
                       choices=['original', 'simplified'],
                       help='Neural network architecture (default: original)')
    parser.add_argument('--no-action-masking', action='store_true',
                       help='Disable continuous action masking (default: enabled)')
    parser.add_argument('--output', default='SAC_OPTIMIZED_PARAMETERS.txt', help='Output file name')
    
    args = parser.parse_args()
    
    # Convert environment string to EnvironmentType
    env_mapping = {
        'basic': EnvironmentType.BASIC,
        'weighted_avg': EnvironmentType.WEIGHTED_AVERAGE,
        'lot_based': EnvironmentType.LOT_BASED
    }
    environment_type = env_mapping[args.environment]
    
    # Convert network string to NetworkType  
    net_mapping = {
        'original': NetworkType.ORIGINAL,
        'simplified': NetworkType.SIMPLIFIED
    }
    network_type = net_mapping[args.network]
    
    use_action_guidance = not args.no_action_masking
    
    # Run optimization
    print("Starting SAC hyperparameter optimization...")
    best_params = run_sac_optimization(
        args.data_path, pd.Timestamp(args.cutoff), args.trials,
        scaling_method=args.scaling, outlier_method=args.outliers,
        environment_type=environment_type, use_action_guidance=use_action_guidance,
        network_type=network_type
    )
    
    # Show results
    print("\n" + "="*60)
    print("🏆 SAC OPTIMIZATION COMPLETE!")
    print("="*60)
    
    perf = best_params['performance']
    env_name = environment_type.value.replace('_', ' ').title()
    net_name = network_type.value.title()
    print(f"Environment: {env_name}")
    print(f"Network: {net_name} Architecture")
    print(f"Action Masking: {'✅ ENABLED' if use_action_guidance else '❌ DISABLED'}")
    print(f"Best Performance:")
    print(f"  Return: {perf['avg_return']:.2%} ± {perf['std_return']:.2%}")
    print(f"  Win Rate: {perf['win_rate']:.1%}")
    print(f"  Sharpe Ratio: {perf['avg_sharpe']:.2f}")
    print(f"  Average Trades: {perf['avg_trades']:.1f}")
    print(f"  Invalid Actions: {perf['avg_invalid_actions']:.0f}")
    
    print(f"\nOptimal SAC Parameters for {env_name} + {net_name}:")
    print(f"  Learning rates: Actor {best_params['actor_learning_rate']:.2e}, Critic {best_params['critic_learning_rate']:.2e}")
    print(f"  SAC params: α={best_params['initial_alpha']:.3f}, τ={best_params['tau']:.3f}, γ={best_params['gamma']:.3f}")
    print(f"  Training: batch={best_params['batch_size']}, update_freq={best_params['update_frequency']}")
    print(f"  Reward scaling: portfolio={best_params['portfolio_scaling']:.3f}, invalid={best_params['invalid_penalty']:.3f}")
    print(f"  Architecture: {net_name} ({'900k+' if network_type == NetworkType.ORIGINAL else '170k'} parameters)")
    
    # Save to file
    save_sac_params_to_file(best_params, args.output, args.scaling, args.outliers)
    
    print(f"\n✅ Done! Check {args.output} for copy-paste instructions.")


if __name__ == "__main__":
    main() 