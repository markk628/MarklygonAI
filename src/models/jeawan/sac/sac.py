"""
SAC Training Function - Clean Modular Implementation
==================================================

Modern SAC training function that uses the new modular architecture:
- sac_agent.py: SAC agent implementation
- sac_networks.py: Network architectures (Original vs Simplified)
- sac_environments.py: Action masking environments  
- sac_config.py: Configuration system

Supports architecture selection and clean separation of concerns.
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, time
from pathlib import Path

from src.config.config import (
    NUM_EPISODES,
    TRAIN_RATIO,
    VALID_RATIO,
    EVALUATE_INTERVAL,
    DEVICE
)

from src.models.jeawan.sac.sac_agent import SAC
from src.models.jeawan.sac.sac_environments import create_environment
from src.models.jeawan.sac.sac_config import (
    SACConfig, 
    EnvironmentType, 
    NetworkType, 
    TradingMode,
    create_basic_sac_config,
    create_weighted_average_sac_config,
    create_lot_based_sac_config
)


def filter_to_regular_hours(df):
    """Filter dataframe to regular market hours using UTC timestamps"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Regular market hours filtering
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    eastern_times = df['timestamp'].dt.tz_convert('US/Eastern')
    market_open = eastern_times.dt.time >= time(9, 30)
    market_close = eastern_times.dt.time < time(16, 0)
    
    filtered_df = df[market_open & market_close].reset_index(drop=True)
    filtered_df['timestamp'] = filtered_df['timestamp'].dt.tz_localize(None)
    
    print(f"Data filtered: {len(df)} → {len(filtered_df)} rows ({len(filtered_df)/len(df)*100:.1f}%)")
    return filtered_df


def load_stock_data(data_path: str, cutoff: pd.Timestamp | None = None, cols_to_keep: list[str] = None) -> tuple[pd.DataFrame, datetime, datetime]:
    """Get saved csv data and filter to regular market hours"""
    df = pd.read_csv(data_path)

    # Apply cutoff first if specified
    if cutoff:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if cutoff.tz is not None:
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert(cutoff.tz)
        else:
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        df = df[df['timestamp'] >= cutoff]
        
    # Filter to regular market hours
    df = filter_to_regular_hours(df)
    
    # Get date range after filtering
    start_date = pd.to_datetime(df['timestamp'].iloc[0]).to_pydatetime()
    end_date = pd.to_datetime(df['timestamp'].iloc[-1]).to_pydatetime()
    
    # Use specific columns if provided
    if cols_to_keep:
        df = df[cols_to_keep]
        
    return df, start_date, end_date


def train_sac(data_path: str,
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
              preprocessor_save_path: Optional[str] = None,
              # NEW: Architecture Selection Parameters
              network_type: NetworkType = NetworkType.SIMPLIFIED,
              environment_type: EnvironmentType = EnvironmentType.WEIGHTED_AVERAGE,
              config_overrides: Optional[Dict] = None) -> Dict:
    """
    Modern SAC training function using modular architecture
    
    Args:
        data_path: Path to CSV file with stock data
        cutoff: Timestamp to start data from
        num_episodes: Number of training episodes
        save_interval: Save model every N episodes
        validation_frequency: Run validation every N episodes
        early_stopping_patience: Stop if validation doesn't improve for N checks
        train_ratio: Ratio of data for training
        valid_ratio: Ratio of data for validation
        use_preprocessing: Whether to apply preprocessing
        scaling_method: Method for scaling features ('robust', 'standard', etc.)
        outlier_method: Method for handling outliers ('winsorize', 'clip', etc.)
        preprocessor_save_path: Path to save the fitted preprocessor
        
        # NEW PARAMETERS:
        network_type: Neural network architecture to use (ORIGINAL or SIMPLIFIED)
        environment_type: Trading environment type (BASIC, WEIGHTED_AVERAGE, LOT_BASED)
        config_overrides: Dict of config parameters to override
    
    Returns:
        Dict with training results, agent, preprocessor, and test metrics
    """
    
    print(f"🚀 Starting SAC Training with Modular Architecture")
    print(f"📊 Network Type: {network_type.value.upper()}")
    print(f"🏪 Environment Type: {environment_type.value.upper()}")
    print("="*60)
    
    # Load data
    print("📁 Loading stock data...")
    from src.config.config import STOCK_FEATURES_V2
    data, start_date, end_date = load_stock_data(data_path, cutoff, STOCK_FEATURES_V2)
    
    # Split data chronologically
    train_end = int(len(data) * train_ratio)
    valid_end = train_end + int(len(data) * valid_ratio)
    
    train_data = data.iloc[:train_end].copy().reset_index(drop=True)
    valid_data = data.iloc[train_end:valid_end].copy().reset_index(drop=True)
    test_data = data.iloc[valid_end:].copy().reset_index(drop=True)
    
    print(f"📈 Data split: Train {len(train_data)}, Validation {len(valid_data)}, Test {len(test_data)}")
    
    # Apply preprocessing if requested
    preprocessor = None
    if use_preprocessing:
        print(f"\n🔧 Applying preprocessing (scaling: {scaling_method}, outliers: {outlier_method})...")
        from src.models.mark.dqn_v2.data_preprocessor_v2 import preprocess_financial_data
        
        if preprocessor_save_path is None:
            data_name = Path(data_path).stem
            preprocessor_save_path = f"sac_preprocessor_{data_name}.pkl"
        
        preprocessor, train_data_scaled, valid_data_scaled, test_data_scaled = preprocess_financial_data(
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            scaling_method=scaling_method,
            outlier_method=outlier_method,
            save_preprocessor=True,
            preprocessor_path=preprocessor_save_path
        )
        print(f"✅ Preprocessing complete. Preprocessor saved to: {preprocessor_save_path}")
    else:
        train_data_scaled = train_data
        valid_data_scaled = valid_data
        test_data_scaled = test_data
    
    # Create SAC configuration with architecture and environment selection
    print(f"\n⚙️ Creating SAC configuration...")
    if environment_type == EnvironmentType.BASIC:
        config = create_basic_sac_config()
    elif environment_type == EnvironmentType.WEIGHTED_AVERAGE:
        config = create_weighted_average_sac_config()
    elif environment_type == EnvironmentType.LOT_BASED:
        config = create_lot_based_sac_config()
    else:
        config = SACConfig()
        config.environment_type = environment_type
    
    # Set network architecture
    config.network_type = network_type
    
    # Apply any configuration overrides
    if config_overrides:
        print(f"🔧 Applying {len(config_overrides)} configuration overrides...")
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"   • {key}: {value}")
            else:
                print(f"   ⚠️ Warning: Unknown config parameter '{key}' ignored")
    
    # Create environments using the new action masking system
    print(f"\n🏭 Creating trading environments...")
    train_env = create_environment(train_data, train_data_scaled, config, mode=TradingMode.TRAIN)
    val_env = create_environment(valid_data, valid_data_scaled, config, mode=TradingMode.VAL)
    test_env = create_environment(test_data, test_data_scaled, config, mode=TradingMode.TEST)
    
    print(f"📊 Data shapes:")
    print(f"   • Train: {train_data_scaled.shape}")
    print(f"   • Valid: {valid_data_scaled.shape}")
    print(f"   • Test: {test_data_scaled.shape}")
    
    # Create SAC agent with selected architecture
    print(f"\n🧠 Creating SAC agent...")
    agent = SAC(config, device=DEVICE)
    
    # Training metrics tracking
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
    
    print(f"\n🎯 Starting SAC training with {network_type.value} networks...")
    
    # Phase 1: Portfolio State Warmup (if needed)
    if train_env.portfolio_normalizer is not None and not train_env.portfolio_normalizer.is_fitted:
        warmup_episodes = train_env.portfolio_normalizer.warmup_episodes
        print(f"\n{'='*60}")
        print(f"📊 PORTFOLIO NORMALIZATION WARMUP PHASE")
        print(f"{'='*60}")
        print(f"🔄 Collecting portfolio states for {warmup_episodes} episodes...")
        print("⏸️ (No training will occur during this phase)")
        
        for warmup_ep in range(warmup_episodes):
            state = train_env.reset()
            episode_portfolio_states = []
            
            while True:
                # Random action during warmup
                action = np.random.uniform(-1, 1)
                next_state, reward, done, info = train_env.step(action)
                
                # Collect portfolio states
                if hasattr(train_env, 'episode_portfolio_states'):
                    episode_portfolio_states.extend(train_env.episode_portfolio_states)
                
                state = next_state
                if done:
                    break
            
            # Add collected states to normalizer
            if episode_portfolio_states:
                train_env.portfolio_normalizer.collect_warmup_data(episode_portfolio_states)
            train_env.portfolio_normalizer.increment_episode()
            
            # Progress update
            if (warmup_ep + 1) % 10 == 0 or warmup_ep == warmup_episodes - 1:
                progress = (warmup_ep + 1) / warmup_episodes
                print(f"   📈 Warmup progress: {warmup_ep + 1}/{warmup_episodes} ({progress*100:.1f}%)")
        
        print(f"\n✅ [PORTFOLIO COLLECTION COMPLETE]")
        print(f"🔧 [FITTING NORMALIZER] Fitting portfolio normalizer...")
        print(f"🎯 [READY] Ready to start SAC training with normalized portfolio features!")
        print(f"\n{'='*60}")
        print(f"🚀 SAC TRAINING PHASE")
        print(f"{'='*60}")
    
    # Main training loop
    for episode in range(num_episodes):
        metrics = agent.train_episode(train_env)
        
        # Store metrics
        episode_rewards.append(metrics['episode_reward'])
        episode_returns.append(metrics['total_return'])
        episode_trades.append(metrics['total_trades'])
        episode_invalid_actions.append(metrics['invalid_actions'])
        
        # Print progress
        print(f"\n📊 Episode {episode+1}/{num_episodes}")
        print(f"   💰 Reward: {metrics['episode_reward']:.4f}")
        print(f"   📈 Return: {metrics['total_return']:.2%}")
        print(f"   💵 Final Value: ${metrics['final_value']:,.2f}")
        print(f"   🔄 Trades: {metrics['total_trades']} (✅{metrics['winning_trades']}/❌{metrics['losing_trades']})")
        print(f"   ⚠️ Invalid Actions: {metrics['invalid_actions']}")
        print(f"   👣 Steps: {metrics['episode_steps']}")
        
        # Print SAC-specific metrics
        if 'alpha' in metrics:
            print(f"   🎛️ Alpha: {metrics['alpha']:.4f}")
        if 'actor_loss' in metrics:
            print(f"   🎭 Actor Loss: {metrics['actor_loss']:.4f}")
        if 'critic1_loss' in metrics:
            print(f"   👁️ Critic Losses: {metrics['critic1_loss']:.4f} / {metrics['critic2_loss']:.4f}")
        
        # Network architecture info
        print(f"   🧠 Network: {network_type.value} ({agent.get_network_summary()['total_params']:,} params)")
        
        # Portfolio normalizer status
        if train_env.portfolio_normalizer is not None:
            print(f"   📊 Portfolio Normalizer: [ACTIVE]")
        
        # Learning rate status (schedulers only step during validation)
        if hasattr(agent, 'get_current_learning_rates') and config.use_lr_scheduler:
            current_lrs = agent.get_current_learning_rates()
            if episode % 20 == 0:  # Print LR status occasionally
                print(f"   📚 Learning Rates: Actor={current_lrs['actor']:.2e}, Critics={current_lrs['critic1']:.2e}")
        
        # Validation
        if (episode + 1) % validation_frequency == 0:
            print(f"\n🔍 Running SAC validation...")
            
            # Use the new evaluate_episode method for cleaner validation
            val_metrics = agent.evaluate_episode(val_env, deterministic=True)
            
            validation_rewards.append(val_metrics['episode_reward'])
            validation_returns.append(val_metrics['total_return'])
            validation_trades.append(val_metrics['total_trades'])
            validation_invalid_actions.append(val_metrics['invalid_actions'])
            
            print(f"✅ Validation Results:")
            print(f"   📈 Return: {val_metrics['total_return']:.2%}")
            print(f"   💵 Final Value: ${val_metrics['final_value']:,.2f}")
            print(f"   🔄 Trades: {val_metrics['total_trades']} (✅{val_metrics['winning_trades']}/❌{val_metrics['losing_trades']})")
            print(f"   ⚠️ Invalid Actions: {val_metrics['invalid_actions']}")
            
            # Early stopping check
            if val_metrics['total_return'] > best_validation_return:
                best_validation_return = val_metrics['total_return']
                patience_counter = 0
                # Save best model state
                best_model_state = {
                    'actor': agent.actor.state_dict(),
                    'critic1': agent.critic1.state_dict(),
                    'critic2': agent.critic2.state_dict(),
                    'target_critic1': agent.target_critic1.state_dict(),
                    'target_critic2': agent.target_critic2.state_dict(),
                    'steps_done': agent.steps_done,
                    'episodes_done': agent.episodes_done
                }
                print(f"   🌟 New best validation return! ({best_validation_return:.2%})")
            else:
                patience_counter += 1
                print(f"   ⏳ Patience: {patience_counter}/{early_stopping_patience}")
            
            # Learning rate scheduler step with validation metric - this is the ONLY place schedulers should step
            if hasattr(agent, 'step_schedulers') and config.use_lr_scheduler:
                old_lrs = agent.get_current_learning_rates()
                agent.step_schedulers(val_metrics['total_return'])
                new_lrs = agent.get_current_learning_rates()
                
                # Check if learning rates changed and report
                lr_changed = any(abs(old_lrs[key] - new_lrs[key]) > 1e-8 for key in old_lrs.keys())
                if lr_changed:
                    print(f"   📉 Learning rates adjusted based on validation metric:")
                    for key in old_lrs.keys():
                        if abs(old_lrs[key] - new_lrs[key]) > 1e-8:
                            print(f"      • {key}: {old_lrs[key]:.2e} → {new_lrs[key]:.2e}")
                else:
                    print(f"   📚 Learning rates unchanged (validation return: {val_metrics['total_return']:.2%})")
            
            if patience_counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping triggered at episode {episode+1}")
                print(f"🏆 Best validation return: {best_validation_return:.2%}")
                
                # Restore best model
                if best_model_state:
                    agent.actor.load_state_dict(best_model_state['actor'])
                    agent.critic1.load_state_dict(best_model_state['critic1'])
                    agent.critic2.load_state_dict(best_model_state['critic2'])
                    agent.target_critic1.load_state_dict(best_model_state['target_critic1'])
                    agent.target_critic2.load_state_dict(best_model_state['target_critic2'])
                    agent.steps_done = best_model_state['steps_done']
                    agent.episodes_done = best_model_state['episodes_done']
                    print(f"🔄 Restored best model weights")
                break
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            checkpoint_path = f"sac_{network_type.value}_{environment_type.value}_episode_{episode}.pt"
            agent.save(checkpoint_path, train_env.portfolio_normalizer)
            print(f"💾 Saved SAC checkpoint: {checkpoint_path}")
    
    # Multi-day test evaluation
    print(f"\n{'='*60}")
    print("🧪 SAC MULTI-DAY TEST EVALUATION (6 DAYS)")
    print(f"{'='*60}")
    
    num_test_days = min(6, test_env.total_days)
    test_days = np.linspace(0, test_env.total_days - 1, num_test_days, dtype=int)
    
    all_test_results = []
    all_portfolio_values = []
    all_price_histories = []
    all_action_histories = []
    
    for i, day_idx in enumerate(test_days):
        print(f"\n🔬 Running SAC backtest for day {day_idx + 1}/{test_env.total_days} (Test {i+1}/{num_test_days})...")
        
        # Use the new evaluate_episode method for consistent testing
        test_state = test_env.reset(day_idx=day_idx)
        test_reward = 0
        test_done = False
        test_action_history = []
        test_portfolio_values = [config.initial_balance]
        test_price_history = []
        
        while not test_done:
            test_action = agent.select_action(test_state, deterministic=True)
            test_next_state, test_r, test_done, test_info = test_env.step(test_action)
            test_reward += test_r
            test_state = test_next_state
            
            # Track for plotting
            test_action_history.append(test_action)
            current_value = test_info['balance'] + (test_info.get('position', 0) * test_info['current_price'])
            test_portfolio_values.append(current_value)
            test_price_history.append(test_info['current_price'])
        
        # Calculate metrics
        test_final_value = test_portfolio_values[-1]  
        test_return = (test_final_value - config.initial_balance) / config.initial_balance
        
        # Performance metrics
        returns = np.diff(test_portfolio_values) / test_portfolio_values[:-1]
        peak = np.maximum.accumulate(test_portfolio_values)
        drawdown = (test_portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        if len(returns) > 0:
            portfolio_volatility = np.std(test_portfolio_values) / np.mean(test_portfolio_values)
            sharpe = test_return / portfolio_volatility if portfolio_volatility > 1e-8 else test_return * 10
        else:
            sharpe = 0.0
        
        day_results = {
            'day_idx': day_idx,
            'final_value': test_final_value,
            'total_return': test_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': test_info['total_trades'],
            'winning_trades': test_info['winning_trades'],
            'losing_trades': test_info['losing_trades'],
            'invalid_actions': test_info['invalid_actions'],
            'episode_reward': test_reward
        }
        
        all_test_results.append(day_results)
        all_portfolio_values.append(test_portfolio_values)
        all_price_histories.append(test_price_history)
        all_action_histories.append(test_action_history)
        
        print(f"   📊 Day {day_idx + 1} Results:")
        print(f"      💵 Final Value: ${test_final_value:,.2f}")
        print(f"      📈 Return: {test_return:.2%}")
        print(f"      📉 Sharpe: {sharpe:.2f}")
        print(f"      📊 Max Drawdown: {max_drawdown:.2%}")
        print(f"      🔄 Trades: {test_info['total_trades']} (✅{test_info['winning_trades']}/❌{test_info['losing_trades']})")
        print(f"      ⚠️ Invalid Actions: {test_info['invalid_actions']}")
    
    # Calculate aggregate statistics
    returns = [r['total_return'] for r in all_test_results]
    final_values = [r['final_value'] for r in all_test_results]
    
    print(f"\n{'='*60}")
    print("📊 AGGREGATE SAC TEST RESULTS")
    print(f"{'='*60}")
    print(f"🧠 Network Architecture: {network_type.value.upper()}")
    print(f"🏪 Environment Type: {environment_type.value.upper()}")
    print(f"📈 Average Return: {np.mean(returns):.2%} ± {np.std(returns):.2%}")
    print(f"🏆 Best Return: {np.max(returns):.2%}")
    print(f"📉 Worst Return: {np.min(returns):.2%}")
    print(f"🎯 Win Rate: {np.sum([r > 0 for r in returns]) / len(returns):.1%}")
    print(f"💰 Average Final Value: ${np.mean(final_values):,.2f}")
    
    # Create results dictionary
    results = {
        'agent': agent,
        'preprocessor': preprocessor,
        'config': config,  # Include config for reference
        'network_type': network_type,
        'environment_type': environment_type,
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'episode_trades': episode_trades,
        'episode_invalid_actions': episode_invalid_actions,
        'validation_rewards': validation_rewards,
        'validation_returns': validation_returns,
        'validation_trades': validation_trades,
        'validation_invalid_actions': validation_invalid_actions,
        'multi_day_test_results': {
            'individual_days': all_test_results,
            'portfolio_values': all_portfolio_values,
            'price_histories': all_price_histories,
            'action_histories': all_action_histories,
            'aggregate_stats': {
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'best_return': np.max(returns),
                'worst_return': np.min(returns),
                'win_rate': np.sum([r > 0 for r in returns]) / len(returns),
                'avg_final_value': np.mean(final_values)  
            }
        },
        'start_date': start_date,
        'end_date': end_date
    }
    
    print(f"\n✅ SAC training completed successfully!")
    print(f"🎯 Using {network_type.value} networks with {environment_type.value} environment")
    return results


if __name__ == "__main__":
    # Example usage with architecture selection
    import pandas as pd
    
    data_path = "data/TSLA_1min_features.csv"
    cutoff = pd.Timestamp("2023-01-01")
    
    # Train with simplified networks (recommended)
    results = train_sac(
        data_path=data_path,
        cutoff=cutoff,
        num_episodes=200,
        save_interval=50,
        validation_frequency=20,
        network_type=NetworkType.SIMPLIFIED,  # Choose architecture
        environment_type=EnvironmentType.WEIGHTED_AVERAGE,  # Choose environment
        config_overrides={  # Custom config overrides
            'actor_learning_rate': 5e-4,
            'target_entropy': -0.3,
            'use_action_guidance': True
        }
    )
    
    print("🎉 SAC training completed!")
    print(f"📊 Final results: {results['multi_day_test_results']['aggregate_stats']}")
