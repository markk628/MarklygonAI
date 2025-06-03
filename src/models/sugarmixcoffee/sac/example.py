"""
Example usage of the SAC (Soft Actor-Critic) agent for stock trading.

This script demonstrates how to:
1. Load and preprocess stock data
2. Train a SAC agent
3. Evaluate the trained agent
4. Compare performance with buy-and-hold
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.models.sugarmixcoffee.sac import SACAgent, SACConfig, train_sac
from src.models.mark.dqn_v2.dqn import TradingEnvironment, TradingMode, load_stock_data
from src.config.config import DATA_DIR, INITIAL_BALANCE


def simple_sac_example():
    """Simple example of training and evaluating a SAC agent"""
    
    print("SAC Trading Agent Example")
    print("=" * 50)
    
    # Configuration
    ticker = 'TSLA'
    data_path = f"{DATA_DIR}/feature_engineered/{ticker}.csv"
    cutoff = pd.Timestamp('2024-05-06 08:00:00', tz='UTC')
    
    # Check if data file exists
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure you have the feature engineered data for the ticker.")
        return
    
    print(f"Training SAC agent on {ticker} data...")
    
    # Train the SAC agent with minimal episodes for demonstration
    training_results = train_sac(
        data_path=data_path,
        cutoff=cutoff,
        num_episodes=20,  # Small number for quick demo
        validation_frequency=5,
        early_stopping_patience=3,
        use_preprocessing=True,
        scaling_method='robust',
        outlier_method='winsorize'
    )
    
    # Print results
    test_results = training_results['test_results']
    
    print("\nTraining Complete!")
    print("-" * 30)
    print(f"Episodes trained: {len(training_results['episode_rewards'])}")
    print(f"Final episode return: {training_results['episode_returns'][-1]:.2%}")
    print(f"Average training return: {np.mean(training_results['episode_returns']):.2%}")
    
    print(f"\nTest Results:")
    print(f"  Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"  Final Value: ${test_results['final_value']:,.2f}")
    print(f"  Total Return: {test_results['total_return']:.2%}")
    print(f"  Sharpe Ratio: {test_results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {test_results['max_drawdown']:.2%}")
    print(f"  Total Trades: {test_results['total_trades']}")
    print(f"  Win Rate: {test_results['win_rate']:.2%}")
    print(f"  Invalid Actions: {test_results['invalid_actions']}")
    
    # Calculate buy-and-hold performance for comparison
    price_history = test_results['price_history']
    if price_history:
        initial_price = price_history[0]
        final_price = price_history[-1]
        shares_bought = INITIAL_BALANCE / initial_price
        buy_hold_final = shares_bought * final_price
        buy_hold_return = (buy_hold_final - INITIAL_BALANCE) / INITIAL_BALANCE
        
        print(f"\nComparison with Buy & Hold:")
        print(f"  Buy & Hold Final Value: ${buy_hold_final:,.2f}")
        print(f"  Buy & Hold Return: {buy_hold_return:.2%}")
        print(f"  SAC Outperformance: {test_results['total_return'] - buy_hold_return:.2%}")
    
    return training_results


def load_and_evaluate_sac_model(model_path: str, data_path: str, cutoff: pd.Timestamp):
    """Load a pre-trained SAC model and evaluate it"""
    
    print(f"Loading SAC model from: {model_path}")
    
    # Load data
    data, _, _ = load_stock_data(data_path, cutoff)
    
    # Use last 20% of data for evaluation
    eval_start = int(len(data) * 0.8)
    eval_data = data.iloc[eval_start:].copy().reset_index(drop=True)
    
    # Create environment
    config = SACConfig()
    env = TradingEnvironment(eval_data, eval_data, config, mode=TradingMode.TEST)
    
    # Create and load agent
    agent = SACAgent(config)
    agent.load(model_path)
    
    print(f"Evaluating on {len(eval_data)} data points...")
    
    # Run evaluation
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    portfolio_values = [config.initial_balance]
    
    while not done:
        action = agent.select_action(state, deterministic=True)  # No exploration
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        current_value = info['balance'] + (info['position'] * info['current_price'])
        portfolio_values.append(current_value)
        
        state = next_state
    
    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - config.initial_balance) / config.initial_balance
    
    print(f"\nEvaluation Results:")
    print(f"  Steps: {step_count}")
    print(f"  Total Reward: {total_reward:.4f}")
    print(f"  Final Value: ${final_value:,.2f}")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Total Trades: {info['total_trades']}")
    print(f"  Win Rate: {info['winning_trades'] / max(1, info['total_trades']):.2%}")
    
    return {
        'total_reward': total_reward,
        'final_value': final_value,
        'total_return': total_return,
        'portfolio_values': portfolio_values,
        'info': info
    }


def compare_sac_vs_random():
    """Compare SAC agent performance vs random actions"""
    
    print("Comparing SAC vs Random Agent")
    print("=" * 40)
    
    ticker = 'TSLA'
    data_path = f"{DATA_DIR}/feature_engineered/{ticker}.csv"
    cutoff = pd.Timestamp('2024-05-06 08:00:00', tz='UTC')
    
    # Load data
    data, _, _ = load_stock_data(data_path, cutoff)
    test_data = data.iloc[-1000:].copy().reset_index(drop=True)  # Last 1000 points
    
    config = SACConfig()
    
    # Test random agent
    print("Testing random agent...")
    env_random = TradingEnvironment(test_data, test_data, config, mode=TradingMode.TEST)
    state = env_random.reset()
    random_reward = 0
    done = False
    
    while not done:
        action = np.random.choice(3)  # Random action
        next_state, reward, done, info = env_random.step(action)
        random_reward += reward
        state = next_state
    
    random_final = info['balance'] + (info['position'] * info['current_price'])
    random_return = (random_final - config.initial_balance) / config.initial_balance
    
    print(f"Random Agent Results:")
    print(f"  Final Value: ${random_final:,.2f}")
    print(f"  Return: {random_return:.2%}")
    print(f"  Trades: {info['total_trades']}")
    
    # Test trained SAC agent (would need a pre-trained model)
    print("\nTo compare with SAC, train a model first using simple_sac_example()")
    
    return {
        'random_return': random_return,
        'random_final': random_final
    }


if __name__ == "__main__":
    # Run the simple example
    try:
        results = simple_sac_example()
        print("\nExample completed successfully!")
        
        # Uncomment the following lines to run additional examples:
        # compare_sac_vs_random()
        
        # To load and evaluate a saved model:
        # eval_results = load_and_evaluate_sac_model("sac_final_model.pt", data_path, cutoff)
        
    except Exception as e:
        print(f"Error running example: {e}")
        print("Make sure you have the required data files and dependencies installed.") 