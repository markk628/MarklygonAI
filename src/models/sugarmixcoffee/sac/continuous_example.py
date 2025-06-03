"""
Example usage of the Continuous SAC agent for stock trading with position sizing.

This demonstrates the key advantage of continuous actions: the agent can decide
not just WHEN to trade, but also HOW MUCH to trade.

Action Space:
- Discrete SAC: Hold (0), Buy (1), Sell (2) - binary decisions
- Continuous SAC: [-1, 1] where:
  * -1.0 = Sell all holdings (0% position)
  * -0.5 = Sell half (reduce position by 50%)  
  * 0.0 = Hold current position
  * +0.5 = Increase position by 50% toward max
  * +1.0 = Buy maximum allowed (100% position)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.sugarmixcoffee.sac.continuous_sac_agent import (
    ContinuousSACAgent, ContinuousSACConfig, train_continuous_sac
)
from src.models.sugarmixcoffee.sac.continuous_sac import ContinuousTradingEnvironment, TradingMode, load_stock_data
from src.config.config import DATA_DIR, INITIAL_BALANCE


def continuous_sac_demo():
    """Demonstrate continuous SAC with position sizing"""
    
    print("Continuous SAC Trading Agent Demo")
    print("=" * 50)
    print("Key Feature: Continuous position sizing instead of binary buy/sell decisions")
    print()
    
    # Configuration
    ticker = 'TSLA'
    data_path = f"{DATA_DIR}/feature_engineered/{ticker}.csv"
    cutoff = pd.Timestamp('2024-05-06 08:00:00', tz='UTC')
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    print(f"Training Continuous SAC agent on {ticker} data...")
    print("This will take longer than discrete SAC due to continuous action space complexity.")
    print()
    
    # Train with few episodes for demo
    training_results = train_continuous_sac(
        data_path=data_path,
        cutoff=cutoff,
        num_episodes=15,  # Small number for demo
        validation_frequency=5,
        early_stopping_patience=3,
        use_preprocessing=True,
        scaling_method='robust',
        outlier_method='winsorize'
    )
    
    # Analyze results
    test_results = training_results['test_results']
    
    print("\nContinuous SAC Training Complete!")
    print("-" * 40)
    print(f"Episodes trained: {len(training_results['episode_rewards'])}")
    print(f"Final training return: {training_results['episode_returns'][-1]:.2%}")
    print(f"Average training return: {np.mean(training_results['episode_returns']):.2%}")
    
    print(f"\nTest Results:")
    print(f"  Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"  Final Value: ${test_results['final_value']:,.2f}")
    print(f"  Total Return: {test_results['total_return']:.2%}")
    print(f"  Sharpe Ratio: {test_results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {test_results['max_drawdown']:.2%}")
    print(f"  Total Trades: {test_results['total_trades']}")
    print(f"  Win Rate: {test_results['win_rate']:.2%}")
    print(f"  Total Fees: ${test_results['total_fees_paid']:.2f}")
    
    # Analyze position sizing behavior
    print(f"\nPosition Sizing Analysis:")
    action_history = test_results['action_history']
    position_history = test_results['position_history']
    
    if action_history and position_history:
        print(f"  Action Range: {min(action_history):.3f} to {max(action_history):.3f}")
        print(f"  Position Range: {min(position_history):.1%} to {max(position_history):.1%}")
        print(f"  Average Position: {np.mean(position_history):.1%}")
        
        # Count different action types
        strong_buy = sum(1 for a in action_history if a > 0.5)
        moderate_buy = sum(1 for a in action_history if 0.1 < a <= 0.5)
        hold = sum(1 for a in action_history if -0.1 <= a <= 0.1)
        moderate_sell = sum(1 for a in action_history if -0.5 <= a < -0.1)
        strong_sell = sum(1 for a in action_history if a < -0.5)
        
        total_actions = len(action_history)
        print(f"\nAction Distribution:")
        print(f"  Strong Buy (>0.5): {strong_buy} ({strong_buy/total_actions:.1%})")
        print(f"  Moderate Buy (0.1-0.5): {moderate_buy} ({moderate_buy/total_actions:.1%})")
        print(f"  Hold (-0.1 to 0.1): {hold} ({hold/total_actions:.1%})")
        print(f"  Moderate Sell (-0.5 to -0.1): {moderate_sell} ({moderate_sell/total_actions:.1%})")
        print(f"  Strong Sell (<-0.5): {strong_sell} ({strong_sell/total_actions:.1%})")
    
    # Compare with buy-and-hold
    price_history = test_results['price_history']
    if price_history:
        initial_price = price_history[0]
        final_price = price_history[-1]
        shares_bought = INITIAL_BALANCE / initial_price
        buy_hold_final = shares_bought * final_price
        buy_hold_return = (buy_hold_final - INITIAL_BALANCE) / INITIAL_BALANCE
        
        print(f"\nComparison with Buy & Hold:")
        print(f"  Buy & Hold Return: {buy_hold_return:.2%}")
        print(f"  Continuous SAC Return: {test_results['total_return']:.2%}")
        print(f"  Outperformance: {test_results['total_return'] - buy_hold_return:.2%}")
        
        # Fee analysis
        fee_impact = test_results['total_fees_paid'] / INITIAL_BALANCE
        print(f"  Fee Impact: {fee_impact:.2%} of initial capital")
    
    return training_results


def plot_continuous_sac_behavior(test_results: dict, ticker: str = "Stock"):
    """Plot the continuous action and position sizing behavior"""
    
    price_history = test_results['price_history']
    action_history = test_results['action_history']
    position_history = test_results['position_history']
    portfolio_values = test_results['portfolio_values']
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Stock price
    axes[0].plot(price_history, label=f'{ticker} Price', color='black', linewidth=2)
    axes[0].set_title(f'{ticker} Stock Price')
    axes[0].set_ylabel('Price ($)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Continuous actions
    axes[1].plot(action_history, label='Action Values', color='blue', alpha=0.7)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Strong Buy')
    axes[1].axhline(y=-0.5, color='red', linestyle=':', alpha=0.5, label='Strong Sell')
    axes[1].set_title('Continuous Actions (-1=Sell All, 0=Hold, +1=Buy Max)')
    axes[1].set_ylabel('Action Value')
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Position sizing
    axes[2].plot([p * 100 for p in position_history], label='Position %', color='orange', linewidth=2)
    axes[2].set_title('Position Sizing Over Time')
    axes[2].set_ylabel('Position (%)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Portfolio value
    axes[3].plot(portfolio_values, label='Continuous SAC', color='blue', linewidth=2)
    
    # Buy and hold comparison
    initial_balance = portfolio_values[0]
    initial_price = price_history[0]
    shares_bought = initial_balance / initial_price
    buy_hold_values = [initial_balance] + [shares_bought * price for price in price_history]
    axes[3].plot(buy_hold_values, '--', label='Buy & Hold', color='orange', alpha=0.7)
    
    axes[3].set_title('Portfolio Value Comparison')
    axes[3].set_ylabel('Portfolio Value ($)')
    axes[3].set_xlabel('Trading Steps')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    plt.tight_layout()
    return fig


def compare_discrete_vs_continuous():
    """Compare discrete SAC vs continuous SAC conceptually"""
    
    print("\nDiscrete vs Continuous SAC Comparison")
    print("=" * 45)
    
    print("\nDiscrete SAC (Original):")
    print("  Actions: Hold (0), Buy (1), Sell (2)")
    print("  Position: Binary - either 0% or 70% position")
    print("  Flexibility: Limited - all-or-nothing trades")
    print("  Use Case: Simple trading decisions")
    
    print("\nContinuous SAC (New):")
    print("  Actions: Continuous range [-1, 1]")
    print("  Position: Flexible - any position from 0% to 100%")
    print("  Flexibility: High - gradual position adjustments")
    print("  Use Case: Sophisticated position management")
    
    print("\nKey Benefits of Continuous Actions:")
    print("  ✓ Gradual position building/unwinding")
    print("  ✓ Risk management through position sizing")
    print("  ✓ Reduced transaction costs (fewer large trades)")
    print("  ✓ More realistic trading behavior")
    print("  ✓ Better capital utilization")
    
    print("\nExample Scenarios:")
    print("  Market Uncertainty: Take 30% position instead of 0% or 70%")
    print("  Strong Signal: Gradually increase from 30% to 80%")
    print("  Risk Management: Reduce from 70% to 40% on volatility")
    print("  Profit Taking: Sell 25% of position, keep 75%")


def load_and_test_continuous_model(model_path: str, data_path: str, cutoff: pd.Timestamp):
    """Load and test a pre-trained continuous SAC model"""
    
    print(f"Loading Continuous SAC model from: {model_path}")
    
    # Load data
    data, _, _ = load_stock_data(data_path, cutoff)
    eval_data = data.iloc[-500:].copy().reset_index(drop=True)  # Last 500 points
    
    # Create environment and agent
    config = ContinuousSACConfig()
    env = ContinuousTradingEnvironment(eval_data, eval_data, config, mode=TradingMode.TEST)
    agent = ContinuousSACAgent(config)
    agent.load(model_path)
    
    print(f"Testing on {len(eval_data)} data points...")
    
    # Run evaluation
    state = env.reset()
    portfolio_values = [config.initial_balance]
    action_values = []
    position_ratios = []
    
    done = False
    while not done:
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        
        current_value = info['balance'] + info['position_value']
        portfolio_values.append(current_value)
        action_values.append(info['action_value'])
        position_ratios.append(info['target_position_ratio'])
        
        state = next_state
    
    # Calculate metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - config.initial_balance) / config.initial_balance
    
    print(f"\nEvaluation Results:")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Final Value: ${final_value:,.2f}")
    print(f"  Total Trades: {info['total_trades']}")
    print(f"  Avg Position: {np.mean(position_ratios):.1%}")
    print(f"  Action Range: {min(action_values):.3f} to {max(action_values):.3f}")
    
    return {
        'portfolio_values': portfolio_values,
        'action_values': action_values,
        'position_ratios': position_ratios,
        'total_return': total_return,
        'info': info
    }


if __name__ == "__main__":
    # Run the demo
    try:
        print("Starting Continuous SAC Demo...")
        results = continuous_sac_demo()
        
        if results:
            print("\nGenerating behavioral analysis plot...")
            plot = plot_continuous_sac_behavior(results['test_results'], 'TSLA')
            plot.savefig('continuous_sac_behavior.png', dpi=150, bbox_inches='tight')
            print("Plot saved as 'continuous_sac_behavior.png'")
            plt.close()
        
        # Show comparison
        compare_discrete_vs_continuous()
        
        print("\nDemo completed successfully!")
        print("\nTo load and test a saved model:")
        print("results = load_and_test_continuous_model('continuous_sac_final_model.pt', data_path, cutoff)")
        
    except Exception as e:
        print(f"Error running demo: {e}")
        print("Make sure you have the required data files and dependencies.") 