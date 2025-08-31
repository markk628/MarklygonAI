"""
SAC Trading Agent Visualization Module
====================================

Visualization functions for SAC training results, backtests, and multi-day comparisons.
Adapted from DQN visualizations but enhanced for SAC-specific metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_sac_training_results(results: Dict, ticker: str = "STOCK", validation_frequency: int = 20):
    """
    Plot comprehensive SAC training results including SAC-specific metrics
    
    Args:
        results: Dictionary containing training results from train_sac()
        ticker: Stock ticker symbol for plot titles
        validation_frequency: Frequency of validation episodes
    
    Returns:
        matplotlib.figure.Figure: Training results plot
    """
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'SAC Training Results - {ticker}', fontsize=16, fontweight='bold')
    
    # Extract data
    episode_rewards = results['episode_rewards']
    episode_returns = results['episode_returns']
    episode_trades = results['episode_trades']
    episode_invalid_actions = results['episode_invalid_actions']
    
    validation_rewards = results.get('validation_rewards', [])
    validation_returns = results.get('validation_returns', [])
    validation_trades = results.get('validation_trades', [])
    validation_invalid_actions = results.get('validation_invalid_actions', [])
    
    episodes = range(1, len(episode_rewards) + 1)
    val_episodes = range(validation_frequency, len(episode_rewards) + 1, validation_frequency)
    
    # 1. Episode Rewards (Training vs Validation)
    axes[0, 0].plot(episodes, episode_rewards, label='Training', alpha=0.7, linewidth=0.8)
    if validation_rewards:
        axes[0, 0].plot(val_episodes[:len(validation_rewards)], validation_rewards, 
                       label='Validation', marker='o', linewidth=2)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Episode Returns (Training vs Validation)
    axes[0, 1].plot(episodes, np.array(episode_returns) * 100, label='Training', alpha=0.7, linewidth=0.8)
    if validation_returns:
        axes[0, 1].plot(val_episodes[:len(validation_returns)], np.array(validation_returns) * 100, 
                       label='Validation', marker='o', linewidth=2)
    axes[0, 1].set_title('Episode Returns (%)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Trading Activity
    axes[0, 2].plot(episodes, episode_trades, label='Training Trades', alpha=0.7, linewidth=0.8)
    if validation_trades:
        axes[0, 2].plot(val_episodes[:len(validation_trades)], validation_trades, 
                       label='Validation Trades', marker='o', linewidth=2)
    axes[0, 2].set_title('Trades per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Number of Trades')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Invalid Actions
    axes[1, 0].plot(episodes, episode_invalid_actions, label='Training', alpha=0.7, linewidth=0.8)
    if validation_invalid_actions:
        axes[1, 0].plot(val_episodes[:len(validation_invalid_actions)], validation_invalid_actions, 
                       label='Validation', marker='o', linewidth=2)
    axes[1, 0].set_title('Invalid Actions per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Invalid Actions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Moving Averages
    window = min(50, len(episode_rewards) // 10)
    if window > 1:
        moving_avg_rewards = pd.Series(episode_rewards).rolling(window=window).mean()
        moving_avg_returns = pd.Series(episode_returns).rolling(window=window).mean()
        
        axes[1, 1].plot(episodes, moving_avg_rewards, label=f'{window}-Episode MA Rewards', linewidth=2)
        axes[1, 1].set_title(f'Moving Average Rewards (Window={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(episodes, moving_avg_returns * 100, label=f'{window}-Episode MA Returns', linewidth=2, color='orange')
        axes[1, 2].set_title(f'Moving Average Returns (Window={window})')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Return (%)')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Not enough episodes\nfor moving average', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 2].text(0.5, 0.5, 'Not enough episodes\nfor moving average', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
    
    # 6. Performance Distribution
    axes[2, 0].hist(episode_returns, bins=30, alpha=0.7, edgecolor='black')
    axes[2, 0].axvline(np.mean(episode_returns), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(episode_returns):.2%}')
    axes[2, 0].set_title('Return Distribution')
    axes[2, 0].set_xlabel('Return')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 7. Win Rate Over Time
    win_rate_window = min(20, len(episode_returns) // 5)
    if win_rate_window > 1:
        rolling_wins = pd.Series([1 if r > 0 else 0 for r in episode_returns]).rolling(window=win_rate_window).mean()
        axes[2, 1].plot(episodes, rolling_wins * 100, linewidth=2, color='green')
        axes[2, 1].set_title(f'Rolling Win Rate (Window={win_rate_window})')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Win Rate (%)')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_ylim(0, 100)
    else:
        axes[2, 1].text(0.5, 0.5, 'Not enough episodes\nfor win rate', 
                       ha='center', va='center', transform=axes[2, 1].transAxes)
    
    # 8. SAC-specific metrics summary
    final_stats = {
        'Total Episodes': len(episode_rewards),
        'Final Avg Return': f"{np.mean(episode_returns[-10:]):.2%}" if len(episode_returns) >= 10 else f"{np.mean(episode_returns):.2%}",
        'Best Episode': f"{np.max(episode_returns):.2%}",
        'Worst Episode': f"{np.min(episode_returns):.2%}",
        'Win Rate': f"{sum(1 for r in episode_returns if r > 0) / len(episode_returns):.1%}",
        'Avg Trades/Episode': f"{np.mean(episode_trades):.1f}",
        'Total Invalid Actions': f"{sum(episode_invalid_actions)}",
    }
    
    # Display stats as text
    stats_text = '\n'.join([f'{k}: {v}' for k, v in final_stats.items()])
    axes[2, 2].text(0.05, 0.95, stats_text, transform=axes[2, 2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[2, 2].set_title('Training Summary')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    return fig


def plot_sac_backtest_results(results: Dict, ticker: str = "STOCK"):
    """
    Plot SAC backtest results for multi-day testing
    
    Args:
        results: Dictionary containing training results from train_sac()
        ticker: Stock ticker symbol for plot titles
    
    Returns:
        matplotlib.figure.Figure: Backtest results plot
    """
    
    multi_day_results = results.get('multi_day_test_results', {})
    if not multi_day_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No multi-day test results available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'SAC Backtest Results - {ticker}')
        return fig
    
    individual_days = multi_day_results['individual_days']
    portfolio_values = multi_day_results['portfolio_values']
    price_histories = multi_day_results['price_histories']
    action_histories = multi_day_results['action_histories']
    
    num_days = len(individual_days)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'SAC Multi-Day Backtest Results - {ticker}', fontsize=16, fontweight='bold')
    
    # 1. Portfolio Values for Each Day
    colors = plt.cm.tab10(np.linspace(0, 1, num_days))
    for i, (portfolio_vals, day_result) in enumerate(zip(portfolio_values, individual_days)):
        day_return = day_result['total_return']
        axes[0, 0].plot(portfolio_vals, color=colors[i], 
                       label=f'Day {i+1}: {day_return:.1%}', linewidth=1.5)
    
    axes[0, 0].set_title('Portfolio Value Evolution')
    axes[0, 0].set_xlabel('Minutes')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Daily Returns
    daily_returns = [day['total_return'] * 100 for day in individual_days]
    bars = axes[0, 1].bar(range(1, num_days + 1), daily_returns, 
                         color=['green' if r > 0 else 'red' for r in daily_returns],
                         alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Daily Returns (%)')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, daily_returns):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                       f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 3. SAC Action Distribution (Continuous Actions)
    all_actions = [action for day_actions in action_histories for action in day_actions]
    if all_actions:
        axes[0, 2].hist(all_actions, bins=50, alpha=0.7, edgecolor='black', density=True)
        axes[0, 2].axvline(0, color='red', linestyle='--', linewidth=2, label='Hold (0)')
        axes[0, 2].axvline(np.mean(all_actions), color='orange', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(all_actions):.3f}')
        axes[0, 2].set_title('SAC Action Distribution')
        axes[0, 2].set_xlabel('Action Value')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add text annotations
        buy_actions = sum(1 for a in all_actions if a > 0.01)
        sell_actions = sum(1 for a in all_actions if a < -0.01)
        hold_actions = len(all_actions) - buy_actions - sell_actions
        
        action_text = f'Buy: {buy_actions/len(all_actions):.1%}\nSell: {sell_actions/len(all_actions):.1%}\nHold: {hold_actions/len(all_actions):.1%}'
        axes[0, 2].text(0.02, 0.98, action_text, transform=axes[0, 2].transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Trading Statistics
    trade_stats = []
    for day_result in individual_days:
        trade_stats.append({
            'Total Trades': day_result['total_trades'],
            'Winning Trades': day_result['winning_trades'],
            'Losing Trades': day_result['losing_trades'],
            'Invalid Actions': day_result['invalid_actions']
        })
    
    trade_df = pd.DataFrame(trade_stats)
    x_pos = np.arange(len(trade_df))
    
    width = 0.2
    axes[1, 0].bar(x_pos - width*1.5, trade_df['Total Trades'], width, label='Total', alpha=0.8)
    axes[1, 0].bar(x_pos - width*0.5, trade_df['Winning Trades'], width, label='Winning', alpha=0.8)
    axes[1, 0].bar(x_pos + width*0.5, trade_df['Losing Trades'], width, label='Losing', alpha=0.8)
    axes[1, 0].bar(x_pos + width*1.5, trade_df['Invalid Actions'], width, label='Invalid', alpha=0.8)
    
    axes[1, 0].set_title('Trading Activity by Day')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f'Day {i+1}' for i in range(len(trade_df))])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Risk Metrics
    aggregate_stats = multi_day_results['aggregate_stats']
    
    risk_metrics = {
        'Average Return': f"{aggregate_stats['avg_return']:.2%}",
        'Return Std Dev': f"{aggregate_stats['std_return']:.2%}",
        'Best Return': f"{aggregate_stats['best_return']:.2%}",
        'Worst Return': f"{aggregate_stats['worst_return']:.2%}",
        'Win Rate': f"{aggregate_stats['win_rate']:.1%}",
        'Avg Final Value': f"${aggregate_stats['avg_final_value']:,.0f}",
    }
    
    risk_text = '\n'.join([f'{k}: {v}' for k, v in risk_metrics.items()])
    axes[1, 1].text(0.05, 0.95, risk_text, transform=axes[1, 1].transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_title('SAC Performance Metrics')
    axes[1, 1].axis('off')
    
    # 6. Final Portfolio Values
    final_values = [day['final_value'] for day in individual_days]
    initial_balance = 100000  # Default initial balance
    
    bars = axes[1, 2].bar(range(1, num_days + 1), final_values, 
                         color=['green' if v > initial_balance else 'red' for v in final_values],
                         alpha=0.7, edgecolor='black')
    axes[1, 2].axhline(y=initial_balance, color='blue', linestyle='--', linewidth=2, 
                      label=f'Initial: ${initial_balance:,.0f}')
    axes[1, 2].set_title('Final Portfolio Values')
    axes[1, 2].set_xlabel('Day')
    axes[1, 2].set_ylabel('Portfolio Value ($)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_values):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + (height * 0.01),
                       f'${value:,.0f}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    plt.tight_layout()
    return fig


def plot_sac_multi_day_comparison(results: Dict, ticker: str = "STOCK"):
    """
    Plot SAC multi-day comparison with detailed analysis
    
    Args:
        results: Dictionary containing training results from train_sac()
        ticker: Stock ticker symbol for plot titles
    
    Returns:
        matplotlib.figure.Figure: Multi-day comparison plot
    """
    
    multi_day_results = results.get('multi_day_test_results', {})
    if not multi_day_results:
        return None
    
    individual_days = multi_day_results['individual_days']
    portfolio_values = multi_day_results['portfolio_values']
    action_histories = multi_day_results['action_histories']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'SAC Multi-Day Analysis - {ticker}', fontsize=16, fontweight='bold')
    
    # 1. Normalized Portfolio Performance
    for i, portfolio_vals in enumerate(portfolio_values):
        normalized_vals = np.array(portfolio_vals) / portfolio_vals[0] * 100
        day_return = individual_days[i]['total_return']
        axes[0, 0].plot(normalized_vals, label=f'Day {i+1}: {day_return:.1%}', linewidth=1.5)
    
    axes[0, 0].set_title('Normalized Portfolio Performance (Base=100)')
    axes[0, 0].set_xlabel('Minutes')
    axes[0, 0].set_ylabel('Normalized Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=100, color='black', linestyle='--', alpha=0.5)
    
    # 2. SAC Action Patterns by Day
    for i, actions in enumerate(action_histories):
        axes[0, 1].plot(actions, alpha=0.7, label=f'Day {i+1}', linewidth=1)
    
    axes[0, 1].set_title('SAC Action Patterns')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Action Value')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Hold')
    axes[0, 1].axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Strong Buy')
    axes[0, 1].axhline(y=-0.5, color='orange', linestyle=':', alpha=0.5, label='Strong Sell')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Return vs Trading Activity
    returns = [day['total_return'] * 100 for day in individual_days]
    total_trades = [day['total_trades'] for day in individual_days]
    
    scatter = axes[1, 0].scatter(total_trades, returns, 
                                c=range(len(returns)), cmap='viridis', 
                                s=100, alpha=0.7, edgecolors='black')
    
    # Add day labels
    for i, (trades, ret) in enumerate(zip(total_trades, returns)):
        axes[1, 0].annotate(f'D{i+1}', (trades, ret), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    axes[1, 0].set_title('Return vs Trading Activity')
    axes[1, 0].set_xlabel('Total Trades')
    axes[1, 0].set_ylabel('Return (%)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add correlation text
    correlation = np.corrcoef(total_trades, returns)[0, 1]
    axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axes[1, 0].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. SAC Performance Summary
    aggregate_stats = multi_day_results['aggregate_stats']
    
    summary_stats = {
        'Days Tested': len(individual_days),
        'Avg Return': f"{aggregate_stats['avg_return']:.2%}",
        'Std Dev': f"{aggregate_stats['std_return']:.2%}",
        'Win Rate': f"{aggregate_stats['win_rate']:.1%}",
        'Best Day': f"{aggregate_stats['best_return']:.2%}",
        'Worst Day': f"{aggregate_stats['worst_return']:.2%}",
        'Sharpe Ratio': f"{aggregate_stats['avg_return'] / aggregate_stats['std_return']:.2f}" if aggregate_stats['std_return'] > 0 else "N/A",
    }
    
    # Create performance summary table
    summary_text = "SAC PERFORMANCE SUMMARY\n" + "="*25 + "\n"
    summary_text += '\n'.join([f'{k:<15}: {v}' for k, v in summary_stats.items()])
    
    # Add SAC-specific insights
    summary_text += "\n\nSAC INSIGHTS:\n" + "-"*15 + "\n"
    
    # Action analysis
    all_actions = [action for day_actions in action_histories for action in day_actions]
    buy_ratio = sum(1 for a in all_actions if a > 0.01) / len(all_actions)
    sell_ratio = sum(1 for a in all_actions if a < -0.01) / len(all_actions)
    hold_ratio = 1 - buy_ratio - sell_ratio
    
    summary_text += f"Action Dist    : {buy_ratio:.1%}B/{sell_ratio:.1%}S/{hold_ratio:.1%}H\n"
    summary_text += f"Avg Action     : {np.mean(all_actions):.3f}\n"
    summary_text += f"Action Std     : {np.std(all_actions):.3f}\n"
    
    # Trading efficiency
    total_invalid = sum(day['invalid_actions'] for day in individual_days)
    total_valid_trades = sum(day['total_trades'] for day in individual_days)
    efficiency = total_valid_trades / (total_valid_trades + total_invalid) if (total_valid_trades + total_invalid) > 0 else 0
    
    summary_text += f"Trade Efficiency: {efficiency:.1%}\n"
    summary_text += f"Avg Trades/Day : {np.mean(total_trades):.1f}"
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[1, 1].set_title('SAC Analysis Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig 