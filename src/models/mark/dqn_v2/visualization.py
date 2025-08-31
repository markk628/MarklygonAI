import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional


def plot_multi_day_backtests(results: dict, save_path: str = None, show_plot: bool = True):
    """
    Plot portfolio performance for multiple days on the same figure
    
    Args:
        results: Results dictionary from train_dqn function
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta
    
    # Extract multi-day test results
    multi_day_results = results['multi_day_test_results']
    portfolio_values = multi_day_results['portfolio_values']
    individual_days = multi_day_results['individual_days']
    aggregate_stats = multi_day_results['aggregate_stats']
    
    num_days = len(individual_days)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Expanded color palette for up to 20+ days using matplotlib colormap
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    if num_days <= 10:
        # Use distinct colors for smaller numbers
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    else:
        # Generate colors using colormap for larger numbers
        colormap = cm.get_cmap('tab20')  # Good for up to 20 distinct colors
        colors = [colormap(i / max(num_days - 1, 1)) for i in range(num_days)]
    
    # Plot 1: Portfolio Values
    ax1.set_title('Multi-Day Portfolio Performance Comparison', fontsize=16, fontweight='bold')
    
    for i, (portfolio_vals, day_info) in enumerate(zip(portfolio_values, individual_days)):
        day_idx = day_info['day_idx']
        return_pct = day_info['total_return']
        
        # Create time axis (minutes within trading day)
        time_points = list(range(len(portfolio_vals)))
        
        # Plot portfolio value
        color = colors[i % len(colors)]
        ax1.plot(time_points, portfolio_vals, 
                label=f'Day {day_idx + 1} (Return: {return_pct:.1%})', 
                color=color, linewidth=2, alpha=0.8)
        
        # Add final value annotation for first 10 days only (to avoid clutter)
        if i < 10:
            final_val = portfolio_vals[-1]
            ax1.annotate(f'${final_val:,.0f}', 
                        xy=(len(time_points)-1, final_val),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=9, color=color, fontweight='bold')
    
    # Add horizontal line for initial balance
    initial_balance = results['agent'].config.initial_balance
    ax1.axhline(y=initial_balance, color='black', linestyle='--', alpha=0.5, 
                label=f'Initial Balance (${initial_balance:,.0f})')
    
    ax1.set_xlabel('Minutes into Trading Day')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Normalized Returns (all starting at 100%)
    ax2.set_title('Normalized Returns Comparison (Starting at 100%)', fontsize=14, fontweight='bold')
    
    for i, (portfolio_vals, day_info) in enumerate(zip(portfolio_values, individual_days)):
        day_idx = day_info['day_idx']
        return_pct = day_info['total_return']
        
        # Normalize to percentage returns starting at 100%
        normalized_returns = [(val / portfolio_vals[0]) * 100 for val in portfolio_vals]
        time_points = list(range(len(normalized_returns)))
        
        color = colors[i % len(colors)]
        ax2.plot(time_points, normalized_returns, 
                label=f'Day {day_idx + 1} (Final: {normalized_returns[-1]:.1f}%)', 
                color=color, linewidth=2, alpha=0.8)
        
        # Add final percentage annotation for first 10 days only (to avoid clutter)
        if i < 10:
            final_pct = normalized_returns[-1]
            ax2.annotate(f'{final_pct:.1f}%', 
                        xy=(len(time_points)-1, final_pct),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=9, color=color, fontweight='bold')
    
    # Add horizontal line at 100%
    ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Break-even (100%)')
    
    ax2.set_xlabel('Minutes into Trading Day')
    ax2.set_ylabel('Portfolio Value (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    # Add aggregate statistics as text box (dynamic number of days)
    stats_text = f"""Aggregate Statistics ({num_days} Days):
    Average Return: {aggregate_stats['avg_return']:.1%} ± {aggregate_stats['std_return']:.1%}
    Best Return: {aggregate_stats['best_return']:.1%}
    Worst Return: {aggregate_stats['worst_return']:.1%}
    Win Rate: {aggregate_stats['win_rate']:.0%}
    Avg Trades/Day: {aggregate_stats['avg_trades']:.1f}
    Avg Sharpe Ratio: {aggregate_stats['avg_sharpe_ratio']:.2f}"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_training_results(training_results: dict, ticker: str = "Stock", validation_frequency: int = 5):
    """
    Visualize training and performance metrics
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training rewards/scores
    plt.subplot(2, 3, 1)
    plt.plot(training_results['episode_rewards'], label='Training Reward', alpha=0.7)
    if training_results['validation_rewards']:
        validation_episodes = [i*validation_frequency for i in range(len(training_results['validation_rewards']))]
        plt.plot(validation_episodes, training_results['validation_rewards'], 'r-', label='Validation Reward', marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('DQN Learning Curve - Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot returns
    plt.subplot(2, 3, 2)
    plt.plot(np.array(training_results['episode_returns']) * 100, label='Training Return (%)', alpha=0.7)
    if training_results['validation_returns']:
        validation_episodes = [i*validation_frequency for i in range(len(training_results['validation_returns']))]
        plt.plot(validation_episodes, np.array(training_results['validation_returns']) * 100, 'r-', 
                label='Validation Return (%)', marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Return (%)')
    plt.title('Portfolio Returns During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot invalid actions
    plt.subplot(2, 3, 3)
    plt.plot(training_results['episode_invalid_actions'], label='Training Invalid Actions', alpha=0.7)
    if training_results['validation_invalid_actions']:
        validation_episodes = [i*validation_frequency for i in range(len(training_results['validation_invalid_actions']))]
        plt.plot(validation_episodes, training_results['validation_invalid_actions'], 'r-', 
                label='Validation Invalid Actions', marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Invalid Actions')
    plt.title('Invalid Action Attempts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot trade counts
    plt.subplot(2, 3, 4)
    plt.plot(training_results['episode_trades'], label='Training Trades', alpha=0.7)
    if training_results['validation_trades']:
        validation_episodes = [i*validation_frequency for i in range(len(training_results['validation_trades']))]
        plt.plot(validation_episodes, training_results['validation_trades'], 'r-', 
                label='Validation Trades', marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Number of Trades')
    plt.title('Trading Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot return distribution
    plt.subplot(2, 3, 5)
    plt.hist(np.array(training_results['episode_returns']) * 100, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.title('Return Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add multi-day test results summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    if 'multi_day_test_results' in training_results:
        aggregate_stats = training_results['multi_day_test_results']['aggregate_stats']
        summary_text = f"""Multi-Day Test Results:
        
Average Return: {aggregate_stats['avg_return']:.2%} ± {aggregate_stats['std_return']:.2%}
Best/Worst: {aggregate_stats['best_return']:.2%} / {aggregate_stats['worst_return']:.2%}
Win Rate: {aggregate_stats['win_rate']:.0%}
Avg Sharpe: {aggregate_stats['avg_sharpe_ratio']:.2f}
Avg Drawdown: {aggregate_stats['avg_max_drawdown']:.2%}
Avg Trades/Day: {aggregate_stats['avg_trades']:.1f}
Avg Invalid Actions: {aggregate_stats['avg_invalid_actions']:.1f}"""
    else:
        # Fallback for old single-day format
        test_results = training_results.get('test_results', {})
        summary_text = f"""Test Results Summary:
        
Total Return: {test_results.get('total_return', 0):.2%}
Sharpe Ratio: {test_results.get('sharpe_ratio', 0):.2f}
Max Drawdown: {test_results.get('max_drawdown', 0):.2%}
Total Trades: {test_results.get('total_trades', 0)}
Invalid Actions: {test_results.get('invalid_actions', 0)}
Final Value: ${test_results.get('final_value', 0):,.2f}"""
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{ticker} DQN Training Results', fontsize=16)
    plt.tight_layout()
    
    return plt


def plot_backtest_results(test_results: dict, ticker: str = "Stock", show_all_days: bool = True):
    """
    Visualize backtesting results - can show all days or just the best performing day
    
    Args:
        test_results: Results dictionary from train_dqn function
        ticker: Stock ticker symbol
        show_all_days: If True, show all days in detail. If False, show only best day.
    """
    # Handle multi-day results
    if 'multi_day_test_results' in test_results and show_all_days:
        return plot_all_backtest_days(test_results, ticker)
    elif 'multi_day_test_results' in test_results:
        multi_day_data = test_results['multi_day_test_results']
        individual_days = multi_day_data['individual_days']
        
        # Find the best performing day
        best_day_idx = max(range(len(individual_days)), key=lambda i: individual_days[i]['total_return'])
        best_day = individual_days[best_day_idx]
        
        portfolio_values = multi_day_data['portfolio_values'][best_day_idx]
        price_history = multi_day_data['price_histories'][best_day_idx]
        action_history = multi_day_data['action_histories'][best_day_idx]
        
        # Get invalid action mask for this day if available
        invalid_action_masks = multi_day_data.get('invalid_action_masks', None)
        invalid_mask = invalid_action_masks[best_day_idx] if invalid_action_masks and best_day_idx < len(invalid_action_masks) else None
        
        print(f"Plotting best performing day (Day {best_day_idx + 1}): {best_day['total_return']:.1%} return")
    else:
        # Handle single day results (legacy format)
        portfolio_values = test_results['portfolio_values']
        price_history = test_results['price_history']
        action_history = test_results['action_history']
        invalid_mask = None  # No invalid action filtering for legacy format
    
    plt.figure(figsize=(15, 10))
    
    # Plot stock price with buy/sell markers
    plt.subplot(2, 1, 1)
    plt.plot(price_history, label=f'{ticker} Price', linewidth=2, color='black', alpha=0.7)
    
    # Mark buy and sell actions (filter out invalid actions if mask is available)
    # Actions: 0=Hold, 1=Buy, 2=Sell
    if invalid_mask is not None:
        # Only include valid actions
        buy_indices = [i for i, (a, is_invalid) in enumerate(zip(action_history, invalid_mask)) if a == 1 and not is_invalid]
        sell_indices = [i for i, (a, is_invalid) in enumerate(zip(action_history, invalid_mask)) if a == 2 and not is_invalid]
    else:
        # Fallback: show all actions (for backward compatibility or single-day results)
        buy_indices = [i for i, a in enumerate(action_history) if a == 1]
        sell_indices = [i for i, a in enumerate(action_history) if a == 2]
    
    if buy_indices:
        plt.scatter(buy_indices, [price_history[i] for i in buy_indices], 
                   color='green', marker='^', s=100, label='Buy', zorder=5)
    if sell_indices:
        plt.scatter(sell_indices, [price_history[i] for i in sell_indices], 
                   color='red', marker='v', s=100, label='Sell', zorder=5)
    
    plt.xlabel('Trading Step')
    plt.ylabel('Price ($)')
    plt.title(f'{ticker} Price and Trading Actions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot portfolio value vs buy-and-hold
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_values, label='DQN Strategy', linewidth=2)
    
    # Calculate buy-and-hold strategy
    initial_balance = portfolio_values[0]
    initial_price = price_history[0]
    shares_bought = initial_balance / initial_price
    buy_hold_values = [initial_balance] + [shares_bought * price for price in price_history]
    plt.plot(buy_hold_values, '--', label='Buy & Hold Strategy', linewidth=2, alpha=0.7)
    
    # Add horizontal line for initial balance
    plt.axhline(y=initial_balance, color='gray', linestyle=':', alpha=0.5, label='Initial Balance')
    
    # Calculate and display performance difference
    dqn_return = (portfolio_values[-1] - initial_balance) / initial_balance * 100
    bh_return = (buy_hold_values[-1] - initial_balance) / initial_balance * 100
    outperformance = dqn_return - bh_return
    
    plt.xlabel('Trading Step')
    plt.ylabel('Portfolio Value ($)')
    plt.title(f'Portfolio Value Comparison (DQN: {dqn_return:.1f}%, B&H: {bh_return:.1f}%, Diff: {outperformance:+.1f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt


def plot_all_backtest_days(test_results: dict, ticker: str = "Stock"):
    """
    Plot detailed trading actions and performance for all backtest days
    
    Args:
        test_results: Results dictionary from train_dqn function
        ticker: Stock ticker symbol
        
    Returns:
        matplotlib figure object
    """
    multi_day_data = test_results['multi_day_test_results']
    individual_days = multi_day_data['individual_days']
    portfolio_values_list = multi_day_data['portfolio_values']
    price_histories = multi_day_data['price_histories']
    action_histories = multi_day_data['action_histories']
    
    # Get invalid action masks if available (for filtering)
    invalid_action_masks = multi_day_data.get('invalid_action_masks', None)
    
    num_days = len(individual_days)
    
    # Create a large figure with subplots for each day
    fig = plt.figure(figsize=(20, 4 * num_days))
    
    # Expanded color palette for multiple days
    import matplotlib.cm as cm
    
    if num_days <= 10:
        # Use distinct colors for smaller numbers
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    else:
        # Generate colors using colormap for larger numbers
        colormap = cm.get_cmap('tab20')  # Good for up to 20 distinct colors
        colors = [colormap(i / max(num_days - 1, 1)) for i in range(num_days)]
    
    print(f"📊 Plotting detailed results for all {num_days} backtest days...")
    
    for i, (day_info, portfolio_vals, price_history, action_history) in enumerate(
        zip(individual_days, portfolio_values_list, price_histories, action_histories)
    ):
        day_idx = day_info['day_idx']
        return_pct = day_info['total_return']
        color = colors[i % len(colors)]
        
        # Price and actions subplot
        ax1 = plt.subplot(num_days, 2, i * 2 + 1)
        ax1.plot(price_history, label=f'{ticker} Price', linewidth=2, color='black', alpha=0.7)
        
        # Mark buy and sell actions (filter out invalid actions if mask is available)
        if invalid_action_masks and i < len(invalid_action_masks):
            invalid_mask = invalid_action_masks[i]
            # Only include valid actions
            buy_indices = [j for j, (a, is_invalid) in enumerate(zip(action_history, invalid_mask)) if a == 1 and not is_invalid]
            sell_indices = [j for j, (a, is_invalid) in enumerate(zip(action_history, invalid_mask)) if a == 2 and not is_invalid]
        else:
            # Fallback: show all actions (for backward compatibility)
            buy_indices = [j for j, a in enumerate(action_history) if a == 1]
            sell_indices = [j for j, a in enumerate(action_history) if a == 2]
        
        if buy_indices:
            ax1.scatter(buy_indices, [price_history[j] for j in buy_indices], 
                       color='green', marker='^', s=60, label='Buy', zorder=5)
        if sell_indices:
            ax1.scatter(sell_indices, [price_history[j] for j in sell_indices], 
                       color='red', marker='v', s=60, label='Sell', zorder=5)
        
        ax1.set_title(f'Day {day_idx + 1} - {ticker} Price & Actions (Return: {return_pct:.1%})', 
                     fontweight='bold', color=color)
        ax1.set_xlabel('Trading Step')
        ax1.set_ylabel('Price ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Portfolio performance subplot
        ax2 = plt.subplot(num_days, 2, i * 2 + 2)
        ax2.plot(portfolio_vals, label='DQN Strategy', linewidth=2, color=color)
        
        # Calculate buy-and-hold strategy
        initial_balance = portfolio_vals[0]
        initial_price = price_history[0]
        shares_bought = initial_balance / initial_price
        buy_hold_values = [initial_balance] + [shares_bought * price for price in price_history]
        ax2.plot(buy_hold_values, '--', label='Buy & Hold', linewidth=2, alpha=0.7, color='gray')
        
        # Add horizontal line for initial balance
        ax2.axhline(y=initial_balance, color='gray', linestyle=':', alpha=0.5, label='Initial Balance')
        
        # Calculate performance metrics
        dqn_return = (portfolio_vals[-1] - initial_balance) / initial_balance * 100
        bh_return = (buy_hold_values[-1] - initial_balance) / initial_balance * 100
        outperformance = dqn_return - bh_return
        
        # Add performance stats as text
        stats_text = f"""Trades: {day_info['total_trades']} (W:{day_info['winning_trades']}, L:{day_info['losing_trades']})
DQN: {dqn_return:.1f}% | B&H: {bh_return:.1f}% | Diff: {outperformance:+.1f}%
Sharpe: {day_info['sharpe_ratio']:.2f} | Drawdown: {day_info['max_drawdown']:.1%}"""
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_title(f'Day {day_idx + 1} - Portfolio Performance', fontweight='bold', color=color)
        ax2.set_xlabel('Trading Step')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Format y-axis
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add overall title
    aggregate_stats = multi_day_data['aggregate_stats']
    fig.suptitle(
        f'{ticker} - All {num_days} Days Detailed Backtest Results\n'
        f'Avg Return: {aggregate_stats["avg_return"]:.1%} ± {aggregate_stats["std_return"]:.1%} | '
        f'Win Rate: {aggregate_stats["win_rate"]:.0%} | '
        f'Avg Trades/Day: {aggregate_stats["avg_trades"]:.1f}',
        fontsize=16, fontweight='bold', y=0.995
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)  # Make room for the suptitle
    
    print(f"✅ Generated detailed plots for all {num_days} trading days!")
    
    return fig


def plot_multi_day_comparison(training_results: dict, ticker: str = "Stock"):
    """
    Plot all days of backtesting on the same chart for comparison
    """
    if 'multi_day_test_results' not in training_results:
        print("No multi-day results available for comparison plot")
        return None
        
    multi_day_data = training_results['multi_day_test_results']
    individual_days = multi_day_data['individual_days']
    portfolio_values_list = multi_day_data['portfolio_values']
    
    num_days = len(individual_days)
    
    plt.figure(figsize=(15, 10))
    
    # Expanded color palette for multiple days
    import matplotlib.cm as cm
    
    if num_days <= 10:
        # Use distinct colors for smaller numbers
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    else:
        # Generate colors using colormap for larger numbers
        colormap = cm.get_cmap('tab20')  # Good for up to 20 distinct colors
        colors = [colormap(i / max(num_days - 1, 1)) for i in range(num_days)]
    
    # Plot 1: Portfolio Values
    plt.subplot(2, 1, 1)
    plt.title(f'{ticker} - Multi-Day Portfolio Performance Comparison', fontsize=16, fontweight='bold')
    
    initial_balance = portfolio_values_list[0][0] if portfolio_values_list else 10000
    
    for i, (portfolio_vals, day_info) in enumerate(zip(portfolio_values_list, individual_days)):
        day_num = i + 1
        return_pct = day_info['total_return']
        color = colors[i % len(colors)]
        
        # Create time axis (minutes within trading day)
        time_points = list(range(len(portfolio_vals)))
        
        # Plot portfolio value
        plt.plot(time_points, portfolio_vals, 
                label=f'Day {day_num} (Return: {return_pct:.1%})', 
                color=color, linewidth=2, alpha=0.8)
        
        # Add final value annotation for first 10 days only (to avoid clutter)
        if i < 10:
            final_val = portfolio_vals[-1]
            plt.annotate(f'${final_val:,.0f}', 
                        xy=(len(time_points)-1, final_val),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=9, color=color, fontweight='bold')
    
    # Add horizontal line for initial balance
    plt.axhline(y=initial_balance, color='black', linestyle='--', alpha=0.5, 
                label=f'Initial Balance (${initial_balance:,.0f})')
    
    plt.xlabel('Minutes into Trading Day')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: Normalized Returns (all starting at 100%)
    plt.subplot(2, 1, 2)
    plt.title('Normalized Returns Comparison (Starting at 100%)', fontsize=14, fontweight='bold')
    
    for i, (portfolio_vals, day_info) in enumerate(zip(portfolio_values_list, individual_days)):
        day_num = i + 1
        return_pct = day_info['total_return']
        color = colors[i % len(colors)]
        
        # Normalize to percentage returns starting at 100%
        if len(portfolio_vals) > 0:
            normalized_returns = [(val / portfolio_vals[0]) * 100 for val in portfolio_vals]
            time_points = list(range(len(normalized_returns)))
            
            plt.plot(time_points, normalized_returns, 
                    label=f'Day {day_num} (Final: {normalized_returns[-1]:.1f}%)', 
                    color=color, linewidth=2, alpha=0.8)
            
            # Add final percentage annotation
            final_pct = normalized_returns[-1]
            plt.annotate(f'{final_pct:.1f}%', 
                        xy=(len(time_points)-1, final_pct),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=9, color=color, fontweight='bold')
    
    # Add horizontal line at 100%
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Break-even (100%)')
    
    plt.xlabel('Minutes into Trading Day')
    plt.ylabel('Portfolio Value (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    # Add aggregate statistics as text box
    aggregate_stats = multi_day_data['aggregate_stats']
    stats_text = f"""Aggregate Statistics:
Average Return: {aggregate_stats['avg_return']:.1%} ± {aggregate_stats['std_return']:.1%}
Best Return: {aggregate_stats['best_return']:.1%}
Worst Return: {aggregate_stats['worst_return']:.1%}
Win Rate: {aggregate_stats['win_rate']:.0%}
Avg Trades/Day: {aggregate_stats['avg_trades']:.1f}
Avg Sharpe Ratio: {aggregate_stats['avg_sharpe_ratio']:.2f}"""
    
    plt.gca().text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return plt 