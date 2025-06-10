import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path

# from src.models.mark.dqn_v2.dqn import train_dqn, save_backtest_results_to_db
# from src.models.mark.dqn_v2.dqn_v2 import train_dqn, save_backtest_results_to_db
# from src.models.mark.dqn_v2.dqn_v3 import train_dqn, save_backtest_results_to_db
# from src.models.mark.dqn_v2.dqn_v4 import train_dqn, save_backtest_results_to_db
from src.models.mark.dqn_v2.dqn_v5 import train_dqn, save_backtest_results_to_db, save_multi_day_backtest_to_db
from src.config.config import CUTOFF_TIMESTAMP, DATA_DIR, MODELS_DIR, RESULTS_DIR, EVALUATE_INTERVAL, INITIAL_BALANCE
from src.utils.utils import create_directory
from src.web.models import ModelType
from src.web.extensions import app
from src.web.models import BacktestHistory, db

def plot_training_results(training_results: dict, ticker: str = "Stock", validation_frequency: int = 5):
    """
    Visualize training and performance metrics similar to trainer.py
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


def plot_backtest_results(test_results: dict, ticker: str = "Stock"):
    """
    Visualize backtesting results - handles both single day and multi-day results
    For multi-day results, shows the best performing day
    """
    # Handle multi-day results
    if 'multi_day_test_results' in test_results:
        multi_day_data = test_results['multi_day_test_results']
        individual_days = multi_day_data['individual_days']
        
        # Find the best performing day
        best_day_idx = max(range(len(individual_days)), key=lambda i: individual_days[i]['total_return'])
        best_day = individual_days[best_day_idx]
        
        portfolio_values = multi_day_data['portfolio_values'][best_day_idx]
        price_history = multi_day_data['price_histories'][best_day_idx]
        action_history = multi_day_data['action_histories'][best_day_idx]
        
        print(f"Plotting best performing day (Day {best_day_idx + 1}): {best_day['total_return']:.1%} return")
    else:
        # Handle single day results (legacy format)
        portfolio_values = test_results['portfolio_values']
        price_history = test_results['price_history']
        action_history = test_results['action_history']
    
    plt.figure(figsize=(15, 10))
    
    # Plot stock price with buy/sell markers
    plt.subplot(2, 1, 1)
    plt.plot(price_history, label=f'{ticker} Price', linewidth=2, color='black', alpha=0.7)
    
    # Mark buy and sell actions
    # Actions: 0=Hold, 1=Buy, 2=Sell
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


def plot_multi_day_comparison(training_results: dict, ticker: str = "Stock"):
    """
    Plot all 6 days of backtesting on the same chart for comparison
    """
    if 'multi_day_test_results' not in training_results:
        print("No multi-day results available for comparison plot")
        return None
        
    multi_day_data = training_results['multi_day_test_results']
    individual_days = multi_day_data['individual_days']
    portfolio_values_list = multi_day_data['portfolio_values']
    
    plt.figure(figsize=(15, 10))
    
    # Color palette for different days
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
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
        
        # Add final value annotation
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


def main():
    # Configuration
    ticker = 'TSLA'  # Change to your stock ticker
    # data_path = f"{DATA_DIR}/feature_engineered/{ticker}.csv"
    data_path = f"{DATA_DIR}/feature_engineered_v2/{ticker}.csv"
    
    # Create results directory if it doesn't exist
    models_dir = MODELS_DIR / 'dqn_v2'
    results_dir = RESULTS_DIR / 'dqn_v2'
    create_directory(models_dir)
    create_directory(results_dir)
    
    # Preprocessor save path
    preprocessor_path = models_dir / f'preprocessor_{ticker}.pkl'
    
    # Train the agent with validation
    print(f"Starting DQN training for {ticker}...")
    print("="*50)
    
    cutoff = pd.Timestamp(CUTOFF_TIMESTAMP, tz='UTC')
    
    training_results = train_dqn(
        data_path=data_path,
        cutoff=cutoff,
        save_interval=50,
        early_stopping_patience=10,
        use_preprocessing=True,  # Enable preprocessing
        scaling_method='robust',  # Best for financial data
        outlier_method='winsorize',  # Handle outliers
        preprocessor_save_path=str(preprocessor_path)
    )
    
    # Get start and end dates from training results
    start_date = training_results['start_date']
    end_date = training_results['end_date']
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot and save training results
    print("\nGenerating training plots...")
    training_plot = plot_training_results(training_results, ticker=ticker, validation_frequency=EVALUATE_INTERVAL)
    training_filename = results_dir / f'dqn_v2_{ticker}_{timestamp}_training.png'
    training_plot.savefig(training_filename, dpi=150, bbox_inches='tight')
    print(f"Training results saved to: {training_filename}")
    plt.close()
    
    # Plot and save backtest results
    print("\nGenerating backtest plots...")
    backtest_plot = plot_backtest_results(training_results, ticker=ticker)
    backtest_filename = results_dir / f'dqn_v2_{ticker}_{timestamp}_backtest.png'
    backtest_plot.savefig(backtest_filename, dpi=150, bbox_inches='tight')
    print(f"Backtest results saved to: {backtest_filename}")
    plt.close()
    
    # Plot and save multi-day comparison
    print("\nGenerating multi-day comparison plots...")
    multiday_plot = plot_multi_day_comparison(training_results, ticker=ticker)
    if multiday_plot is not None:
        multiday_filename = results_dir / f'dqn_v2_{ticker}_{timestamp}_multiday.png'
        multiday_plot.savefig(multiday_filename, dpi=150, bbox_inches='tight')
        print(f"Multi-day comparison saved to: {multiday_filename}")
        plt.close()
    else:
        print("Multi-day comparison plot skipped (no multi-day data available)")
    
    # Print final summary
    multi_day_results = training_results['multi_day_test_results']
    aggregate_stats = multi_day_results['aggregate_stats']
    individual_days = multi_day_results['individual_days']
    
    print("\n" + "="*50)
    print("MULTI-DAY TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Ticker: {ticker}")
    print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"Number of Test Days: {len(individual_days)}")
    print(f"Average Return: {aggregate_stats['avg_return']:.2%} ± {aggregate_stats['std_return']:.2%}")
    print(f"Best Day Return: {aggregate_stats['best_return']:.2%}")
    print(f"Worst Day Return: {aggregate_stats['worst_return']:.2%}")
    print(f"Win Rate: {aggregate_stats['win_rate']:.1%}")
    print(f"Average Sharpe Ratio: {aggregate_stats['avg_sharpe_ratio']:.2f}")
    print(f"Average Max Drawdown: {aggregate_stats['avg_max_drawdown']:.2%}")
    print(f"Average Trades per Day: {aggregate_stats['avg_trades']:.1f}")
    print(f"Average Invalid Actions: {aggregate_stats['avg_invalid_actions']:.1f}")
    
    print(f"\nIndividual Day Results:")
    for i, day_result in enumerate(individual_days, 1):
        print(f"  Day {i}: {day_result['total_return']:.1%} return, "
              f"{day_result['total_trades']} trades, "
              f"{day_result['invalid_actions']} invalid actions")
    print("="*50)
    
    # Save multi-day backtest results to database
    print("\nSaving multi-day results to database...")
    
    # Save one model with multiple backtest entries
    model_id, model_path, model_dir, backtest_ids = save_multi_day_backtest_to_db(
        model_type=ModelType.DQN,
        ticker=ticker,
        multi_day_results=multi_day_results,
        start_date=start_date,
        end_date=end_date,
        initial_balance=INITIAL_BALANCE,
        preprocessor_path=None  # Will be updated later
    )
    
    print(f"✅ Model created with ID: {model_id}")
    print(f"✅ Created {len(backtest_ids)} backtest entries:")
    print(f"   • Aggregate summary (backtest ID: {backtest_ids[0]})")
    for i, backtest_id in enumerate(backtest_ids[1:], 1):
        day_return = individual_days[i-1]['total_return']
        print(f"   • Day {i} (backtest ID: {backtest_id}): {day_return:.1%} return")
    
    print(f"All results linked to model ID: {model_id}")
    
    # Ensure model directory exists
    create_directory(model_dir)
    print(f"Model directory: {model_dir}")
    
    # Save the trained model
    print(f"\nSaving trained model to: {model_path}")
    training_results['agent'].save(model_path)
    
    # Save the preprocessor in the same directory
    if training_results.get('preprocessor') is not None:
        preprocessor_model_path = str(Path(model_dir) / 'preprocessor.pkl')
        print(f"\nSaving preprocessor to: {preprocessor_model_path}")
        
        # Copy the preprocessor from temporary location to model directory
        import shutil
        shutil.copy2(str(preprocessor_path), preprocessor_model_path)
        
        # Update the database with the preprocessor path for all backtest entries
        with app.app_context():
            backtests = BacktestHistory.query.filter_by(model_id=model_id).all()
            for backtest in backtests:
                backtest.preprocessor_path = preprocessor_model_path
            db.session.commit()
            print(f"Updated {len(backtests)} backtest entries with preprocessor path")
        
        print(f"Preprocessing Configuration: {training_results['preprocessor'].get_preprocessing_info()}")
    
    # Clean up temporary preprocessor file if it exists
    if preprocessor_path.exists():
        preprocessor_path.unlink()
        print(f"Cleaned up temporary preprocessor file")
    
    print(f"\nAll results saved to: {model_dir}")
    print("Training complete!")


if __name__ == "__main__":
    main() 