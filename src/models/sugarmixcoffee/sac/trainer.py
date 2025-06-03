import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path

from src.models.sugarmixcoffee.sac.sac import train_sac, save_backtest_results_to_db, SACConfig
from src.config.config import DATA_DIR, MODELS_DIR, RESULTS_DIR, EVALUATE_INTERVAL, INITIAL_BALANCE
from src.utils.utils import create_directory
from src.web.models import ModelType
from src.web.extensions import app
from src.web.models import BacktestHistory, db


def plot_sac_training_results(training_results: dict, ticker: str = "Stock", validation_frequency: int = 5):
    """
    Visualize SAC training and performance metrics
    """
    plt.figure(figsize=(15, 12))
    
    # Plot training rewards
    plt.subplot(2, 3, 1)
    plt.plot(training_results['episode_rewards'], label='Training Reward', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('SAC Learning Curve - Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot returns
    plt.subplot(2, 3, 2)
    plt.plot(np.array(training_results['episode_returns']) * 100, label='Training Return (%)', alpha=0.7)
    if training_results.get('validation_returns'):
        validation_episodes = [i*validation_frequency for i in range(len(training_results['validation_returns']))]
        plt.plot(validation_episodes, np.array(training_results['validation_returns']) * 100, 'r-', 
                label='Validation Return (%)', marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Return (%)')
    plt.title('Portfolio Returns During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot return distribution
    plt.subplot(2, 3, 3)
    plt.hist(np.array(training_results['episode_returns']) * 100, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.title('Return Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot moving average of returns
    plt.subplot(2, 3, 4)
    returns = np.array(training_results['episode_returns']) * 100
    window = min(20, len(returns) // 4)
    if window > 0:
        moving_avg = pd.Series(returns).rolling(window=window).mean()
        plt.plot(moving_avg, label=f'{window}-Episode Moving Average', linewidth=2)
        plt.plot(returns, alpha=0.3, label='Raw Returns')
    else:
        plt.plot(returns, label='Returns')
    plt.xlabel('Episode')
    plt.ylabel('Return (%)')
    plt.title('Return Trend Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Performance comparison with benchmark
    plt.subplot(2, 3, 5)
    if len(training_results['episode_returns']) > 10:
        recent_returns = training_results['episode_returns'][-10:]
        avg_recent = np.mean(recent_returns) * 100
        plt.bar(['Last 10 Episodes', 'Overall Average'], 
               [avg_recent, np.mean(training_results['episode_returns']) * 100])
        plt.ylabel('Average Return (%)')
        plt.title('Recent vs Overall Performance')
    plt.grid(True, alpha=0.3)
    
    # Test results summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    test_results = training_results['test_results']
    summary_text = f"""SAC Test Results:
    
Total Return: {test_results['total_return']:.2%}
Sharpe Ratio: {test_results['sharpe_ratio']:.2f}
Max Drawdown: {test_results['max_drawdown']:.2%}
Win Rate: {test_results['win_rate']:.2%}
Total Trades: {test_results['total_trades']}
Invalid Actions: {test_results['invalid_actions']}
Final Value: ${test_results['final_value']:,.2f}"""
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle(f'{ticker} SAC Training Results', fontsize=16)
    plt.tight_layout()
    
    return plt


def plot_sac_backtest_results(test_results: dict, ticker: str = "Stock"):
    """
    Visualize SAC backtesting results
    """
    portfolio_values = test_results['portfolio_values']
    price_history = test_results['price_history']
    action_history = test_results['action_history']
    
    plt.figure(figsize=(15, 10))
    
    # Plot stock price with buy/sell markers
    plt.subplot(2, 1, 1)
    plt.plot(price_history, label=f'{ticker} Price', linewidth=2, color='black', alpha=0.7)
    
    # Mark buy and sell actions
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
    plt.title(f'{ticker} Price and SAC Trading Actions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot portfolio value vs buy-and-hold
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_values, label='SAC Strategy', linewidth=2, color='blue')
    
    # Calculate buy-and-hold strategy
    initial_balance = portfolio_values[0]
    initial_price = price_history[0]
    shares_bought = initial_balance / initial_price
    buy_hold_values = [initial_balance] + [shares_bought * price for price in price_history]
    plt.plot(buy_hold_values, '--', label='Buy & Hold Strategy', linewidth=2, alpha=0.7, color='orange')
    
    # Add horizontal line for initial balance
    plt.axhline(y=initial_balance, color='gray', linestyle=':', alpha=0.5, label='Initial Balance')
    
    # Calculate performance difference
    sac_return = (portfolio_values[-1] - initial_balance) / initial_balance * 100
    bh_return = (buy_hold_values[-1] - initial_balance) / initial_balance * 100
    outperformance = sac_return - bh_return
    
    plt.xlabel('Trading Step')
    plt.ylabel('Portfolio Value ($)')
    plt.title(f'Portfolio Value Comparison (SAC: {sac_return:.1f}%, B&H: {bh_return:.1f}%, Diff: {outperformance:+.1f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt


def compare_strategies(sac_results: dict, dqn_results: dict = None, ticker: str = "Stock"):
    """
    Compare SAC with other strategies if available
    """
    plt.figure(figsize=(12, 8))
    
    sac_portfolio = sac_results['portfolio_values']
    price_history = sac_results['price_history']
    
    # Plot portfolio values
    plt.subplot(2, 1, 1)
    plt.plot(sac_portfolio, label='SAC Strategy', linewidth=2, color='blue')
    
    if dqn_results:
        dqn_portfolio = dqn_results.get('portfolio_values', [])
        if dqn_portfolio:
            plt.plot(dqn_portfolio, label='DQN Strategy', linewidth=2, color='red', alpha=0.8)
    
    # Buy and hold
    initial_balance = sac_portfolio[0]
    initial_price = price_history[0]
    shares_bought = initial_balance / initial_price
    buy_hold_values = [initial_balance] + [shares_bought * price for price in price_history]
    plt.plot(buy_hold_values, '--', label='Buy & Hold', linewidth=2, alpha=0.7, color='gray')
    
    plt.xlabel('Trading Step')
    plt.ylabel('Portfolio Value ($)')
    plt.title(f'{ticker} Strategy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Returns comparison
    plt.subplot(2, 1, 2)
    sac_returns = np.diff(sac_portfolio) / sac_portfolio[:-1] * 100
    bh_returns = np.diff(buy_hold_values) / buy_hold_values[:-1] * 100
    
    plt.plot(sac_returns, label='SAC Returns', alpha=0.7, color='blue')
    plt.plot(bh_returns, label='Buy & Hold Returns', alpha=0.7, color='gray')
    
    if dqn_results and 'portfolio_values' in dqn_results:
        dqn_portfolio = dqn_results['portfolio_values']
        if len(dqn_portfolio) > 1:
            dqn_returns = np.diff(dqn_portfolio) / dqn_portfolio[:-1] * 100
            plt.plot(dqn_returns, label='DQN Returns', alpha=0.7, color='red')
    
    plt.xlabel('Trading Step')
    plt.ylabel('Returns (%)')
    plt.title('Daily Returns Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt


def main():
    # Configuration
    ticker = 'NVDA'
    data_path = f"{DATA_DIR}/feature_engineered/{ticker}.csv"
    
    # Create results directory
    models_dir = MODELS_DIR / 'sac'
    results_dir = RESULTS_DIR / 'sac'
    create_directory(models_dir)
    create_directory(results_dir)
    
    # Preprocessor save path
    preprocessor_path = models_dir / f'preprocessor_sac_{ticker}.pkl'
    
    print(f"Starting SAC training for {ticker}...")
    print("="*50)
    
    cutoff = pd.Timestamp('2024-05-06 08:00:00', tz='UTC')
    
    # Train SAC agent
    training_results = train_sac(
        data_path=data_path,
        cutoff=cutoff,
        num_episodes=100,  # SAC typically needs more episodes
        save_interval=50,
        early_stopping_patience=15,
        use_preprocessing=True,
        scaling_method='robust',
        outlier_method='winsorize',
        preprocessor_save_path=str(preprocessor_path)
    )
    
    # Get dates
    start_date = training_results['start_date']
    end_date = training_results['end_date']
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot and save training results
    print("\nGenerating SAC training plots...")
    training_plot = plot_sac_training_results(training_results, ticker=ticker, validation_frequency=EVALUATE_INTERVAL)
    training_filename = results_dir / f'sac_{ticker}_{timestamp}_training.png'
    training_plot.savefig(training_filename, dpi=150, bbox_inches='tight')
    print(f"Training results saved to: {training_filename}")
    plt.close()
    
    # Plot and save backtest results
    print("\nGenerating SAC backtest plots...")
    backtest_plot = plot_sac_backtest_results(training_results['test_results'], ticker=ticker)
    backtest_filename = results_dir / f'sac_{ticker}_{timestamp}_backtest.png'
    backtest_plot.savefig(backtest_filename, dpi=150, bbox_inches='tight')
    print(f"Backtest results saved to: {backtest_filename}")
    plt.close()
    
    # Print summary
    test_results = training_results['test_results']
    print("\n" + "="*50)
    print("SAC TEST RESULTS SUMMARY")
    print("="*50)
    print(f"Ticker: {ticker}")
    print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"Final Value: ${test_results['final_value']:,.2f}")
    print(f"Total Return: {test_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {test_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {test_results['max_drawdown']:.2%}")
    print(f"Total Trades: {test_results['total_trades']}")
    print(f"Winning Trades: {test_results['winning_trades']}")
    print(f"Losing Trades: {test_results['losing_trades']}")
    print(f"Win Rate: {test_results['win_rate']:.2%}")
    print(f"Invalid Actions: {test_results['invalid_actions']}")
    print("="*50)
    
    # Save to database
    print("\nSaving SAC results to database...")
    db_info = {
        'backtest_date': datetime.now(timezone.utc),
        'start_date': start_date,
        'end_date': end_date,
        'initial_balance': INITIAL_BALANCE,
        'final_balance': test_results['final_value'],
        'net_profit': test_results['final_value'] - INITIAL_BALANCE,
        'total_trades': test_results['total_trades'],
        'winning_trades': test_results['winning_trades'],
        'losing_trades': test_results['losing_trades'],
        'return_rate': test_results['total_return'],
        'max_drawdown': abs(test_results['max_drawdown']),
        'sharpe_ratio': test_results['sharpe_ratio'],
        'invalid_actions': test_results['invalid_actions'],
    }
    
    # Save to database
    model_id, model_path, model_dir = save_backtest_results_to_db(ModelType.SAC, ticker, db_info)
    print(f"SAC model saved to database with ID: {model_id}")
    print(f"Model directory: {model_dir}")
    
    # Save the trained model
    print(f"\nSaving trained SAC model to: {model_path}")
    training_results['agent'].save(model_path)
    
    # Save preprocessor
    if training_results.get('preprocessor') is not None:
        preprocessor_model_path = str(Path(model_dir) / 'preprocessor.pkl')
        print(f"\nSaving preprocessor to: {preprocessor_model_path}")
        
        import shutil
        shutil.copy2(str(preprocessor_path), preprocessor_model_path)
        
        # Update database
        with app.app_context():
            backtest = BacktestHistory.query.filter_by(model_id=model_id).first()
            if backtest:
                backtest.preprocessor_path = preprocessor_model_path
                db.session.commit()
                print(f"Updated database with preprocessor path")
        
        print(f"Preprocessing Configuration: {training_results['preprocessor'].get_preprocessing_info()}")
    
    # Clean up
    if preprocessor_path.exists():
        preprocessor_path.unlink()
        print(f"Cleaned up temporary preprocessor file")
    
    print(f"\nAll SAC results saved to: {model_dir}")
    print("SAC training complete!")


if __name__ == "__main__":
    main() 