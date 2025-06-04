import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path

from src.models.mark.dqn_v2.dqn import train_dqn, save_backtest_results_to_db
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
    
    # Add test results summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    test_results = training_results['test_results']
    summary_text = f"""Test Results Summary:
    
Total Return: {test_results['total_return']:.2%}
Sharpe Ratio: {test_results['sharpe_ratio']:.2f}
Max Drawdown: {test_results['max_drawdown']:.2%}
Win Rate: {test_results['win_rate']:.2%}
Total Trades: {test_results['total_trades']}
Invalid Actions: {test_results['invalid_actions']}
Final Value: ${test_results['final_value']:,.2f}"""
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{ticker} DQN Training Results', fontsize=16)
    plt.tight_layout()
    
    return plt


def plot_backtest_results(test_results: dict, ticker: str = "Stock"):
    """
    Visualize backtesting results including price chart, portfolio value, and buy/sell actions
    Similar to trainer.py's plot_backtest_results
    """
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


def main():
    # Configuration
    ticker = 'TSLA'  # Change to your stock ticker
    data_path = f"{DATA_DIR}/feature_engineered/{ticker}.csv"
    
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
    backtest_plot = plot_backtest_results(training_results['test_results'], ticker=ticker)
    backtest_filename = results_dir / f'dqn_v2_{ticker}_{timestamp}_backtest.png'
    backtest_plot.savefig(backtest_filename, dpi=150, bbox_inches='tight')
    print(f"Backtest results saved to: {backtest_filename}")
    plt.close()
    
    # Print final summary
    test_results = training_results['test_results']
    print("\n" + "="*50)
    print("FINAL TEST RESULTS SUMMARY")
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
    
    # Save backtest results to database
    print("\nSaving results to database...")
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
        'max_drawdown': test_results['max_drawdown'],
        'sharpe_ratio': test_results['sharpe_ratio'],
        'invalid_actions': test_results['invalid_actions'],
    }
    
    # First save to database to get model ID and directory
    model_id, model_path, model_dir = save_backtest_results_to_db(ModelType.DQN, ticker, db_info)
    print(f"Model saved to database with ID: {model_id}")
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
        
        # Update the database with the preprocessor path
        with app.app_context():
            backtest = BacktestHistory.query.filter_by(model_id=model_id).first()
            if backtest:
                backtest.preprocessor_path = preprocessor_model_path
                db.session.commit()
                print(f"Updated database with preprocessor path")
        
        print(f"Preprocessing Configuration: {training_results['preprocessor'].get_preprocessing_info()}")
    
    # Clean up temporary preprocessor file if it exists
    if preprocessor_path.exists():
        preprocessor_path.unlink()
        print(f"Cleaned up temporary preprocessor file")
    
    print(f"\nAll results saved to: {model_dir}")
    print("Training complete!")


if __name__ == "__main__":
    main() 