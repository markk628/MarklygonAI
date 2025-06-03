import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path

# Import both discrete and continuous SAC components
from src.models.sugarmixcoffee.sac.sac import train_sac, save_backtest_results_to_db, SACConfig
from src.models.sugarmixcoffee.sac.continuous_sac_agent import train_continuous_sac, ContinuousSACConfig
from src.config.config import DATA_DIR, MODELS_DIR, RESULTS_DIR, EVALUATE_INTERVAL, INITIAL_BALANCE
from src.utils.utils import create_directory
from src.web.models import ModelType
from src.web.extensions import app
from src.web.models import BacktestHistory, db


def plot_sac_training_results(training_results: dict, ticker: str = "Stock", validation_frequency: int = 5, is_continuous: bool = False):
    """
    Visualize SAC training and performance metrics (works for both discrete and continuous SAC)
    """
    plt.figure(figsize=(15, 12))
    
    sac_type = "Continuous SAC" if is_continuous else "Discrete SAC"
    
    # Plot training rewards
    plt.subplot(2, 3, 1)
    plt.plot(training_results['episode_rewards'], label='Training Reward', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title(f'{sac_type} Learning Curve - Rewards')
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
    
    # Add fees info for continuous SAC
    fees_text = ""
    if is_continuous and 'total_fees_paid' in test_results:
        fees_text = f"\nTotal Fees: ${test_results['total_fees_paid']:.2f}"
    
    summary_text = f"""{sac_type} Test Results:
    
Total Return: {test_results['total_return']:.2%}
Sharpe Ratio: {test_results['sharpe_ratio']:.2f}
Max Drawdown: {test_results['max_drawdown']:.2%}
Win Rate: {test_results['win_rate']:.2%}
Total Trades: {test_results['total_trades']}
Invalid Actions: {test_results['invalid_actions']}{fees_text}
Final Value: ${test_results['final_value']:,.2f}"""
    
    color = 'lightblue' if is_continuous else 'wheat'
    plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    plt.suptitle(f'{ticker} {sac_type} Training Results', fontsize=16)
    plt.tight_layout()
    
    return plt


def plot_sac_backtest_results(test_results: dict, ticker: str = "Stock", is_continuous: bool = False):
    """
    Visualize SAC backtesting results (works for both discrete and continuous SAC)
    """
    portfolio_values = test_results['portfolio_values']
    price_history = test_results['price_history']
    
    sac_type = "Continuous SAC" if is_continuous else "Discrete SAC"
    
    # Determine how to handle actions based on SAC type
    if is_continuous:
        # For continuous SAC, we have action values and position history
        action_history = test_results.get('action_history', [])
        position_history = test_results.get('position_history', [])
        
        plt.figure(figsize=(15, 12))
        
        # Stock price
        plt.subplot(2, 2, 1)
        plt.plot(price_history, label=f'{ticker} Price', linewidth=2, color='black', alpha=0.7)
        plt.xlabel('Trading Step')
        plt.ylabel('Price ($)')
        plt.title(f'{ticker} Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Continuous actions
        plt.subplot(2, 2, 2)
        if action_history:
            plt.plot(action_history, label='Action Values', color='blue', alpha=0.7)
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Strong Buy')
            plt.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5, label='Strong Sell')
            plt.ylabel('Action Value')
            plt.title('Continuous Actions (-1=Sell All, 0=Hold, +1=Buy Max)')
            plt.ylim(-1.1, 1.1)
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Position sizing
        plt.subplot(2, 2, 3)
        if position_history:
            plt.plot([p * 100 for p in position_history], label='Position %', color='orange', linewidth=2)
            plt.ylabel('Position (%)')
            plt.title('Position Sizing Over Time')
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Portfolio value vs buy-and-hold
        plt.subplot(2, 2, 4)
        
    else:
        # For discrete SAC, we have discrete action history
        action_history = test_results.get('action_history', [])
        
        plt.figure(figsize=(15, 10))
        
        # Stock price with buy/sell markers
        plt.subplot(2, 1, 1)
        plt.plot(price_history, label=f'{ticker} Price', linewidth=2, color='black', alpha=0.7)
        
        # Mark buy and sell actions for discrete SAC
        if action_history:
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
        plt.title(f'{ticker} Price and {sac_type} Trading Actions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Portfolio value vs buy-and-hold
        plt.subplot(2, 1, 2)
    
    # Common portfolio comparison plot for both types
    strategy_label = f'{sac_type} Strategy'
    plt.plot(portfolio_values, label=strategy_label, linewidth=2, color='blue')
    
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
    plt.title(f'Portfolio Value Comparison ({sac_type}: {sac_return:.1f}%, B&H: {bh_return:.1f}%, Diff: {outperformance:+.1f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt


def compare_strategies(sac_results: dict, dqn_results: dict = None, ticker: str = "Stock", is_continuous: bool = False):
    """
    Compare SAC with other strategies if available
    """
    plt.figure(figsize=(12, 8))
    
    sac_portfolio = sac_results['portfolio_values']
    price_history = sac_results['price_history']
    
    sac_label = "Continuous SAC" if is_continuous else "Discrete SAC"
    
    # Plot portfolio values
    plt.subplot(2, 1, 1)
    plt.plot(sac_portfolio, label=f'{sac_label} Strategy', linewidth=2, color='blue')
    
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
    
    plt.plot(sac_returns, label=f'{sac_label} Returns', alpha=0.7, color='blue')
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


def main(use_continuous_sac: bool = False):
    """
    Main training function with option to use continuous or discrete SAC
    
    Args:
        use_continuous_sac: If True, use continuous SAC with position sizing.
                           If False, use discrete SAC with binary actions.
    """
    # Configuration
    ticker = 'NVDA'
    data_path = f"{DATA_DIR}/feature_engineered/{ticker}.csv"
    
    # Create directories
    sac_type_folder = 'continuous_sac' if use_continuous_sac else 'sac'
    models_dir = MODELS_DIR / sac_type_folder
    results_dir = RESULTS_DIR / sac_type_folder
    create_directory(models_dir)
    create_directory(results_dir)
    
    # Preprocessor save path
    preprocessor_prefix = 'continuous_sac' if use_continuous_sac else 'sac'
    preprocessor_path = models_dir / f'preprocessor_{preprocessor_prefix}_{ticker}.pkl'
    
    sac_type_name = "Continuous SAC" if use_continuous_sac else "Discrete SAC"
    print(f"Starting {sac_type_name} training for {ticker}...")
    print("="*50)
    
    if use_continuous_sac:
        print("🔥 Using Continuous SAC with Position Sizing")
        print("   → Actions: [-1, 1] for granular position control")
        print("   → Benefit: Sophisticated risk management and capital utilization")
    else:
        print("⚡ Using Discrete SAC with Binary Actions")
        print("   → Actions: [Hold, Buy, Sell] for simple trading decisions")
        print("   → Benefit: Straightforward and interpretable actions")
    
    print()
    
    cutoff = pd.Timestamp('2025-05-05 08:00:00', tz='UTC')
    
    # Train SAC agent (discrete or continuous)
    if use_continuous_sac:
        training_results = train_continuous_sac(
            data_path=data_path,
            cutoff=cutoff,
            num_episodes=100,
            save_interval=50,
            early_stopping_patience=15,
            use_preprocessing=True,
            scaling_method='robust',
            outlier_method='winsorize',
            preprocessor_save_path=str(preprocessor_path)
        )
    else:
        training_results = train_sac(
            data_path=data_path,
            cutoff=cutoff,
            num_episodes=100,
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
    print(f"\nGenerating {sac_type_name} training plots...")
    training_plot = plot_sac_training_results(training_results, ticker=ticker, 
                                            validation_frequency=EVALUATE_INTERVAL, 
                                            is_continuous=use_continuous_sac)
    training_filename = results_dir / f'{sac_type_folder}_{ticker}_{timestamp}_training.png'
    training_plot.savefig(training_filename, dpi=150, bbox_inches='tight')
    print(f"Training results saved to: {training_filename}")
    plt.close()
    
    # Plot and save backtest results
    print(f"\nGenerating {sac_type_name} backtest plots...")
    backtest_plot = plot_sac_backtest_results(training_results['test_results'], ticker=ticker, 
                                            is_continuous=use_continuous_sac)
    backtest_filename = results_dir / f'{sac_type_folder}_{ticker}_{timestamp}_backtest.png'
    backtest_plot.savefig(backtest_filename, dpi=150, bbox_inches='tight')
    print(f"Backtest results saved to: {backtest_filename}")
    plt.close()
    
    # Print summary
    test_results = training_results['test_results']
    print("\n" + "="*50)
    print(f"{sac_type_name.upper()} TEST RESULTS SUMMARY")
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
    
    # Additional info for continuous SAC
    if use_continuous_sac and 'total_fees_paid' in test_results:
        print(f"Total Fees Paid: ${test_results['total_fees_paid']:.2f}")
        fee_impact = test_results['total_fees_paid'] / INITIAL_BALANCE * 100
        print(f"Fee Impact: {fee_impact:.2%} of initial capital")
        
        # Position sizing analysis
        if 'action_history' in test_results and 'position_history' in test_results:
            action_history = test_results['action_history']
            position_history = test_results['position_history']
            
            if action_history and position_history:
                print(f"\nPosition Sizing Analysis:")
                print(f"  Action Range: {min(action_history):.3f} to {max(action_history):.3f}")
                print(f"  Position Range: {min(position_history):.1%} to {max(position_history):.1%}")
                print(f"  Average Position: {np.mean(position_history):.1%}")
    
    print("="*50)
    
    # Save to database
    print(f"\nSaving {sac_type_name} results to database...")
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
    print(f"{sac_type_name} model saved to database with ID: {model_id}")
    print(f"Model directory: {model_dir}")
    
    # Save the trained model
    print(f"\nSaving trained {sac_type_name} model to: {model_path}")
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
    
    print(f"\nAll {sac_type_name} results saved to: {model_dir}")
    print(f"{sac_type_name} training complete!")
    
    return training_results


if __name__ == "__main__":
    import sys
    
    # Check command line arguments for SAC type
    use_continuous = False
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['continuous', 'cont', 'c', 'true']:
            use_continuous = True
        elif sys.argv[1].lower() in ['discrete', 'disc', 'd', 'false']:
            use_continuous = False
        else:
            print("Usage: python trainer.py [continuous|discrete]")
            print("  continuous: Use Continuous SAC with position sizing")
            print("  discrete:   Use Discrete SAC with binary actions (default)")
            sys.exit(1)
    
    print("SAC Trainer - Choose Your Action Space!")
    print("="*45)
    
    if use_continuous:
        print("🎯 Selected: Continuous SAC")
        print("   Perfect for: Sophisticated position management")
    else:
        print("⚡ Selected: Discrete SAC")
        print("   Perfect for: Simple and interpretable trading")
    
    print()
    
    main(use_continuous_sac=use_continuous) 