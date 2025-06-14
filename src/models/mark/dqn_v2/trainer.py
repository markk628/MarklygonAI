import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path

# from src.models.mark.dqn_v2.dqn_v5 import train_dqn
from src.models.mark.dqn_v2.dqn_v7 import train_enhanced_dqn
from src.models.mark.dqn_v2.database import save_multi_day_backtest_to_db
from src.models.mark.dqn_v2.visualization import plot_training_results, plot_backtest_results, plot_multi_day_comparison
from src.config.config import CUTOFF_TIMESTAMP, DATA_DIR, MODELS_DIR, RESULTS_DIR, EVALUATE_INTERVAL, INITIAL_BALANCE
from src.utils.utils import create_directory
from src.web.models import ModelType
from src.web.extensions import app
from src.web.models import BacktestHistory, db


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
    print(f"Starting Enhanced DQN training for {ticker}...")
    print("="*50)
    
    cutoff = pd.Timestamp(CUTOFF_TIMESTAMP, tz='UTC')
    
    # training_results = train_dqn(
    #     data_path=data_path,
    #     cutoff=cutoff,
    #     save_interval=50,
    #     early_stopping_patience=10,
    #     use_preprocessing=True,  # Enable preprocessing
    #     scaling_method='robust',  # Best for financial data
    #     outlier_method='winsorize',  # Handle outliers
    #     preprocessor_save_path=str(preprocessor_path)
    # )
    
    training_results = train_enhanced_dqn(
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
    # CRITICAL FIX: Include portfolio normalizer in final save
    portfolio_normalizer = training_results.get('portfolio_normalizer')
    
    training_results['agent'].save(model_path, portfolio_normalizer)
    if portfolio_normalizer and portfolio_normalizer.is_fitted:
        print(f"✅ Portfolio normalizer saved with final model")
    else:
        print(f"⚠️ No portfolio normalizer saved - model may not use portfolio normalization")
    
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