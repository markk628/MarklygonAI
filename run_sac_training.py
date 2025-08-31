#!/usr/bin/env python3
"""
SAC Training Entry Point
=======================

Quick script to run SAC training with default settings.
Equivalent to running the SAC trainer directly.

Usage:
    python run_sac_training.py
    
    # Or with custom ticker
    python run_sac_training.py --ticker AAPL
    
    # Or with more episodes
    python run_sac_training.py --episodes 1000
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.models.jeawan.sac.trainer import main as sac_trainer_main
except ImportError as e:
    print(f"❌ Error importing SAC trainer: {e}")
    print("Make sure you're in the correct directory and have all dependencies installed.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Run SAC Trading Agent Training')
    parser.add_argument('--ticker', default='TSLA', 
                       help='Stock ticker to train on (default: TSLA)')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes (default: 500)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    
    args = parser.parse_args()
    
    print("🚀 Starting SAC Training")
    print("=" * 50)
    print(f"Ticker: {args.ticker}")
    print(f"Episodes: {args.episodes}")
    print(f"Early Stopping Patience: {args.patience}")
    print("=" * 50)
    
    # Temporarily modify the trainer settings
    # (This is a simple approach - in production you'd want a config system)
    import src.models.jeawan.sac.trainer as trainer_module
    
    # Store original values
    original_ticker = getattr(trainer_module, 'ticker', 'TSLA')
    
    # Update global settings in the trainer module
    # Note: This modifies the main() function's defaults
    def custom_main():
        # Configuration
        ticker = args.ticker
        data_path = f"data/feature_engineered_v2/{ticker}.csv"
        
        # Create results directory if it doesn't exist
        from src.config.config import CUTOFF_TIMESTAMP, DATA_DIR, MODELS_DIR, RESULTS_DIR, EVALUATE_INTERVAL, INITIAL_BALANCE
        from src.utils.utils import create_directory
        import pandas as pd
        
        models_dir = MODELS_DIR / 'sac'
        results_dir = RESULTS_DIR / 'sac'
        create_directory(models_dir)
        create_directory(results_dir)
        
        # Preprocessor save path
        preprocessor_path = models_dir / f'preprocessor_{ticker}.pkl'
        
        # Train the SAC agent with validation
        print(f"Starting SAC training for {ticker}...")
        print("="*50)
        
        cutoff = pd.Timestamp(CUTOFF_TIMESTAMP, tz='UTC')
        
        from src.models.jeawan.sac.sac import train_sac
        
        training_results = train_sac(
            data_path=data_path,
            cutoff=cutoff,
            num_episodes=args.episodes,
            save_interval=50,
            validation_frequency=EVALUATE_INTERVAL,
            early_stopping_patience=args.patience,
            use_preprocessing=True,
            scaling_method='robust',
            outlier_method='winsorize',
            preprocessor_save_path=str(preprocessor_path)
        )
        
        # Continue with the rest of the original main() function
        # Import the rest of the trainer functionality
        from datetime import datetime
        from pathlib import Path
        import matplotlib.pyplot as plt
        
        from src.models.mark.dqn_v2.database import save_multi_day_backtest_to_db
        from src.models.jeawan.sac.visualization import plot_sac_training_results, plot_sac_backtest_results, plot_sac_multi_day_comparison
        from src.web.models import ModelType
        from src.web.extensions import app
        from src.web.models import BacktestHistory, db
        
        # Get start and end dates from training results
        start_date = training_results['start_date']
        end_date = training_results['end_date']
        
        # Generate timestamp for file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Plot and save training results
        print("\nGenerating SAC training plots...")
        training_plot = plot_sac_training_results(training_results, ticker=ticker, validation_frequency=EVALUATE_INTERVAL)
        training_filename = results_dir / f'sac_{ticker}_{timestamp}_training.png'
        training_plot.savefig(training_filename, dpi=150, bbox_inches='tight')
        print(f"SAC training results saved to: {training_filename}")
        plt.close()
        
        # Plot and save backtest results
        print("\nGenerating SAC backtest plots...")
        backtest_plot = plot_sac_backtest_results(training_results, ticker=ticker)
        backtest_filename = results_dir / f'sac_{ticker}_{timestamp}_backtest.png'
        backtest_plot.savefig(backtest_filename, dpi=150, bbox_inches='tight')
        print(f"SAC backtest results saved to: {backtest_filename}")
        plt.close()
        
        # Plot and save multi-day comparison
        print("\nGenerating SAC multi-day comparison plots...")
        multiday_plot = plot_sac_multi_day_comparison(training_results, ticker=ticker)
        if multiday_plot is not None:
            multiday_filename = results_dir / f'sac_{ticker}_{timestamp}_multiday.png'
            multiday_plot.savefig(multiday_filename, dpi=150, bbox_inches='tight')
            print(f"SAC multi-day comparison saved to: {multiday_filename}")
            plt.close()
        else:
            print("SAC multi-day comparison plot skipped (no multi-day data available)")
        
        # Print final summary
        multi_day_results = training_results['multi_day_test_results']
        aggregate_stats = multi_day_results['aggregate_stats']
        individual_days = multi_day_results['individual_days']
        
        print("\n" + "="*50)
        print("SAC MULTI-DAY TEST RESULTS SUMMARY")
        print("="*50)
        print(f"Ticker: {ticker}")
        print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
        print(f"Number of Test Days: {len(individual_days)}")
        print(f"Average Return: {aggregate_stats['avg_return']:.2%} ± {aggregate_stats['std_return']:.2%}")
        print(f"Best Day Return: {aggregate_stats['best_return']:.2%}")
        print(f"Worst Day Return: {aggregate_stats['worst_return']:.2%}")
        print(f"Win Rate: {aggregate_stats['win_rate']:.1%}")
        print(f"Average Trades per Day: {aggregate_stats.get('avg_trades', 0):.1f}")
        print(f"Average Invalid Actions: {aggregate_stats.get('avg_invalid_actions', 0):.1f}")
        
        # Save multi-day backtest results to database
        print("\nSaving SAC multi-day results to database...")
        
        model_id, model_path, model_dir, backtest_ids = save_multi_day_backtest_to_db(
            model_type=ModelType.SAC,
            ticker=ticker,
            multi_day_results=multi_day_results,
            start_date=start_date,
            end_date=end_date,
            initial_balance=INITIAL_BALANCE,
            preprocessor_path=None
        )
        
        print(f"✅ SAC Model created with ID: {model_id}")
        print(f"✅ Created {len(backtest_ids)} SAC backtest entries")
        
        # Save the trained SAC model
        create_directory(model_dir)
        print(f"\nSaving trained SAC model to: {model_path}")
        training_results['agent'].save(model_path)
        
        # Save the preprocessor
        if training_results.get('preprocessor') is not None:
            preprocessor_model_path = str(Path(model_dir) / 'preprocessor.pkl')
            print(f"\nSaving SAC preprocessor to: {preprocessor_model_path}")
            
            import shutil
            shutil.copy2(str(preprocessor_path), preprocessor_model_path)
            
            # Update the database with the preprocessor path
            with app.app_context():
                backtests = BacktestHistory.query.filter_by(model_id=model_id).all()
                for backtest in backtests:
                    backtest.preprocessor_path = preprocessor_model_path
                db.session.commit()
                print(f"Updated {len(backtests)} SAC backtest entries with preprocessor path")
        
        # Clean up temporary preprocessor file
        if preprocessor_path.exists():
            preprocessor_path.unlink()
            print(f"Cleaned up temporary SAC preprocessor file")
        
        print(f"\nAll SAC results saved to: {model_dir}")
        print("SAC training complete!")
        
        # Print SAC-specific summary
        print("\n" + "="*50)
        print("SAC IMPLEMENTATION SUMMARY")
        print("="*50)
        print("✅ Continuous Action Space: Precise buy/sell control")
        print("✅ Prioritized Experience Replay: Enhanced sample efficiency")
        print("✅ Twin Critics: Reduced overestimation bias")
        print("✅ Automatic Entropy Tuning: Optimal exploration-exploitation balance")
        print("✅ Enhanced Financial Networks: Multi-scale CNN + Transformers")
        print("✅ Portfolio State Normalization: Stable training features")
        print("="*50)
    
    try:
        custom_main()
        print("\n🎉 SAC Training completed successfully!")
    except Exception as e:
        print(f"\n❌ SAC Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 