import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path

from src.models.jeawan.sac.sac import train_sac
from src.models.jeawan.sac.sac_config import NetworkType, EnvironmentType  # Import the new enums
from src.models.mark.dqn_v2.database import save_multi_day_backtest_to_db
from src.models.jeawan.sac.visualization import plot_sac_training_results, plot_sac_backtest_results, plot_sac_multi_day_comparison
from src.config.config import CUTOFF_TIMESTAMP, DATA_DIR, MODELS_DIR, RESULTS_DIR, EVALUATE_INTERVAL, INITIAL_BALANCE
from src.utils.utils import create_directory
from src.web.models import ModelType
from src.web.extensions import app
from src.web.models import BacktestHistory, db


def main():
    # Configuration
    ticker = 'TSLA'  # Change to your stock ticker
    data_path = f"{DATA_DIR}/feature_engineered_v2/{ticker}.csv"
    
    # NEW: SAC Architecture Selection
    network_type = NetworkType.ORIGINAL  # Choose: SIMPLIFIED (170k params) or ORIGINAL (900k params)
    environment_type = EnvironmentType.WEIGHTED_AVERAGE  # Choose: BASIC, WEIGHTED_AVERAGE, LOT_BASED
    
    print(f"🚀 SAC Training Configuration:")
    print(f"   📊 Network Architecture: {network_type.value.upper()}")
    print(f"   🏪 Environment Type: {environment_type.value.upper()}")
    print(f"   🎯 Ticker: {ticker}")
    print("="*50)
    
    # Create results directory if it doesn't exist
    models_dir = MODELS_DIR / 'sac'
    results_dir = RESULTS_DIR / 'sac'
    create_directory(models_dir)
    create_directory(results_dir)
    
    # Preprocessor save path
    preprocessor_path = models_dir / f'preprocessor_{ticker}_{network_type.value}.pkl'
    
    # Train the SAC agent with the new modular architecture
    print(f"🎯 Starting SAC training for {ticker} with {network_type.value} networks...")
    print("="*50)
    
    cutoff = pd.Timestamp(CUTOFF_TIMESTAMP, tz='UTC')
    
    # NEW: Custom configuration overrides for better performance
    config_overrides = {
        # Enhanced training parameters for the selected architecture
        'actor_learning_rate': 5e-4 if network_type == NetworkType.SIMPLIFIED else 3e-4,
        'critic_learning_rate': 5e-4 if network_type == NetworkType.SIMPLIFIED else 3e-4,
        'target_entropy': -0.2,  # Even more aggressive to encourage active trading (was -0.3)
        'use_action_guidance': True,  # Enable continuous action masking
        'action_guidance_strength': 0.6,  # Strong guidance for better performance
        'soft_invalid_penalty': 0.005,  # Very gentle penalties with guidance
        
        # Reward scaling fix for proper SAC learning  
        'portfolio_scaling': 0.1,  # Balanced scaling - encourages trading while keeping training stable
        'invalid_penalty': 0.05,  # Consistent with soft penalties
        
        # Learning rate scheduling for stability
        'use_lr_scheduler': True,
        'scheduler_type': 'plateau',
        'scheduler_patience': 8,
        'scheduler_factor': 0.7,
        
        # Enhanced replay buffer for continuous actions
        'per_alpha': 0.7,  # Slightly higher prioritization for SAC
        'per_beta_start': 0.5,  # Start with more importance sampling
    }
    
    training_results = train_sac(
        data_path=data_path,
        cutoff=cutoff,
        num_episodes=500,  # More episodes for SAC convergence
        save_interval=50,
        validation_frequency=EVALUATE_INTERVAL,
        early_stopping_patience=15,  # Appropriate patience for SAC
        use_preprocessing=True,  # Enable preprocessing
        scaling_method='robust',  # Best for financial data
        outlier_method='winsorize',  # Handle outliers
        preprocessor_save_path=str(preprocessor_path),
        # NEW PARAMETERS:
        network_type=network_type,  # Select architecture
        environment_type=environment_type,  # Select environment
        config_overrides=config_overrides  # Apply custom settings
    )
    
    # Get start and end dates from training results
    start_date = training_results['start_date']
    end_date = training_results['end_date']
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Include architecture info in filenames
    base_name = f'sac_{ticker}_{network_type.value}_{environment_type.value}_{timestamp}'
    
    # Plot and save training results
    print("\n📊 Generating SAC training plots...")
    training_plot = plot_sac_training_results(training_results, ticker=ticker, validation_frequency=EVALUATE_INTERVAL)
    training_filename = results_dir / f'{base_name}_training.png'
    training_plot.savefig(training_filename, dpi=150, bbox_inches='tight')
    print(f"✅ SAC training results saved to: {training_filename}")
    plt.close()
    
    # Plot and save backtest results
    print("\n📈 Generating SAC backtest plots...")
    backtest_plot = plot_sac_backtest_results(training_results, ticker=ticker)
    backtest_filename = results_dir / f'{base_name}_backtest.png'
    backtest_plot.savefig(backtest_filename, dpi=150, bbox_inches='tight')
    print(f"✅ SAC backtest results saved to: {backtest_filename}")
    plt.close()
    
    # Plot and save multi-day comparison
    print("\n📊 Generating SAC multi-day comparison plots...")
    multiday_plot = plot_sac_multi_day_comparison(training_results, ticker=ticker)
    if multiday_plot is not None:
        multiday_filename = results_dir / f'{base_name}_multiday.png'
        multiday_plot.savefig(multiday_filename, dpi=150, bbox_inches='tight')
        print(f"✅ SAC multi-day comparison saved to: {multiday_filename}")
        plt.close()
    else:
        print("⚠️ SAC multi-day comparison plot skipped (no multi-day data available)")
    
    # Print final summary
    multi_day_results = training_results['multi_day_test_results']
    aggregate_stats = multi_day_results['aggregate_stats']
    individual_days = multi_day_results['individual_days']
    
    print("\n" + "="*50)
    print("📊 SAC MULTI-DAY TEST RESULTS SUMMARY")
    print("="*50)
    print(f"🎯 Ticker: {ticker}")
    print(f"🧠 Network: {network_type.value.upper()} ({training_results['agent'].get_network_summary()['total_params']:,} parameters)")
    print(f"🏪 Environment: {environment_type.value.upper()}")
    print(f"💰 Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"📅 Number of Test Days: {len(individual_days)}")
    print(f"📈 Average Return: {aggregate_stats['avg_return']:.2%} ± {aggregate_stats['std_return']:.2%}")
    print(f"🏆 Best Day Return: {aggregate_stats['best_return']:.2%}")
    print(f"📉 Worst Day Return: {aggregate_stats['worst_return']:.2%}")
    print(f"🎯 Win Rate: {aggregate_stats['win_rate']:.1%}")
    
    # SAC-specific metrics
    avg_trades = np.mean([day['total_trades'] for day in individual_days])
    avg_invalid = np.mean([day['invalid_actions'] for day in individual_days])
    avg_sharpe = np.mean([day['sharpe_ratio'] for day in individual_days])
    avg_drawdown = np.mean([day['max_drawdown'] for day in individual_days])
    
    print(f"📊 Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"📉 Average Max Drawdown: {avg_drawdown:.2%}")
    print(f"🔄 Average Trades per Day: {avg_trades:.1f}")
    print(f"⚠️ Average Invalid Actions: {avg_invalid:.1f}")
    
    print(f"\n📋 Individual Day Results:")
    for i, day_result in enumerate(individual_days, 1):
        print(f"   Day {i}: {day_result['total_return']:.1%} return, "
              f"{day_result['total_trades']} trades, "
              f"{day_result['invalid_actions']} invalid actions, "
              f"Sharpe: {day_result['sharpe_ratio']:.1f}")
    print("="*50)
    
    # Save multi-day backtest results to database
    print("\n💾 Saving SAC multi-day results to database...")
    
    # Create enhanced model type name with architecture info
    model_type_name = f"SAC_{network_type.value.upper()}_{environment_type.value.upper()}"
    
    # Save one model with multiple backtest entries (use SAC model type)
    model_id, model_path, model_dir, backtest_ids = save_multi_day_backtest_to_db(
        model_type=ModelType.SAC,  # Use SAC model type
        ticker=ticker,
        multi_day_results=multi_day_results,
        start_date=start_date,
        end_date=end_date,
        initial_balance=INITIAL_BALANCE,
        preprocessor_path=None  # Will be updated later
    )
    
    print(f"✅ SAC Model created with ID: {model_id}")
    print(f"   🧠 Architecture: {network_type.value} networks")
    print(f"   🏪 Environment: {environment_type.value}")
    print(f"✅ Created {len(backtest_ids)} SAC backtest entries:")
    print(f"   • Aggregate summary (backtest ID: {backtest_ids[0]})")
    for i, backtest_id in enumerate(backtest_ids[1:], 1):
        day_return = individual_days[i-1]['total_return']
        print(f"   • Day {i} (backtest ID: {backtest_id}): {day_return:.1%} return")
    
    print(f"🔗 All SAC results linked to model ID: {model_id}")
    
    # Ensure model directory exists
    create_directory(model_dir)
    print(f"📁 SAC Model directory: {model_dir}")
    
    # Save the trained SAC model with architecture info in filename
    model_filename = f"sac_{network_type.value}_{environment_type.value}_{ticker}.pt"
    final_model_path = str(Path(model_dir) / model_filename)
    print(f"\n💾 Saving trained SAC model to: {final_model_path}")
    training_results['agent'].save(final_model_path, training_results.get('config'))
    
    # Save the preprocessor in the same directory
    if training_results.get('preprocessor') is not None:
        preprocessor_model_path = str(Path(model_dir) / 'preprocessor.pkl')
        print(f"\n📊 Saving SAC preprocessor to: {preprocessor_model_path}")
        
        # Copy the preprocessor from temporary location to model directory
        import shutil
        shutil.copy2(str(preprocessor_path), preprocessor_model_path)
        
        # Update the database with the preprocessor path for all backtest entries
        with app.app_context():
            backtests = BacktestHistory.query.filter_by(model_id=model_id).all()
            for backtest in backtests:
                backtest.preprocessor_path = preprocessor_model_path
            db.session.commit()
            print(f"🔄 Updated {len(backtests)} SAC backtest entries with preprocessor path")
        
        print(f"⚙️ SAC Preprocessing Configuration: {training_results['preprocessor'].get_preprocessing_info()}")
    
    # Clean up temporary preprocessor file if it exists
    if preprocessor_path.exists():
        preprocessor_path.unlink()
        print(f"🧹 Cleaned up temporary SAC preprocessor file")
    
    print(f"\n📁 All SAC results saved to: {model_dir}")
    print("🎉 SAC training complete!")
    
    # Print SAC-specific summary with architecture info
    print("\n" + "="*50)
    print("🧠 SAC MODULAR IMPLEMENTATION SUMMARY")
    print("="*50)
    print(f"✅ Network Architecture: {network_type.value.upper()}")
    print(f"   • Parameters: {training_results['agent'].get_network_summary()['total_params']:,}")
    print(f"   • Type: {'Efficient & Fast' if network_type == NetworkType.SIMPLIFIED else 'Complex & Comprehensive'}")
    print(f"✅ Environment Type: {environment_type.value.upper()}")
    print(f"✅ Continuous Action Space: Precise buy/sell control")
    print(f"✅ Action Masking: {avg_invalid:.1f} avg invalid actions/day (vs 300+ without)")
    print(f"✅ Prioritized Experience Replay: Enhanced sample efficiency") 
    print(f"✅ Twin Critics: Reduced overestimation bias")
    print(f"✅ Automatic Entropy Tuning: Optimal exploration-exploitation")
    print(f"✅ Portfolio State Normalization: Stable training features")
    print(f"✅ Learning Rate Scheduling: Adaptive optimization")
    print("="*50)
    
    # Architecture comparison info
    if network_type == NetworkType.SIMPLIFIED:
        print("💡 TIP: You're using SIMPLIFIED networks (recommended)")
        print("   • 170k parameters vs 900k+ for ORIGINAL")
        print("   • Faster training and inference")
        print("   • Better for most use cases")
        print("   • To try ORIGINAL networks, set network_type=NetworkType.ORIGINAL")
    else:
        print("💡 INFO: You're using ORIGINAL networks (complex)")
        print("   • 900k+ parameters with full multi-scale processing")
        print("   • Slower but potentially more expressive")
        print("   • For faster training, try network_type=NetworkType.SIMPLIFIED")
    
    print("\n🏪 Environment Options:")
    print("   • BASIC: Simple buy-sell cycles")
    print("   • WEIGHTED_AVERAGE: Cost basis tracking (current)")
    print("   • LOT_BASED: Individual lot management")
    
    print(f"\n🎯 To modify architecture/environment:")
    print(f"   network_type = NetworkType.SIMPLIFIED  # or .ORIGINAL")
    print(f"   environment_type = EnvironmentType.WEIGHTED_AVERAGE  # or .BASIC, .LOT_BASED")


if __name__ == "__main__":
    main() 