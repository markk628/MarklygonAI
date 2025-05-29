"""
Eddie Real Data Training Script
실제 MarklygonAI 데이터를 사용한 Eddie 시스템 훈련 및 검증
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import warnings
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import psutil  # 메모리 모니터링용
import gc  # 가비지 컬렉션용

# Setup paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Eddie imports
try:
    from config import EddieConfig, DEFAULT_CONFIG, QUICK_CONFIG, HIGH_PERFORMANCE_CONFIG, MAX_GPU_CONFIG, ULTRA_HIGH_PERFORMANCE_CONFIG
    from data_loader import create_data_loaders, MarklygonDataProcessor
    from train_pipeline import EddieTrainer
    from utils.integration import EddiePredictor
    from utils.model_comparison import ModelComparator
except ImportError:
    # Handle relative imports when running as script
    import sys
    sys.path.append('.')
    from config import EddieConfig, DEFAULT_CONFIG, QUICK_CONFIG, HIGH_PERFORMANCE_CONFIG, MAX_GPU_CONFIG, ULTRA_HIGH_PERFORMANCE_CONFIG
    from data_loader import create_data_loaders, MarklygonDataProcessor
    from train_pipeline import EddieTrainer
    from utils.integration import EddiePredictor
    from utils.model_comparison import ModelComparator

# Suppress warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

def setup_logging(log_level: str = 'INFO', verbose: bool = True) -> logging.Logger:
    """로깅 설정"""
    # 더 상세한 포맷을 사용 (verbose 모드일 때)
    if verbose:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    else:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'eddie_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
        ]
    )
    
    # verbose 모드에서는 더 많은 라이브러리의 로그도 보여줍니다
    if verbose:
        logging.getLogger('MarklygonDataProcessor').setLevel(logging.INFO)
        logging.getLogger('root').setLevel(logging.INFO)
    
    return logging.getLogger('EddieRealDataTraining')

def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Eddie Real Data Training')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='./data',
                       help='Path to MarklygonAI data directory')
    parser.add_argument('--symbols', type=str, nargs='+',
                       default=['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL'],
                       help='Stock symbols to train on')
    
    # Model arguments  
    parser.add_argument('--config', type=str, choices=['default', 'quick', 'high_perf', 'max_gpu', 'ultra'],
                       default='quick', help='Configuration preset (optimized for RTX 4060 Ti)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    # Training arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--num_workers', type=int, default=12,
                       help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience (number of epochs)')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, 
                       default=f'eddie_real_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='Experiment name for tracking')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--dry_run', action='store_true',
                       help='Perform dry run (data loading only)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Enable verbose logging (default: True)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()

def get_config(config_name: str) -> EddieConfig:
    """설정 프리셋 선택"""
    configs = {
        'default': DEFAULT_CONFIG,
        'quick': QUICK_CONFIG,
        'high_perf': HIGH_PERFORMANCE_CONFIG,
        'max_gpu': MAX_GPU_CONFIG,
        'ultra': ULTRA_HIGH_PERFORMANCE_CONFIG
    }
    return configs.get(config_name, QUICK_CONFIG)  # Default to optimized quick config

def setup_device(device_arg: str) -> torch.device:
    """디바이스 설정"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    else:
        print("Using CPU")
    
    return device

def validate_data_path(data_path: str) -> Path:
    """데이터 경로 검증"""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Check for data directories
    required_dirs = ['feature_engineered', 'processed']
    available_dirs = [d for d in required_dirs if (data_path / d).exists()]
    
    if not available_dirs:
        raise FileNotFoundError(f"No data directories found in {data_path}. "
                               f"Expected: {required_dirs}")
    
    print(f"Found data directories: {available_dirs}")
    return data_path

def perform_data_analysis(
    train_loader, 
    val_loader, 
    test_loader, 
    processor: MarklygonDataProcessor,
    save_dir: Path
):
    """데이터 분석 및 시각화"""
    print("\n" + "="*50)
    print("PERFORMING DATA ANALYSIS")
    print("="*50)
    
    # 1. Dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # 2. Feature statistics
    sample_batch = next(iter(train_loader))
    features_shape = sample_batch['features'].shape
    print(f"\nFeature Information:")
    print(f"  Sequence length: {features_shape[1]}")
    print(f"  Feature dimension: {features_shape[2]}")
    print(f"  Selected features: {len(processor.feature_columns)}")
    
    # 3. Target distribution analysis
    targets_analysis = {}
    for key in ['sell_intensity', 'market_regime', 'volatility', 'uncertainty']:
        if key in sample_batch:
            values = []
            for batch in train_loader:
                values.append(batch[key].numpy())
            values = np.concatenate(values)
            
            targets_analysis[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'unique_values': len(np.unique(values)) if key == 'market_regime' else None
            }
    
    print(f"\nTarget Analysis:")
    for key, stats in targets_analysis.items():
        print(f"  {key}:")
        print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Mean±Std: {stats['mean']:.3f}±{stats['std']:.3f}")
        if stats['unique_values']:
            print(f"    Unique values: {stats['unique_values']}")
    
    # 4. Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (key, stats) in enumerate(targets_analysis.items()):
        if i >= 4:
            break
        
        # Collect values for histogram
        values = []
        for batch in train_loader:
            if key in batch:
                values.append(batch[key].numpy())
        
        if values:
            values = np.concatenate(values)
            axes[i].hist(values, bins=50, alpha=0.7, density=True)
            axes[i].set_title(f'{key.replace("_", " ").title()} Distribution')
            axes[i].set_xlabel(key)
            axes[i].set_ylabel('Density')
    
    # Hide empty subplots
    for i in range(len(targets_analysis), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'data_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nData analysis plots saved to {save_dir / 'data_analysis.png'}")

def monitor_memory_usage(stage: str = "Unknown"):
    """메모리 사용량 모니터링 및 로깅"""
    # System Memory
    memory = psutil.virtual_memory()
    memory_gb = memory.used / (1024**3)
    memory_percent = memory.percent
    
    # GPU Memory
    gpu_memory_str = "N/A"
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
        gpu_memory_str = f"Allocated: {gpu_memory_allocated:.2f}GB, Reserved: {gpu_memory_reserved:.2f}GB"
    
    log_msg = f"[{stage}] Memory - System: {memory_gb:.2f}GB ({memory_percent:.1f}%), GPU: {gpu_memory_str}"
    print(log_msg)
    logging.info(log_msg)
    
    return {
        'system_memory_gb': memory_gb,
        'system_memory_percent': memory_percent,
        'gpu_memory_allocated_gb': gpu_memory_allocated if torch.cuda.is_available() else 0,
        'gpu_memory_reserved_gb': gpu_memory_reserved if torch.cuda.is_available() else 0
    }

def cleanup_memory():
    """메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 메모리 정리 완료")

def main():
    """메인 실행 함수"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(log_level=args.log_level, verbose=args.verbose)
    logger.info(f"🚀 Starting Eddie Real Data Training: {args.experiment_name}")
    
    if args.verbose:
        logger.info(f"💡 Verbose mode enabled - detailed progress logging active")
        logger.info(f"📊 Configuration: {args.config}")
        logger.info(f"🎯 Target symbols: {args.symbols[:5]}{'...' if len(args.symbols) > 5 else ''}")
        logger.info(f"🏋️ Training epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    # Setup directories
    save_dir = Path(args.save_dir) / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {save_dir}")
    
    # Validate data path
    try:
        data_path = validate_data_path(args.data_path)
        logger.info(f"Data path validated: {data_path}")
    except Exception as e:
        logger.error(f"Data path validation failed: {e}")
        return
    
    # Setup device
    device = setup_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Get configuration
    config = get_config(args.config)
    logger.info(f"Using configuration: {args.config}")
    
    # Update config with command line arguments
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.device = str(device)
    config.training.early_stopping_patience = args.early_stopping_patience
    
    try:
        print("\n" + "="*50)
        print("LOADING AND PREPROCESSING DATA")
        print("="*50)
        
        # Initial memory monitoring
        monitor_memory_usage("Start")
        
        # Create data loaders
        print(f"Loading data for {len(args.symbols)} symbols...")
        train_loader, val_loader, test_loader, processor = create_data_loaders(
            data_path=str(data_path),
            config=config,
            symbols=args.symbols,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Monitor memory after data loading
        monitor_memory_usage("After Data Loading")
        cleanup_memory()
        
        logger.info("Data loading completed successfully")
        
        # Save preprocessor
        processor.save_preprocessor(str(save_dir / 'preprocessor.pkl'))
        
        # Perform data analysis
        perform_data_analysis(train_loader, val_loader, test_loader, processor, save_dir)
        
        # If dry run, stop here
        if args.dry_run:
            print("\n" + "="*50)
            print("DRY RUN COMPLETED SUCCESSFULLY")
            print("="*50)
            logger.info("Dry run completed - data loading and analysis successful")
            return
        
        print("\n" + "="*50)
        print("STARTING MODEL TRAINING")
        print("="*50)
        
        # Initialize trainer
        trainer = EddieTrainer(
            config=config,
            device=device,
            use_wandb=args.use_wandb
        )
        
        # Set data loaders
        trainer.set_data_loaders(train_loader, val_loader, test_loader)
        
        # Train the model
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            save_dir=str(save_dir)
        )
        
        logger.info("Training completed successfully")
        
        print("\n" + "="*50)
        print("PERFORMING MODEL EVALUATION")
        print("="*50)
        
        # Load best model for evaluation
        best_model_path = save_dir / 'best_model.pt'
        if best_model_path.exists():
            predictor = EddiePredictor(
                model_path=str(best_model_path),
                config=config,
                preprocessor_path=str(save_dir / 'preprocessor.pkl'),
                device=device
            )
            
            # Evaluate on test set
            test_results = trainer.evaluate(test_loader, return_predictions=True)
            
            # Save test results
            test_results_path = save_dir / 'test_results.pkl'
            import pickle
            with open(test_results_path, 'wb') as f:
                pickle.dump(test_results, f)
            
            # Print test metrics
            print(f"\nTest Set Results:")
            for metric, value in test_results['metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
            
            logger.info(f"Test results saved to {test_results_path}")
        
        print("\n" + "="*50)
        print("CREATING FINAL REPORT")
        print("="*50)
        
        # Create final report
        report = {
            'experiment_name': args.experiment_name,
            'config': config,
            'data_info': {
                'symbols': args.symbols,
                'train_samples': len(train_loader.dataset),
                'val_samples': len(val_loader.dataset),
                'test_samples': len(test_loader.dataset),
                'feature_count': len(processor.feature_columns),
                'sequence_length': config.signal_generator.seq_len
            },
            'training_results': results,
            'test_results': test_results if 'test_results' in locals() else None,
            'model_path': str(best_model_path) if best_model_path.exists() else None,
            'preprocessor_path': str(save_dir / 'preprocessor.pkl')
        }
        
        # Save report
        report_path = save_dir / 'experiment_report.pkl'
        with open(report_path, 'wb') as f:
            pickle.dump(report, f)
        
        # Create human-readable summary
        summary_path = save_dir / 'experiment_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Eddie Real Data Training Summary\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Experiment: {args.experiment_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {args.config}\n")
            f.write(f"Symbols: {', '.join(args.symbols)}\n\n")
            f.write(f"Data Information:\n")
            f.write(f"  - Training samples: {len(train_loader.dataset)}\n")
            f.write(f"  - Validation samples: {len(val_loader.dataset)}\n")
            f.write(f"  - Test samples: {len(test_loader.dataset)}\n")
            f.write(f"  - Feature count: {len(processor.feature_columns)}\n")
            f.write(f"  - Sequence length: {config.signal_generator.seq_len}\n\n")
            f.write(f"Training Results:\n")
            if results and 'best_metrics' in results:
                for metric, value in results['best_metrics'].items():
                    if isinstance(value, (int, float)):
                        f.write(f"  - {metric}: {value:.4f}\n")
            f.write(f"\nModel saved to: {best_model_path}\n")
            f.write(f"Preprocessor saved to: {save_dir / 'preprocessor.pkl'}\n")
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {save_dir}")
        print(f"Summary available at: {summary_path}")
        
        logger.info(f"Experiment completed successfully: {args.experiment_name}")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        print("\nTraining interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        print(f"\nTraining failed: {e}")
        raise

if __name__ == "__main__":
    main() 