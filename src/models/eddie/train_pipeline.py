"""
Eddie's Main Training Pipeline
Signal Generator + Pattern Analyzer 통합 훈련 시스템
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import wandb
from typing import Dict, List, Tuple, Optional, Any
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from config import EddieConfig, DEFAULT_CONFIG, QUICK_CONFIG, HIGH_PERFORMANCE_CONFIG
from signal_pipeline.signal_generator import SignalGenerator, SignalGeneratorTrainer
from pattern_pipeline.pattern_analyzer import PatternAnalyzer


class EddieDataset(Dataset):
    """
    Eddie 훈련용 데이터셋
    TA-Lib 지표 데이터와 타겟 변수를 로드
    """
    
    def __init__(
        self,
        data_path: str,
        seq_len: int = 60,
        target_col: str = 'future_return',
        split: str = 'train',
        normalize: bool = True
    ):
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.target_col = target_col
        self.split = split
        self.normalize = normalize
        
        # Load data
        self.features, self.targets, self.metadata = self._load_data()
        
        # Create sequences
        self.sequences, self.sequence_targets = self._create_sequences()
        
        logging.info(f"Dataset {split}: {len(self.sequences)} sequences loaded")
        
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """데이터 로드 및 전처리"""
        
        # Load feature engineered data
        features_file = self.data_path / f"features_{self.split}.pkl"
        targets_file = self.data_path / f"targets_{self.split}.pkl"
        
        if not features_file.exists() or not targets_file.exists():
            raise FileNotFoundError(f"Data files not found in {self.data_path}")
        
        with open(features_file, 'rb') as f:
            features_data = pickle.load(f)
        
        with open(targets_file, 'rb') as f:
            targets_data = pickle.load(f)
        
        # Extract features (TA-Lib indicators)
        features = features_data['talib_indicators']  # (N, num_features)
        
        # Extract targets
        targets = {
            'sell_intensity': targets_data['sell_intensity'],  # [-1, 1]
            'market_regime': targets_data['market_regime'],    # [0, 1, 2, 3]
            'volatility': targets_data['volatility'],          # [0, 1]
            'future_return': targets_data['future_return']     # Continuous
        }
        
        # Normalization
        if self.normalize:
            features = self._normalize_features(features)
        
        # Metadata
        metadata = {
            'feature_names': features_data.get('feature_names', []),
            'timestamps': features_data.get('timestamps', []),
            'symbols': features_data.get('symbols', [])
        }
        
        return features, targets, metadata
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """특징 정규화 (Z-score)"""
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        return (features - mean) / std
    
    def _create_sequences(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """시퀀스 데이터 생성"""
        num_samples = len(self.features) - self.seq_len + 1
        
        sequences = np.zeros((num_samples, self.seq_len, self.features.shape[1]))
        sequence_targets = {key: np.zeros(num_samples) for key in self.targets.keys()}
        
        for i in range(num_samples):
            sequences[i] = self.features[i:i + self.seq_len]
            
            # Use the last timestep as target
            for key, values in self.targets.items():
                sequence_targets[key][i] = values[i + self.seq_len - 1]
        
        return sequences, sequence_targets
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        features = torch.FloatTensor(self.sequences[idx])
        
        targets = {}
        for key, values in self.sequence_targets.items():
            targets[key] = torch.FloatTensor([values[idx]])
        
        return {
            'features': features,
            **targets
        }


class EddieTrainer:
    """
    Eddie 통합 훈련 시스템
    Signal Generator와 Pattern Analyzer를 함께 훈련
    """
    
    def __init__(
        self,
        config: EddieConfig,
        device: torch.device = None,
        use_wandb: bool = True
    ):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb
        
        # Setup logging
        self._setup_logging()
        
        # Initialize models
        self._initialize_models()
        
        # Initialize optimizers and schedulers
        self._initialize_optimizers()
        
        # Data loaders (will be set externally)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Initialize wandb
        if self.use_wandb:
            self._setup_wandb()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('EddieTrainer')
        
    def _initialize_models(self):
        """모델 초기화"""
        # Signal Generator
        self.signal_generator = SignalGenerator(
            num_inputs=self.config.talib.pca_components,
            tcn_channels=self.config.signal_generator.tcn.num_channels,
            tcn_kernel_size=self.config.signal_generator.tcn.kernel_size,
            tcn_dropout=self.config.signal_generator.tcn.dropout,
            attention_heads=self.config.signal_generator.attention.num_heads,
            attention_dropout=self.config.signal_generator.attention.dropout,
            seq_len=self.config.signal_generator.seq_len,
            use_uncertainty=self.config.signal_generator.use_uncertainty
        ).to(self.device)
        
        # Pattern Analyzer
        self.pattern_analyzer = PatternAnalyzer(
            num_features=self.config.talib.pca_components,
            seq_len=self.config.signal_generator.seq_len,
            timesnet_config={
                'seq_len': self.config.pattern_analyzer.timesnet.seq_len,
                'pred_len': self.config.pattern_analyzer.timesnet.pred_len,
                'top_k': self.config.pattern_analyzer.timesnet.top_k,
                'd_model': self.config.pattern_analyzer.timesnet.d_model,
                'd_ff': self.config.pattern_analyzer.timesnet.d_ff,
                'num_kernels': self.config.pattern_analyzer.timesnet.num_kernels,
                'num_layers': 2,
                'dropout': 0.1
            },
            wavelet_config={
                'wavelet': self.config.pattern_analyzer.wavelet.wavelet,
                'levels': self.config.pattern_analyzer.wavelet.levels,
                'feature_dim': self.config.pattern_analyzer.feature_dim,
                'use_denoising': True
            },
            feature_dim=self.config.pattern_analyzer.feature_dim,
            integration_method=self.config.pattern_analyzer.integration_method,
            use_market_regime=self.config.pattern_analyzer.use_market_regime
        ).to(self.device)
        
        self.logger.info(f"Models initialized on {self.device}")
        
    def _initialize_optimizers(self):
        """옵티마이저 및 스케줄러 초기화"""
        # Signal Generator optimizer
        self.signal_optimizer = torch.optim.AdamW(
            self.signal_generator.parameters(),
            lr=self.config.training.signal_lr,
            weight_decay=self.config.training.signal_weight_decay
        )
        
        # Pattern Analyzer optimizer
        self.pattern_optimizer = torch.optim.AdamW(
            self.pattern_analyzer.parameters(),
            lr=self.config.training.pattern_lr,
            weight_decay=self.config.training.pattern_weight_decay
        )
        
        # Schedulers
        self.signal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.signal_optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.pattern_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.pattern_optimizer, mode='min', factor=0.5, patience=10
        )
        
    def set_data_loaders(self, train_loader, val_loader=None, test_loader=None):
        """외부에서 데이터 로더 설정"""
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.logger.info(f"Data loaders set: Train={len(train_loader.dataset)}")
        if val_loader:
            self.logger.info(f"Validation={len(val_loader.dataset)}")
        if test_loader:
            self.logger.info(f"Test={len(test_loader.dataset)}")
        
    def _setup_wandb(self):
        """Weights & Biases 설정"""
        wandb.init(
            project="MarklygonAI-Eddie",
            name=f"{self.config.experiment_name}_{self.config.version}",
            config=self.config.__dict__
        )
        
    def compute_combined_loss(
        self,
        signal_outputs: Dict[str, torch.Tensor],
        pattern_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """통합 손실 함수"""
        losses = {}
        
        # 1. Signal Generator Loss
        signal_loss = F.mse_loss(
            signal_outputs['sell_intensity'],
            targets['sell_intensity'].squeeze()
        )
        losses['signal_mse'] = signal_loss
        
        # 2. Pattern Analyzer Losses
        # Volatility prediction
        volatility_loss = F.mse_loss(
            pattern_outputs['volatility'],
            targets['volatility'].squeeze()
        )
        losses['volatility_mse'] = volatility_loss
        
        # Market regime classification
        if 'regime_probabilities' in pattern_outputs:
            regime_loss = F.cross_entropy(
                pattern_outputs['regime_probabilities'],
                targets['market_regime'].squeeze().long()
            )
            losses['regime_ce'] = regime_loss
        
        # 3. Consistency Loss (align signal and pattern predictions)
        signal_intensity = signal_outputs['sell_intensity']
        trend_strength = pattern_outputs['trend_strength']
        
        consistency_loss = F.mse_loss(signal_intensity, trend_strength)
        losses['consistency'] = consistency_loss
        
        # 4. Uncertainty Regularization
        if 'uncertainty' in signal_outputs:
            uncertainty_reg = signal_outputs['uncertainty'].mean()
            losses['uncertainty_reg'] = uncertainty_reg
        
        # 5. Combine losses with weights
        total_loss = (
            1.0 * losses['signal_mse'] +
            0.5 * losses['volatility_mse'] +
            0.3 * losses.get('regime_ce', 0) +
            0.2 * losses['consistency'] +
            0.01 * losses.get('uncertainty_reg', 0)
        )
        
        # Convert to float for logging
        loss_components = {k: v.item() if isinstance(v, torch.Tensor) else v 
                          for k, v in losses.items()}
        
        return total_loss, loss_components
        
    def train_epoch(self) -> Dict[str, float]:
        """한 에포크 훈련"""
        self.signal_generator.train()
        self.pattern_analyzer.train()
        
        total_loss = 0.0
        total_components = {}
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {self.epoch}")
        
        for batch in progress_bar:
            # Move to device
            features = batch['features'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != 'features'}
            
            # Zero gradients
            self.signal_optimizer.zero_grad()
            self.pattern_optimizer.zero_grad()
            
            # Forward passes
            signal_outputs = self.signal_generator(features)
            pattern_outputs = self.pattern_analyzer(features)
            
            # Compute loss
            loss, loss_components = self.compute_combined_loss(
                signal_outputs, pattern_outputs, targets
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.signal_generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.pattern_analyzer.parameters(), max_norm=1.0)
            
            # Optimizer steps
            self.signal_optimizer.step()
            self.pattern_optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_components.items():
                total_components[key] = total_components.get(key, 0) + value
            
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'signal': f"{loss_components['signal_mse']:.4f}",
                'pattern': f"{loss_components['volatility_mse']:.4f}"
            })
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        self.train_losses.append(avg_loss)
        
        return {'total_loss': avg_loss, **avg_components}
    
    def validate_epoch(self) -> Dict[str, float]:
        """검증"""
        self.signal_generator.eval()
        self.pattern_analyzer.eval()
        
        total_loss = 0.0
        total_components = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                features = batch['features'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch.items() if k != 'features'}
                
                # Forward passes
                signal_outputs = self.signal_generator(features)
                pattern_outputs = self.pattern_analyzer(features)
                
                # Compute loss
                loss, loss_components = self.compute_combined_loss(
                    signal_outputs, pattern_outputs, targets
                )
                
                # Accumulate losses
                total_loss += loss.item()
                for key, value in loss_components.items():
                    total_components[key] = total_components.get(key, 0) + value
                
                num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_components.items()}
        
        self.val_losses.append(avg_loss)
        
        # Update schedulers
        self.signal_scheduler.step(avg_loss)
        self.pattern_scheduler.step(avg_loss)
        
        return {'total_loss': avg_loss, **avg_components}
    
    def train(self, train_loader=None, val_loader=None, num_epochs: int = None, save_dir: str = None):
        """전체 훈련 루프"""
        if num_epochs is None:
            num_epochs = max(self.config.training.signal_epochs, 
                           self.config.training.pattern_epochs)
        
        # Set data loaders
        if train_loader is not None:
            self.train_loader = train_loader
        if val_loader is not None:
            self.val_loader = val_loader
            
        if self.train_loader is None:
            raise ValueError("Training data loader must be provided")
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Logging
            self.logger.info(
                f"Epoch {epoch}: Train Loss={train_metrics['total_loss']:.4f}, "
                f"Val Loss={val_metrics['total_loss']:.4f}"
            )
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/total_loss': train_metrics['total_loss'],
                    'val/total_loss': val_metrics['total_loss'],
                    **{f'train/{k}': v for k, v in train_metrics.items() if k != 'total_loss'},
                    **{f'val/{k}': v for k, v in val_metrics.items() if k != 'total_loss'}
                })
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_checkpoint(save_dir, is_best=True)
                self.logger.info(f"New best model saved with val_loss={self.best_val_loss:.4f}")
            
            # Regular checkpoint
            if (epoch + 1) % getattr(self.config.training, 'save_every', 10) == 0:
                self.save_checkpoint(save_dir, is_best=False)
            
            # Early stopping
            if self._should_early_stop():
                self.logger.info("Early stopping triggered")
                break
        
        # Final evaluation
        if self.test_loader is not None:
            self.evaluate()
        
        if self.use_wandb:
            wandb.finish()
            
        # Return training results
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.epoch
        }
    
    def _should_early_stop(self) -> bool:
        """조기 종료 확인"""
        if len(self.val_losses) < self.config.training.early_stopping_patience:
            return False
        
        recent_losses = self.val_losses[-self.config.training.early_stopping_patience:]
        min_loss = min(recent_losses)
        
        return min_loss == recent_losses[0]  # No improvement
    
    def save_checkpoint(self, save_dir: str = None, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.epoch,
            'signal_generator_state': self.signal_generator.state_dict(),
            'pattern_analyzer_state': self.pattern_analyzer.state_dict(),
            'signal_optimizer_state': self.signal_optimizer.state_dict(),
            'pattern_optimizer_state': self.pattern_optimizer.state_dict(),
            'signal_scheduler_state': self.signal_scheduler.state_dict(),
            'pattern_scheduler_state': self.pattern_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Create directory
        if save_dir is None:
            save_dir = getattr(self.config, 'model_save_path', './model_checkpoints')
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        if is_best:
            save_path = save_dir / 'best_model.pt'
        else:
            save_path = save_dir / f'checkpoint_epoch_{self.epoch}.pt'
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved: {save_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.signal_generator.load_state_dict(checkpoint['signal_generator_state'])
        self.pattern_analyzer.load_state_dict(checkpoint['pattern_analyzer_state'])
        self.signal_optimizer.load_state_dict(checkpoint['signal_optimizer_state'])
        self.pattern_optimizer.load_state_dict(checkpoint['pattern_optimizer_state'])
        self.signal_scheduler.load_state_dict(checkpoint['signal_scheduler_state'])
        self.pattern_scheduler.load_state_dict(checkpoint['pattern_scheduler_state'])
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        
    def evaluate(self, test_loader=None, return_predictions=False):
        """모델 평가"""
        self.logger.info("Starting evaluation...")
        
        # Use provided test loader or default to self.test_loader
        if test_loader is None:
            test_loader = self.test_loader
        
        if test_loader is None:
            self.logger.warning("No test loader available for evaluation")
            return None
        
        self.signal_generator.eval()
        self.pattern_analyzer.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation"):
                features = batch['features'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch.items() if k != 'features'}
                
                signal_outputs = self.signal_generator(features)
                pattern_outputs = self.pattern_analyzer(features)
                
                # Collect predictions and targets
                batch_pred = {
                    'sell_intensity': signal_outputs['sell_intensity'].cpu().numpy(),
                    'volatility': pattern_outputs['volatility'].cpu().numpy(),
                    'trend_strength': pattern_outputs['trend_strength'].cpu().numpy(),
                    'pattern_confidence': pattern_outputs['pattern_confidence'].cpu().numpy()
                }
                
                if 'regime_probabilities' in pattern_outputs:
                    batch_pred['regime_probs'] = pattern_outputs['regime_probabilities'].cpu().numpy()
                
                batch_targets = {k: v.cpu().numpy() for k, v in targets.items()}
                
                all_predictions.append(batch_pred)
                all_targets.append(batch_targets)
        
        # Generate evaluation report
        report = self._generate_evaluation_report(all_predictions, all_targets)
        
        if return_predictions:
            return {
                'metrics': report,
                'predictions': all_predictions,
                'targets': all_targets
            }
        else:
            return report
    
    def _generate_evaluation_report(self, predictions: List, targets: List):
        """평가 리포트 생성"""
        # Combine all predictions and targets
        combined_pred = {}
        combined_targets = {}
        
        for key in predictions[0].keys():
            combined_pred[key] = np.concatenate([p[key] for p in predictions])
        
        for key in targets[0].keys():
            combined_targets[key] = np.concatenate([t[key] for t in targets])
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        report = {}
        
        # Signal intensity metrics
        signal_true = combined_targets['sell_intensity'].squeeze()
        signal_pred = combined_pred['sell_intensity']
        
        report['signal_metrics'] = {
            'mse': mean_squared_error(signal_true, signal_pred),
            'mae': mean_absolute_error(signal_true, signal_pred),
            'r2': r2_score(signal_true, signal_pred)
        }
        
        # Volatility metrics
        vol_true = combined_targets['volatility'].squeeze()
        vol_pred = combined_pred['volatility']
        
        report['volatility_metrics'] = {
            'mse': mean_squared_error(vol_true, vol_pred),
            'mae': mean_absolute_error(vol_true, vol_pred),
            'r2': r2_score(vol_true, vol_pred)
        }
        
        # Market regime classification (if available)
        if 'regime_probs' in combined_pred:
            regime_true = combined_targets['market_regime'].squeeze().astype(int)
            regime_pred = np.argmax(combined_pred['regime_probs'], axis=1)
            
            from sklearn.metrics import accuracy_score, classification_report
            
            report['regime_metrics'] = {
                'accuracy': accuracy_score(regime_true, regime_pred),
                'classification_report': classification_report(regime_true, regime_pred)
            }
        
        # Log metrics
        self.logger.info("Evaluation Results:")
        self.logger.info(f"Signal MSE: {report['signal_metrics']['mse']:.4f}")
        self.logger.info(f"Signal R²: {report['signal_metrics']['r2']:.4f}")
        self.logger.info(f"Volatility MSE: {report['volatility_metrics']['mse']:.4f}")
        self.logger.info(f"Volatility R²: {report['volatility_metrics']['r2']:.4f}")
        
        if 'regime_metrics' in report:
            self.logger.info(f"Regime Accuracy: {report['regime_metrics']['accuracy']:.4f}")
        
        return report


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Eddie Training Pipeline')
    parser.add_argument('--config', type=str, default='default', 
                       choices=['default', 'quick', 'high_performance'],
                       help='Configuration preset')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only run evaluation')
    
    args = parser.parse_args()
    
    # Select configuration
    if args.config == 'quick':
        config = QUICK_CONFIG
    elif args.config == 'high_performance':
        config = HIGH_PERFORMANCE_CONFIG
    else:
        config = DEFAULT_CONFIG
    
    # Initialize trainer
    trainer = EddieTrainer(config, use_wandb=True)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run training or evaluation
    if args.evaluate_only:
        trainer.evaluate()
    else:
        trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main() 