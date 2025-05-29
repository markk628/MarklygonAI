"""
Signal Generator: TCN + Attention Integration
매도 강도 예측을 위한 통합 모델
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path

from .tcn import EnhancedTCN, TCNWithUncertainty
from .attention import (
    TemporalMultiHeadAttention, 
    AdaptiveAttention, 
    AttentionPooling,
    CrossTimeAttention
)


class SignalGenerator(nn.Module):
    """
    Signal Generator: TCN + Attention 통합 모델
    
    Architecture:
    Input (TA-Lib Indicators) -> TCN Feature Extraction -> 
    Multi-Head Attention -> Signal Prediction [-1, 1]
    
    매도 강도를 [-1, 1] 범위로 예측:
    - -1: 강한 매도 신호
    -  0: 중립 (HOLD)
    - +1: 강한 매수 신호
    """
    
    def __init__(
        self,
        num_inputs: int = 30,  # TA-Lib 지표 수
        tcn_channels: list = None,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.2,
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        seq_len: int = 60,
        use_uncertainty: bool = True,
        use_adaptive_attention: bool = True,
        use_cross_time: bool = True
    ):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.seq_len = seq_len
        self.use_uncertainty = use_uncertainty
        self.use_adaptive_attention = use_adaptive_attention
        self.use_cross_time = use_cross_time
        
        if tcn_channels is None:
            tcn_channels = [64, 128, 256, 128, 64]
        
        # TCN Feature Extractor
        if use_uncertainty:
            self.tcn = TCNWithUncertainty(
                num_inputs=num_inputs,
                num_channels=tcn_channels,
                kernel_size=tcn_kernel_size,
                dropout=tcn_dropout,
                uncertainty_samples=10
            )
        else:
            self.tcn = EnhancedTCN(
                num_inputs=num_inputs,
                num_channels=tcn_channels,
                kernel_size=tcn_kernel_size,
                dropout=tcn_dropout,
                use_global_context=True,
                use_multi_scale=True
            )
        
        self.feature_dim = tcn_channels[-1]
        
        # Attention Modules
        attention_modules = []
        
        # 1. Temporal Attention (시간적 패턴 학습)
        self.temporal_attention = TemporalMultiHeadAttention(
            d_model=self.feature_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            temporal_decay=0.1
        )
        
        # 2. Adaptive Attention (시장 상황 적응)
        if use_adaptive_attention:
            self.adaptive_attention = AdaptiveAttention(
                d_model=self.feature_dim,
                num_heads=attention_heads,
                num_market_states=4,
                dropout=attention_dropout
            )
        
        # 3. Cross-Time Attention (시간 구간 비교)
        if use_cross_time:
            self.cross_time_attention = CrossTimeAttention(
                d_model=self.feature_dim,
                num_heads=attention_heads,
                time_window=seq_len,
                num_windows=4,
                dropout=attention_dropout
            )
        
        # Feature Fusion
        fusion_input_dim = self.feature_dim
        if use_adaptive_attention:
            fusion_input_dim += self.feature_dim
        if use_cross_time:
            fusion_input_dim += self.feature_dim
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention Pooling (시퀀스 요약)
        self.attention_pooling = AttentionPooling(self.feature_dim)
        
        # Signal Head (매도 강도 예측)
        self.signal_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim // 4, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Uncertainty Head (모델 신뢰도)
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 4),
                nn.ReLU(),
                nn.Linear(self.feature_dim // 4, 1),
                nn.Sigmoid()  # Uncertainty in [0, 1]
            )
        
        # Market Regime Classifier (시장 상태 분류)
        self.regime_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, 4),  # 4 market regimes
            nn.Softmax(dim=-1)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, num_inputs) - TA-Lib indicators
            return_features: Whether to return intermediate features
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
            - sell_intensity: (batch_size,) - Main signal [-1, 1]
            - uncertainty: (batch_size,) - Model confidence [0, 1]
            - market_regime: (batch_size, 4) - Market state probabilities
            - features: (batch_size, feature_dim) - If return_features=True
        """
        batch_size, seq_len, num_inputs = x.size()
        
        # Transpose for TCN: (batch_size, num_inputs, seq_len)
        x_tcn = x.transpose(1, 2)
        
        # 1. TCN Feature Extraction
        if self.use_uncertainty and isinstance(self.tcn, TCNWithUncertainty):
            if self.training:
                tcn_features = self.tcn(x_tcn, return_uncertainty=False)
            else:
                tcn_features, tcn_uncertainty = self.tcn(x_tcn, return_uncertainty=True)
        else:
            tcn_features = self.tcn(x_tcn)
        
        # Transpose back for attention: (batch_size, seq_len, feature_dim)
        tcn_features = tcn_features.transpose(1, 2)
        
        # 2. Attention Processing
        attention_outputs = []
        
        # Temporal Attention
        temporal_output = self.temporal_attention(tcn_features)
        attention_outputs.append(temporal_output)
        
        # Adaptive Attention
        if self.use_adaptive_attention:
            adaptive_output = self.adaptive_attention(tcn_features)
            attention_outputs.append(adaptive_output)
        
        # Cross-Time Attention  
        if self.use_cross_time:
            cross_time_output = self.cross_time_attention(tcn_features)
            attention_outputs.append(cross_time_output)
        
        # 3. Feature Fusion
        if len(attention_outputs) > 1:
            # Concatenate different attention outputs
            fused_features = torch.cat(attention_outputs, dim=-1)
            fused_features = self.feature_fusion(fused_features)
        else:
            fused_features = attention_outputs[0]
        
        # 4. Sequence Summarization via Attention Pooling
        pooled_features = self.attention_pooling(fused_features)
        
        # 5. Signal Generation
        sell_intensity = self.signal_head(pooled_features).squeeze(-1)
        
        # 6. Additional Outputs
        results = {
            'sell_intensity': sell_intensity
        }
        
        # Uncertainty estimation
        if self.use_uncertainty:
            if hasattr(self, 'uncertainty_head'):
                uncertainty = self.uncertainty_head(pooled_features).squeeze(-1)
            else:
                # Use TCN uncertainty if available
                if not self.training and 'tcn_uncertainty' in locals():
                    uncertainty = tcn_uncertainty.mean(dim=(1, 2))  # Average over seq_len and features
                else:
                    uncertainty = torch.zeros_like(sell_intensity)
            results['uncertainty'] = uncertainty
        
        # Market regime classification
        market_regime = self.regime_classifier(pooled_features)
        results['market_regime'] = market_regime
        
        # Optional returns
        if return_features:
            results['features'] = pooled_features
            results['tcn_features'] = tcn_features
            results['fused_features'] = fused_features
        
        if return_attention:
            # Return attention weights from each module
            results['attention_weights'] = {}
        
        return results
    
    def predict_sell_intensity(self, x: torch.Tensor) -> torch.Tensor:
        """단순 매도 강도 예측 (추론용)"""
        self.eval()
        with torch.no_grad():
            results = self.forward(x)
            return results['sell_intensity']
    
    def get_model_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """모델 불확실성 추정 (추론용)"""
        self.eval()
        with torch.no_grad():
            results = self.forward(x)
            return results.get('uncertainty', torch.zeros(x.size(0)))
    
    def get_market_regime(self, x: torch.Tensor) -> torch.Tensor:
        """시장 레짐 분류 (추론용)"""
        self.eval()
        with torch.no_grad():
            results = self.forward(x)
            return results['market_regime']


class SignalGeneratorTrainer:
    """Signal Generator 훈련 클래스"""
    
    def __init__(
        self,
        model: SignalGenerator,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Loss functions
        self.signal_loss_fn = nn.MSELoss()
        self.regime_loss_fn = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Loss computation with multiple objectives
        
        Args:
            outputs: Model outputs dictionary
            targets: Target values dictionary
            
        Returns:
            total_loss: Combined loss
            loss_components: Individual loss values
        """
        losses = {}
        
        # 1. Signal prediction loss
        signal_loss = self.signal_loss_fn(
            outputs['sell_intensity'], 
            targets['sell_intensity']
        )
        losses['signal'] = signal_loss
        
        # 2. Market regime classification loss (if available)
        if 'market_regime' in targets:
            regime_loss = self.regime_loss_fn(
                outputs['market_regime'],
                targets['market_regime']
            )
            losses['regime'] = regime_loss
        
        # 3. Uncertainty regularization (encourage confident predictions)
        if 'uncertainty' in outputs:
            uncertainty_reg = outputs['uncertainty'].mean()
            losses['uncertainty_reg'] = uncertainty_reg
        
        # 4. Combine losses with weights
        total_loss = (
            1.0 * losses['signal'] +
            0.1 * losses.get('regime', 0) +
            0.01 * losses.get('uncertainty_reg', 0)
        )
        
        # Convert to float for logging
        loss_components = {k: v.item() if isinstance(v, torch.Tensor) else v 
                          for k, v in losses.items()}
        
        return total_loss, loss_components
    
    def train_epoch(self, train_loader) -> float:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            inputs = batch['features'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != 'features'}
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            loss, loss_components = self.compute_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, val_loader) -> float:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['features'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch.items() if k != 'features'}
                
                outputs = self.model(inputs)
                loss, _ = self.compute_loss(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # Update scheduler
        self.scheduler.step(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, path: Path, epoch: int, best_loss: float):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        return checkpoint['epoch'], checkpoint['best_loss']


def test_signal_generator():
    """Signal Generator 테스트"""
    print("Testing Signal Generator...")
    
    # Test parameters
    batch_size, seq_len, num_inputs = 4, 60, 30
    device = torch.device('cpu')
    
    # Create test data
    x = torch.randn(batch_size, seq_len, num_inputs)
    targets = {
        'sell_intensity': torch.randn(batch_size),
        'market_regime': torch.randint(0, 4, (batch_size,))
    }
    
    # Create model
    model = SignalGenerator(
        num_inputs=num_inputs,
        seq_len=seq_len,
        use_uncertainty=True,
        use_adaptive_attention=True,
        use_cross_time=True
    )
    
    # Test forward pass
    outputs = model(x, return_features=True)
    
    print(f"Sell intensity shape: {outputs['sell_intensity'].shape}")
    print(f"Uncertainty shape: {outputs['uncertainty'].shape}")
    print(f"Market regime shape: {outputs['market_regime'].shape}")
    print(f"Features shape: {outputs['features'].shape}")
    
    # Check output ranges
    assert torch.all(outputs['sell_intensity'] >= -1) and torch.all(outputs['sell_intensity'] <= 1)
    assert torch.all(outputs['uncertainty'] >= 0) and torch.all(outputs['uncertainty'] <= 1)
    assert torch.allclose(outputs['market_regime'].sum(dim=1), torch.ones(batch_size))
    
    # Test trainer
    trainer = SignalGeneratorTrainer(model, device)
    loss, loss_components = trainer.compute_loss(outputs, targets)
    
    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {loss_components}")
    
    print("Signal Generator test passed!")


if __name__ == "__main__":
    test_signal_generator() 