import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import FinancialTransformerBlock


class ImprovedDuelingNetwork(nn.Module):
    """Enhanced Dueling DQN with Transformer and financial-specific improvements"""
    
    def __init__(self, config):
        super(ImprovedDuelingNetwork, self).__init__()
        self.config = config
        
        # Feature embedding for stock data
        self.feature_embedding = nn.Linear(config.num_stock_features, 128)
        
        # Positional encoding for time awareness
        self.pos_encoding = nn.Parameter(torch.randn(config.window_size, 128) * 0.02)
        
        # Multi-scale CNN branch (parallel processing at different scales)
        self.multiscale_cnn = nn.ModuleList([
            self._create_cnn_branch(128, [3, 5, 7][i], f'scale_{i}') 
            for i in range(3)
        ])
        
        # Transformer blocks for temporal modeling
        self.transformer_blocks = nn.ModuleList([
            FinancialTransformerBlock(128, nhead=8, dropout=0.1)
            for _ in range(2)
        ])
        
        # Attention pooling instead of max/average pooling
        self.attention_pool = nn.MultiheadAttention(128, 4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 128))
        
        # Portfolio branch
        self.portfolio_branch = nn.Sequential(
            nn.Linear(config.num_portfolio_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # Combined features: multiscale (128*3) + transformer (128) + portfolio (64) = 576
        combined_size = 128 * 3 + 128 + 64
        
        # Shared layers with residual connections
        self.shared_layers = nn.ModuleList([
            nn.Linear(combined_size, config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size)
        ])
        
        self.shared_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(3)
        ])
        
        # Value stream with improved architecture
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1)
        )
        
        # Advantage stream with improved architecture
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, config.num_actions)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_cnn_branch(self, in_channels, kernel_size, name):
        """Create a single-scale CNN branch"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
    
    def _initialize_weights(self):
        """Improved weight initialization for financial networks"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module in [self.value_stream[-1], self.advantage_stream[-1]]:
                    # Small initialization for output layers
                    nn.init.uniform_(module.weight, -3e-4, 3e-4)
                    nn.init.constant_(module.bias, 0)
                else:
                    # He initialization for hidden layers
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Split stock data and portfolio data
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, 0, self.config.num_stock_features:]
        
        # ========== Stock Data Processing ==========
        
        # Feature embedding
        embedded = self.feature_embedding(stock_data)  # (batch_size, window_size, 128)
        
        # Add positional encoding for time awareness
        embedded = embedded + self.pos_encoding.unsqueeze(0)
        
        # Multi-scale CNN processing
        multiscale_features = []
        stock_data_cnn = embedded.transpose(1, 2)  # (batch_size, 128, window_size)
        
        for cnn_branch in self.multiscale_cnn:
            features = cnn_branch(stock_data_cnn)  # (batch_size, 128, 1)
            features = features.squeeze(-1)  # (batch_size, 128)
            multiscale_features.append(features)
        
        # Transformer processing for temporal dependencies
        transformer_out = embedded
        for transformer_block in self.transformer_blocks:
            transformer_out = transformer_block(transformer_out)
        
        # Attention pooling for transformer features
        query = self.pool_query.expand(batch_size, -1, -1)  # (batch_size, 1, 128)
        pooled_features, _ = self.attention_pool(query, transformer_out, transformer_out)
        pooled_features = pooled_features.squeeze(1)  # (batch_size, 128)
        
        # ========== Portfolio Data Processing ==========
        portfolio_features = self.portfolio_branch(portfolio_data)
        
        # ========== Feature Combination ==========
        combined_features = torch.cat([
            *multiscale_features,  # 3 x 128 = 384
            pooled_features,       # 128
            portfolio_features     # 64
        ], dim=1)  # (batch_size, 576)
        
        # ========== Shared Processing with Residuals ==========
        x = combined_features
        for i, (linear, norm) in enumerate(zip(self.shared_layers, self.shared_norms)):
            if i == 0:
                # First layer (no residual)
                x = F.gelu(norm(linear(x)))
            else:
                # Subsequent layers with residual connections
                residual = x
                x = linear(x)
                if x.size() == residual.size():  # Only add residual if dimensions match
                    x = x + residual
                x = F.gelu(norm(x))
        
        # ========== Dueling Streams ==========
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Dueling combination with improved numerical stability
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values 