import torch
import torch.nn as nn


class HybridCNNLSTMNetwork(nn.Module):
    """Hybrid CNN-LSTM network combining convolutional and recurrent processing"""
    
    def __init__(self, config):
        super(HybridCNNLSTMNetwork, self).__init__()
        self.config = config
        
        # Multi-scale CNN for local pattern extraction
        self.cnn_branches = nn.ModuleList([
            self._create_cnn_branch(config.num_stock_features, kernel_size)
            for kernel_size in config.cnn_scales
        ])
        
        # Combine CNN outputs
        cnn_output_size = len(config.cnn_scales) * 64
        self.cnn_combiner = nn.Sequential(
            nn.Linear(cnn_output_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Attention mechanism for LSTM outputs
        self.lstm_attention = nn.MultiheadAttention(256, 8, batch_first=True)
        
        # Portfolio branch
        self.portfolio_branch = nn.Sequential(
            nn.Linear(config.num_portfolio_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64)
        )
        
        # Combined processing
        combined_size = 256 + 64  # LSTM output + portfolio
        
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )
        
        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, config.num_actions)
        )
        
        self._initialize_weights()
    
    def _create_cnn_branch(self, in_channels, kernel_size):
        """Create CNN branch for specific kernel size"""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveMaxPool1d(1)
        )
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                if hasattr(module, 'weight'):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Split stock and portfolio data
        stock_data = x[:, :, :self.config.num_stock_features]
        portfolio_data = x[:, 0, self.config.num_stock_features:]
        
        # CNN processing for each timestep
        cnn_outputs = []
        for t in range(seq_len):
            timestep_data = stock_data[:, t, :].unsqueeze(2)  # (batch, features, 1)
            
            # Multi-scale CNN
            scale_features = []
            for cnn_branch in self.cnn_branches:
                features = cnn_branch(timestep_data)  # (batch, 64, 1) - timestep_data is already (batch, features, 1)
                features = features.squeeze(-1)  # (batch, 64)
                scale_features.append(features)
            
            # Combine scales
            combined = torch.cat(scale_features, dim=1)  # (batch, 64 * num_scales)
            combined = self.cnn_combiner(combined)  # (batch, 128)
            cnn_outputs.append(combined)
        
        # Stack CNN outputs for LSTM
        cnn_sequence = torch.stack(cnn_outputs, dim=1)  # (batch, seq_len, 128)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_sequence)  # (batch, seq_len, 256)
        
        # Attention pooling
        pooled_features, _ = self.lstm_attention(lstm_out, lstm_out, lstm_out)
        pooled_features = pooled_features.mean(dim=1)  # (batch, 256)
        
        # Portfolio processing
        portfolio_features = self.portfolio_branch(portfolio_data)  # (batch, 64)
        
        # Combine features
        combined_features = torch.cat([pooled_features, portfolio_features], dim=1)
        
        # Shared processing
        shared_features = self.shared_layers(combined_features)
        
        # Dueling streams
        value = self.value_stream(shared_features)
        advantages = self.advantage_stream(shared_features)
        
        # Dueling combination
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values 