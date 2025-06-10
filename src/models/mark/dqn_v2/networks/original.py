import torch
import torch.nn as nn


class DuelingNetworkOriginal(nn.Module):
    """Original Dueling DQN architecture"""
    
    def __init__(self, config):
        super(DuelingNetworkOriginal, self).__init__()
        self.config = config
        
        # Stock data branch - 1D CNN for time series processing
        # Input: (batch_size, window_size, self.config.num_stock_features)
        self.stock_data_branch = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_stock_features, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1), # prevents overfitting and curse of dimensionality
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.1),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Portfolio state branch - Fully connected layers
        # Input: (batch_size, self.config.num_portfolio_features) - portfolio features
        self.portfolio_branch = nn.Sequential(
            nn.Linear(self.config.num_portfolio_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # Combined features: 256 (stock data branch output) + 64 (portfolio branch output) = 320
        combined_size = 256 + 64
        
        # Shared layers after combining branches
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.num_actions)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization for ReLU networks"""
        # Initialize stock data branch
        for layer in self.stock_data_branch:
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
        
        # Initialize portfolio branch
        for layer in self.portfolio_branch:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
        
        # Initialize shared layers
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.01)
        
        # Initialize value stream
        for layer in self.value_stream:
            if isinstance(layer, nn.Linear):
                if layer == self.value_stream[-1]:
                    nn.init.uniform_(layer.weight, -3e-4, 3e-4)
                    nn.init.constant_(layer.bias, 0)
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.01)
        
        # Initialize advantage stream
        for layer in self.advantage_stream:
            if isinstance(layer, nn.Linear):
                if layer == self.advantage_stream[-1]:
                    nn.init.uniform_(layer.weight, -3e-4, 3e-4)
                    nn.init.constant_(layer.bias, 0)
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining value and advantage streams"""        
        # Split stock data and portfolio data
        stock_data = x[:, :, :self.config.num_stock_features]  # (batch_size, window_size, self.config.num_stock_features)
        portfolio_data = x[:, 0, self.config.num_stock_features:]  # (batch_size, self.config.num_portfolio_features)
        
        # Process stock data through CNN
        # Conv1d expects (batch_size, channels, window_size)
        stock_data = stock_data.permute(0, 2, 1)  # (batch_size, 40, window_size)
        stock_data_features = self.stock_data_branch(stock_data)  # (batch_size, 256, 1)
        stock_data_features = stock_data_features.squeeze(-1)  # (batch_size, 256)
        
        # Process portfolio data through FC layers
        portfolio_features = self.portfolio_branch(portfolio_data)  # (batch_size, 64)
        
        # Combine features
        combined_features = torch.cat([stock_data_features, portfolio_features], dim=1)  # (batch_size, 320)
        
        # Process through shared layers
        shared_features = self.shared_layers(combined_features)  # (batch_size, hidden_size)
        
        # Compute value and advantages
        value = self.value_stream(shared_features)
        advantages = self.advantage_stream(shared_features)
        
        # Combine using dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values 