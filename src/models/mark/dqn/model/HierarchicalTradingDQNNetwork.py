import torch
import torch.nn as nn

from src.config.config import BATCH_SIZE
from src.models.mark.dqn.model.HelperLayers import *


class HierarchicalTradingDQNNetwork(nn.Module):
    def __init__(self, sizes: dict[str, int], use_dueling: bool=True):
        super(HierarchicalTradingDQNNetwork, self).__init__()
        
        self.stock_data_window_size = sizes['stock_data_window_size']
        self.stock_data_feature_size = sizes['stock_data_feature_size']
        self.portfolio_metrics_size = sizes['portfolio_metrics_size']
        self.performance_metrics_size = sizes['performance_metrics_size']
        self.risk_metrics_size = sizes['risk_metrics_size']
        self.market_state_metrics_size = sizes['market_state_metrics_size']
        self.position_management_metrics_size = sizes['position_management_metrics_size']
        self.trading_behavior_metrics_size = sizes['trading_behavior_metrics_size']
        self.temporal_metrics_size = sizes['temporal_metrics_size']
        self.temporal_metrics_types_count = sizes['temporal_metrics_types_count']
        self.action_size = sizes['action_size']
        self.use_dueling = use_dueling
        
        # Simplified stock data processing
        self.stock_data_conv1d = nn.Sequential(
            nn.Conv1d(self.stock_data_feature_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Simplified auxiliary features processing
        self.auxiliary_net = nn.Sequential(
            nn.Linear(self.portfolio_metrics_size + 
                     self.performance_metrics_size + 
                     self.risk_metrics_size + 
                     self.market_state_metrics_size + 
                     self.position_management_metrics_size + 
                     self.trading_behavior_metrics_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Simplified temporal features processing
        self.temporal_net = nn.Sequential(
            nn.Linear(self.temporal_metrics_size * self.temporal_metrics_types_count, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Combined features dimension
        combined_size = 64 + 64 + 16  # stock_data + auxiliary + temporal
        
        if use_dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(combined_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(combined_size, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_size)
            )
        else:
            self.q_net = nn.Sequential(
                nn.Linear(combined_size, 64),
                nn.ReLU(),
                nn.Linear(64, self.action_size)
            )
            
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Custom weight initialization for different layer types"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Process stock data
        stock_data = x[:, :self.stock_data_window_size * self.stock_data_feature_size]
        stock_data = stock_data.view(-1, self.stock_data_feature_size, self.stock_data_window_size)
        stock_features = self.stock_data_conv1d(stock_data).squeeze(-1)
        
        # Process auxiliary features
        aux_start = self.stock_data_window_size * self.stock_data_feature_size
        temporal_start = aux_start + self.portfolio_metrics_size + self.performance_metrics_size + \
                        self.risk_metrics_size + self.market_state_metrics_size + \
                        self.position_management_metrics_size + self.trading_behavior_metrics_size
        
        auxiliary_features = x[:, aux_start:temporal_start]
        auxiliary_features = self.auxiliary_net(auxiliary_features)
        
        # Process temporal features
        temporal_features = x[:, temporal_start:]
        temporal_features = self.temporal_net(temporal_features)
        
        # Combine features
        combined = torch.cat([stock_features, auxiliary_features, temporal_features], dim=1)
        
        if self.use_dueling:
            value = self.value_stream(combined)
            advantage = self.advantage_stream(combined)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            return self.q_net(combined)
    
if __name__=='__main__':
    sizes = {
        'stock_data_window_size': 60,
        'stock_data_feature_size': 22,
        'portfolio_metrics_size': 6,
        'performance_metrics_size': 9,
        'risk_metrics_size': 4,
        'market_state_metrics_size': 14,
        'position_management_metrics_size': 9,
        'trading_behavior_metrics_size': 7,
        'temporal_metrics_size': 2,
        'temporal_metrics_types_count': 3,
        'action_size': 3
    }
    dqn = HierarchicalTradingDQNNetwork(sizes)
    print(dqn)

    # dummy data
    stock_data_flattened_dim = sizes['stock_data_window_size'] * sizes['stock_data_feature_size']
    temporal_states_dim = sizes['temporal_metrics_size'] * 3
    everything_else =  sizes['portfolio_metrics_size'] + sizes['performance_metrics_size'] + sizes['risk_metrics_size'] + sizes['market_state_metrics_size'] + sizes['position_management_metrics_size'] + sizes['trading_behavior_metrics_size'] + temporal_states_dim
    dummy_input = torch.randn(BATCH_SIZE, stock_data_flattened_dim + everything_else) 
    print(dummy_input.shape)
    
    output = dqn(dummy_input)
    print("Output Q-values shape:", output.shape)
    assert output.shape == (BATCH_SIZE, sizes['action_size'])