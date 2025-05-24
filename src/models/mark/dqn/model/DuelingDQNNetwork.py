import torch
import torch.nn as nn

from src.config.config import BATCH_SIZE

class DuelingDQNNetwork(nn.Module):
    def __init__(self, sizes):
        super(DuelingDQNNetwork, self).__init__()

        self.market_data_timesteps = sizes['stock_data_window_size']
        self.market_data_features = sizes['stock_data_feature_size']
        self.portfolio_info_size = sizes['portfolio_metrics_size']
        self.market_info_size_flat = sizes['market_state_metrics_size']
        self.constraint_size = sizes['constraint_metrics_size']
        self.action_size = sizes['action_size']

        # input dim for conv1d: (batch_size, in_channels, sequence_length)
        # in_channels: tock_data_feature_size (22)
        # sequence_length: stock_data_window_size (60)
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(in_channels=self.market_data_features, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.AdaptiveMaxPool1d(1) # pool to get features ready for Linear layer
        )
        
        # out_channels of the last conv1d layer
        self.processed_market_data_size = 256 
        
        self.state_linear_layers = nn.Sequential(
            nn.Linear(self.processed_market_data_size, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.portfolio_layers = nn.Sequential(
            nn.Linear(self.portfolio_info_size, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.market_layers = nn.Sequential(
            nn.Linear(self.market_info_size_flat, 64), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.constraint_layers = nn.Sequential(
            nn.Linear(self.constraint_size, 64), 
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # dueling DQN specific layers
        # input size for both value and advantage
        # output of state_linear_layers (64) + portfolio_layers (32) + market_layers (16) + constraint_layers (16) = 128
        self.combined_feature_size = 64 + 32 + 16 + 16
        
        # value stream: estimates the state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(self.combined_feature_size, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        # advantage stream: estimates the advantage A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.combined_feature_size, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, self.action_size)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initializes weights using Kaiming Normal and biases to zero.
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # extract and reshape market data, then process through covn1d -> linear layer
        # input dim (batch_size, num_channels, sequence_length)
        # output dim (batch_size, processed_market_data_size, 1)
        # squeeze the last dimension to make final dim (batch_size, processed_market_data_size)
        market_data_flat = x[:, :self.market_data_timesteps * self.market_data_features]
        market_data_reshaped = market_data_flat.view(
            -1, self.market_data_features, self.market_data_timesteps
        )
        conv_output = self.conv1d_layers(market_data_reshaped).squeeze(-1)
        stock_data_features = self.state_linear_layers(conv_output)

        # portfolio_metrics
        portfolio_start_idx = self.market_data_timesteps * self.market_data_features
        portfolio_features = self.portfolio_layers(x[:, portfolio_start_idx : portfolio_start_idx + self.portfolio_info_size])
        
        # market_state_metrics
        market_info_flat_start_idx = portfolio_start_idx + self.portfolio_info_size
        market_features = self.market_layers(x[:, market_info_flat_start_idx : market_info_flat_start_idx + self.market_info_size_flat])
        
        # constraint_metrics
        constraint_start_idx = market_info_flat_start_idx + self.market_info_size_flat
        constraint_features = self.constraint_layers(x[:, constraint_start_idx:])

        # concatenate the outputs
        combined_features = torch.cat([stock_data_features, portfolio_features, market_features, constraint_features], dim=1)

        # pass combined features through Value and Advantage streams
        value = self.value_stream(combined_features)
        advantage = self.advantage_stream(combined_features)

        # combine Value and Advantage to get q-values
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
    
if __name__=='__main__':
    sizes = {
        'stock_data_window_size': 60,
        'stock_data_feature_size': 22,
        'portfolio_metrics_size': 28,
        'market_state_metrics_size': 11,
        'constraint_metrics_size': 7,
        'action_size': 3
    }
    dqn = DuelingDQNNetwork(sizes)
    print(dqn)

    # dummy data
    stock_data_flattened_dim = sizes['stock_data_window_size'] * sizes['stock_data_feature_size']
    everything_else = sizes['portfolio_metrics_size'] + sizes['market_state_metrics_size'] + sizes['constraint_metrics_size']
    dummy_input = torch.randn(BATCH_SIZE, stock_data_flattened_dim + everything_else) 
    print(dummy_input.shape)
    
    output = dqn(dummy_input)
    print("Output Q-values shape:", output.shape) # Should be (batch_size, action_size)
    assert output.shape == (BATCH_SIZE, sizes['action_size'])