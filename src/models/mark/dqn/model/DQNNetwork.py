import torch
import torch.nn as nn

from src.config.config import BATCH_SIZE


class DQNNetwork(nn.Module):
    def __init__(self, sizes: dict[str, int], use_dueling: bool=True):
        super(DQNNetwork, self).__init__()
        
        self.stock_data_window_size = sizes['stock_data_window_size']
        self.stock_data_feature_size = sizes['stock_data_feature_size']
        self.portfolio_metrics_size = sizes['portfolio_metrics_size']
        self.market_state_metrics_size = sizes['market_state_metrics_size']
        self.constraint_metrics_size = sizes['constraint_metrics_size']
        self.action_size = sizes['action_size']
        self.use_dueling = use_dueling
        
        # input dim for conv1d: (batch_size, in_channels, sequence_length)
        # in_channels: tock_data_feature_size (22)
        # sequence_length: stock_data_window_size (60)
        self.stock_data_conv1d_branch = nn.Sequential(
            nn.Conv1d(in_channels=self.stock_data_feature_size, out_channels=64, kernel_size=3, padding=1),
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
        
        self.stock_data_linear_branch = nn.Sequential(
            nn.Linear(self.processed_market_data_size, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        self.portfolio_branch = nn.Sequential(
            nn.Linear(self.portfolio_metrics_size, 256),
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
        
        self.market_branch = nn.Sequential(
            nn.Linear(self.market_state_metrics_size, 64), 
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

        self.constraint_branch = nn.Sequential(
            nn.Linear(self.constraint_metrics_size, 64), 
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
        
        
        # output of state_layers (64) + portfolio_branch (32) + market_branch (16) + of constraint_branch (16) = 128
        combined_feature_size = 64 + 32 + 16 + 16
        if use_dueling:
             # value stream: estimates the state value V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(combined_feature_size, 64),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )

            # advantage stream: estimates the advantage A(s, a) for each action
            self.advantage_stream = nn.Sequential(
                nn.Linear(combined_feature_size, 64),
                nn.LeakyReLU(negative_slope=0.01),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, self.action_size)
            )
        else:
            # action layer
            self.action_layer = nn.Sequential(
                nn.Linear(combined_feature_size, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                nn.Linear(64, self.action_size)
            )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initializes weights using Kaiming Normal and biases to zero.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # indices for state components
        portfolio_start_idx = self.stock_data_window_size * self.stock_data_feature_size
        market_state_start_idx = portfolio_start_idx + self.portfolio_metrics_size
        constraint_start_idx = market_state_start_idx + self.market_state_metrics_size
        
        # process stock data
        # extract and reshape market data, then process through covn1d -> linear layer
        # input dim (batch_size, num_channels, sequence_length)
        # output dim (batch_size, processed_market_data_size, 1)
        # squeeze the last dimension to make final dim (batch_size, processed_market_data_size)
        stock_data_flat = x[:, :portfolio_start_idx]
        stock_data_reshaped = stock_data_flat.view(-1, self.stock_data_feature_size, self.stock_data_window_size)
        conv_output = self.stock_data_conv1d_branch(stock_data_reshaped).squeeze(-1)
        
        # process the rest
        stock_data_features  = self.stock_data_linear_branch(conv_output)
        portfolio_features = self.portfolio_branch(x[:, portfolio_start_idx: market_state_start_idx])
        market_features = self.market_branch(x[:, market_state_start_idx: constraint_start_idx])
        constraint_features = self.constraint_branch(x[:, constraint_start_idx:])
        combined_features = torch.cat([stock_data_features , portfolio_features, market_features, constraint_features], dim=1)

        if self.use_dueling:
            # pass combined features through Value and Advantage streams
            # combine Value and Advantage to get q-values
            # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
            value = self.value_stream(combined_features)
            advantage = self.advantage_stream(combined_features)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        return self.action_layer(combined_features)
    
if __name__=='__main__':
    sizes = {
        'stock_data_window_size': 60,
        'stock_data_feature_size': 22,
        'portfolio_metrics_size': 28,
        'market_state_metrics_size': 11,
        'constraint_metrics_size': 7,
        'action_size': 3
    }
    dqn = DQNNetwork(sizes)
    print(dqn)

    stock_data_flattened_dim = sizes['stock_data_window_size'] * sizes['stock_data_feature_size']
    everything_else = sizes['portfolio_metrics_size'] + sizes['market_state_metrics_size'] + sizes['constraint_metrics_size']
    dummy_input = torch.randn(BATCH_SIZE, stock_data_flattened_dim + everything_else) 
    print(dummy_input.shape)
    
    output = dqn(dummy_input)
    print("Output Q-values shape:", output.shape)
    assert output.shape == (BATCH_SIZE, sizes['action_size'])