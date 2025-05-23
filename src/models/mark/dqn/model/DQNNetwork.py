import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn

pd.set_option('display.max_columns', None)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class DQNNetwork(nn.Module):
    def __init__(self, 
                 state_size: int,
                 constraint_size: int,
                 action_size: int):
        super(DQNNetwork, self).__init__()
        
        self.constraint_size = constraint_size
        self.state_layers = nn.Sequential(
            nn.Linear(state_size - constraint_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.constraint_layers = nn.Sequential(
            nn.Linear(constraint_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # action layer
        self.action_layer = nn.Sequential(
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, action_size)
        )
        
        # initialize optimized weights
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
        state_features = self.state_layers(x[:, :-self.constraint_size])
        constraint_features = self.constraint_layers(x[:, -self.constraint_size:])
        combined_features = torch.cat([state_features, constraint_features], dim=1)
        
        return self.action_layer(combined_features)
    
if __name__=='__main__':
    dqn = DQNNetwork(1215, 6, 3)
    print(dqn)