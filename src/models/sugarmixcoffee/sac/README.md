# SAC (Soft Actor-Critic) Stock Trading Agent

This directory contains a Soft Actor-Critic (SAC) implementation for stock trading using minute-level market data. The SAC agent is built on the same infrastructure as the DQN implementation but uses an actor-critic architecture with entropy regularization for better exploration and performance.

## Features

### Core SAC Components
- **Actor Network**: Policy network that outputs action probabilities
- **Twin Critic Networks**: Two Q-value networks to reduce overestimation bias
- **Automatic Temperature Tuning**: Adaptive entropy regularization coefficient
- **Prioritized Experience Replay**: GPU-accelerated replay buffer for efficient training
- **Soft Target Updates**: Gradual target network updates for stability

### Trading-Specific Features
- **Discrete Action Space**: Hold (0), Buy (1), Sell (2)
- **Portfolio State Integration**: Combines market data with portfolio information
- **Invalid Action Handling**: Prevents impossible trades (e.g., buying when already holding)
- **Financial Metrics**: Sharpe ratio, max drawdown, win rate calculation
- **Data Preprocessing**: Robust scaling and outlier handling

### Technical Advantages over DQN
- **Better Exploration**: Entropy regularization encourages exploration of diverse strategies
- **Stable Training**: Actor-critic architecture with soft updates reduces training instability
- **Reduced Overestimation**: Twin Q-networks mitigate Q-value overestimation
- **Continuous Learning**: No need for epsilon-greedy exploration schedule

## Architecture

### Network Structure
```
Input: [Stock Data (39 features), Portfolio Data (8 features)]
       ↓
[Stock Data Branch (1D CNN)] + [Portfolio Branch (FC)]
       ↓
[Combined Features (320 dimensions)]
       ↓
Actor Network → Action Probabilities
Critic Networks → Q-values for each action
```

### Key Hyperparameters
- **Learning Rates**: Actor: 3e-4, Critics: 3e-4, Temperature: 3e-4
- **Soft Update Coefficient (τ)**: 0.005
- **Target Entropy**: -1.0 (for discrete actions)
- **Discount Factor (γ)**: 0.99
- **Batch Size**: 64
- **Replay Buffer Size**: 100,000

## Usage

### Basic Training Example
```python
from src.models.sugarmixcoffee.sac import train_sac
import pandas as pd

# Train SAC agent
results = train_sac(
    data_path="data/feature_engineered/TSLA.csv",
    cutoff=pd.Timestamp('2024-05-06 08:00:00', tz='UTC'),
    num_episodes=100,
    use_preprocessing=True,
    scaling_method='robust',
    outlier_method='winsorize'
)

# Access results
test_results = results['test_results']
print(f"Total Return: {test_results['total_return']:.2%}")
print(f"Sharpe Ratio: {test_results['sharpe_ratio']:.2f}")
```

### Loading and Evaluating a Trained Model
```python
from src.models.sugarmixcoffee.sac import SACAgent, SACConfig

# Create agent and load trained model
config = SACConfig()
agent = SACAgent(config)
agent.load("sac_final_model.pt")

# Use for trading (in your trading environment)
action = agent.select_action(state, deterministic=True)
```

### Using the Trainer Script
```bash
cd src/models/sugarmixcoffee/sac
python trainer.py
```

### Running Examples
```bash
python example.py
```

## Configuration

### SACConfig Parameters
```python
@dataclass
class SACConfig:
    # Trading Environment
    initial_balance: float = 100000.0
    transaction_fee_pct: float = 0.001
    max_position_size: float = 0.7
    
    # SAC Learning
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    
    # Entropy Regularization
    target_entropy: float = -1.0
    alpha_auto_tune: bool = True
    
    # Data
    window_size: int = 60  # Minutes of historical data
    num_actions: int = 3   # Hold, Buy, Sell
```

## Data Requirements

### Input Data Format
The SAC agent expects CSV files with the following features:
- **Price Data**: open, high, low, close, volume
- **Technical Indicators**: RSI, MACD, Bollinger Bands, etc.
- **Volume Indicators**: OBV, volume-based features
- **Temporal Features**: Cyclical time encodings

### Preprocessing
- **Scaling**: RobustScaler (default) for financial data robustness
- **Outlier Handling**: Winsorization to cap extreme values
- **Feature Engineering**: 39 market features + 8 portfolio features

## Performance Metrics

### Financial Metrics
- **Total Return**: (Final Value - Initial Balance) / Initial Balance
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Calmar Ratio**: Return / Maximum Drawdown

### Training Metrics
- **Actor Loss**: Policy gradient loss with entropy regularization
- **Critic Loss**: TD error-based Q-learning loss
- **Temperature (α)**: Automatically tuned entropy coefficient
- **Invalid Actions**: Count of attempted impossible trades

## Comparison with DQN

| Aspect | DQN | SAC |
|--------|-----|-----|
| **Architecture** | Value-based (Q-network) | Actor-Critic |
| **Exploration** | ε-greedy | Entropy regularization |
| **Action Selection** | Discrete max Q-value | Probabilistic sampling |
| **Target Updates** | Hard updates | Soft updates |
| **Overestimation** | Single Q-network | Twin Q-networks |
| **Training Stability** | Can be unstable | Generally more stable |
| **Sample Efficiency** | Good | Excellent |

## File Structure

```
src/models/sugarmixcoffee/sac/
├── __init__.py           # Module exports
├── sac.py               # Core SAC implementation
├── trainer.py           # Training and visualization
├── example.py           # Usage examples
└── README.md           # This documentation
```

## Key Classes

### SACAgent
Main agent class that orchestrates training and inference.
- `select_action()`: Choose action given state
- `update()`: Perform one training step
- `train_episode()`: Train for one complete episode
- `save()/load()`: Model persistence

### ActorNetwork
Policy network that outputs action probabilities.
- CNN for market data processing
- FC layers for portfolio state
- Outputs logits for discrete action distribution

### CriticNetwork
Q-value estimation networks (two instances for twin Q-learning).
- Same architecture as Actor but outputs Q-values
- Used for both current Q-values and target Q-values

## Training Tips

### Hyperparameter Tuning
1. **Learning Rates**: Start with 3e-4 for all networks
2. **Target Entropy**: Use -log(num_actions) as starting point
3. **Soft Update Rate**: 0.005 works well for most cases
4. **Batch Size**: Larger batches (64-128) generally better

### Data Considerations
1. **Window Size**: 60 minutes captures short-term patterns
2. **Data Quality**: Ensure no missing values or extreme outliers
3. **Feature Engineering**: Include diverse technical indicators
4. **Time Period**: Use recent data for better relevance

### Training Stability
1. **Gradient Clipping**: Prevents exploding gradients
2. **Learning Rate Scheduling**: Reduce LR when validation plateaus
3. **Early Stopping**: Prevent overfitting with patience mechanism
4. **Regular Evaluation**: Monitor validation performance

## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Scikit-learn >= 0.24.0

## Citation

If you use this SAC implementation in your research, please cite:

```bibtex
@software{marklygon_sac,
  title={SAC Stock Trading Agent},
  author={MarklygonAI},
  year={2024},
  url={https://github.com/marklygon/sac-trading}
}
```

## License

This implementation is part of the MarklygonAI trading system. Please refer to the main project license for usage terms.

## Future Improvements

- [ ] Multi-asset trading support
- [ ] Continuous action space for position sizing
- [ ] Integration with live trading APIs
- [ ] Advanced reward shaping techniques
- [ ] Ensemble methods with multiple SAC agents
- [ ] Hierarchical reinforcement learning for strategy selection 