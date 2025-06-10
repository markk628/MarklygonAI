# DQN v5 Reward Parameter Optimization

This directory contains a complete Optuna-based optimization system for finding the optimal reward parameters for the DQN v5 trading agent.

## 🚀 Quick Start

### 1. Run Basic Optimization
```bash
# Quick test (10 trials, 25 episodes each)
python dqn_v5_optimize_rewards.py --data-path "data/TSLA_1min_features.csv" --cutoff "2023-01-01" --trials 10 --episodes-per-trial 25

# Thorough optimization (50 trials, 50 episodes each)
python dqn_v5_optimize_rewards.py --data-path "data/TSLA_1min_features.csv" --cutoff "2023-01-01" --trials 50 --episodes-per-trial 50
```

### 2. Apply Optimized Parameters
```bash
# View results and generate code
python apply_optimized_rewards.py --config best_dqn_v5_config.json

# Save generated code to file
python apply_optimized_rewards.py --config best_dqn_v5_config.json --output optimized_config.py
```

### 3. Use Examples
```bash
# Interactive examples
python reward_optimization_example.py
```

## 📊 What Gets Optimized

The system optimizes these key parameters:

### Core Reward Parameters
- **`invalid_penalty`** (0.05-0.8): Penalty for invalid actions
- **`portfolio_scaling`** (0.01-0.3): Scaling factor for portfolio value changes

### Trading Behavior Parameters  
- **`min_profit_threshold`** (0.001-0.05): Minimum profit threshold for trades
- **`transaction_fee_percent`** (0.0005-0.003): Transaction fee percentage

### Learning Parameters
- **`learning_rate`** (1e-5 to 5e-4): Neural network learning rate
- **`epsilon_decay`** (1000-5000): Exploration decay rate

### Network Parameters
- **`batch_size`** (32/64/128): Training batch size
- **`hidden_size`** (256/512/1024): Neural network hidden layer size

## 🎯 Fitness Function

The multi-objective fitness function balances:

```python
fitness_score = (
    results['avg_return'] * 3.0 +           # Primary: actual returns
    results['win_rate'] * 1.5 +             # Secondary: consistency  
    results['sharpe_ratio'] * 0.5 +         # Risk-adjusted performance
    -results['invalid_rate'] * 2.0 +        # Penalty: invalid actions
    -abs(results['avg_trades'] - 15) * 0.1  # Target ~15 trades/day
)
```

You can customize these weights in `dqn_v5_optimize_rewards.py`.

## 📁 Files Overview

### Core Optimization
- **`dqn_v5_optimize_rewards.py`**: Main Optuna optimization script
- **`apply_optimized_rewards.py`**: Apply optimized parameters to your model
- **`reward_optimization_example.py`**: Usage examples and tutorials

### Updated Components
- **`dqn_v5.py`**: Updated with configurable reward parameters
- **`config.py`**: Added `portfolio_scaling` and `invalid_penalty` parameters

## 🔧 How It Works

### 1. Parameter Search
The optimization runs multiple trials, each with different parameter combinations:
- Creates a `TradingConfig` with trial parameters
- Trains a DQN agent for N episodes  
- Evaluates performance metrics
- Calculates fitness score
- Optuna learns which parameters work best

### 2. Performance Evaluation
Each trial measures:
- **Returns**: Average portfolio return per episode
- **Win Rate**: Percentage of profitable episodes
- **Invalid Actions**: Number of invalid actions per episode
- **Trade Frequency**: Average trades per episode
- **Sharpe Ratio**: Risk-adjusted returns
- **Consistency**: Standard deviation of returns

### 3. Multi-Objective Optimization
The fitness function combines multiple objectives:
- **Maximize**: Returns, win rate, Sharpe ratio
- **Minimize**: Invalid actions, deviation from target trade frequency
- **Balance**: Risk vs reward, exploration vs exploitation

## 📈 Expected Results

### Good Optimized Model Should Achieve:
- **Average Return**: 0.5-2% per episode
- **Win Rate**: 60-70% of episodes profitable
- **Invalid Actions**: <20 per episode (vs 100-200+ currently)
- **Trade Frequency**: 10-20 trades per episode
- **Sharpe Ratio**: >1.0

### Parameter Ranges Typically Found:
- **`invalid_penalty`**: 0.2-0.5 (higher than default 0.1)
- **`portfolio_scaling`**: 0.05-0.15 (may be lower than 0.1)
- **`learning_rate`**: 1e-4 to 3e-4 (moderate)
- **`epsilon_decay`**: 2000-4000 (faster than very slow decay)

## 🚀 Advanced Usage

### Multi-Phase Optimization
```python
# Phase 1: Coarse search (30 trials, 25 episodes)
study1 = run_optimization(trials=30, episodes_per_trial=25)

# Phase 2: Fine search around best parameters (20 trials, 100 episodes)  
study2 = run_optimization(trials=20, episodes_per_trial=100)
```

### Custom Fitness Functions
Edit the `objective` function in `dqn_v5_optimize_rewards.py`:

```python
# For maximum returns (aggressive)
fitness_score = results['avg_return'] * 5.0 - results['invalid_rate'] * 1.0

# For risk-adjusted performance (conservative)
fitness_score = results['sharpe_ratio'] * 3.0 + results['win_rate'] * 2.0

# For high-frequency trading
fitness_score = results['avg_return'] * 2.0 + results['avg_trades'] * 0.1
```

### Persistent Studies
Optuna saves studies to SQLite database:
```python
# Resume previous study
study = optuna.create_study(
    study_name="my_optimization",
    storage='sqlite:///optimization_studies.db',
    load_if_exists=True
)
```

## 📊 Monitoring Progress

### Real-time Progress
The optimization prints progress for each trial:
```
🔍 Trial 23: Testing parameter combination...
  Invalid Penalty: 0.347
  Portfolio Scaling: 0.089
  Min Profit Threshold: 0.023
  Learning Rate: 1.84e-04
  Epsilon Decay: 3247
  Training 50 episodes...
    Episodes 1-10: Avg Return: 0.34%, Avg Invalid: 87.2
    Episodes 11-20: Avg Return: 0.71%, Avg Invalid: 52.3
  Results: Return=0.58%, Win Rate=64%, Invalid=45.7
  📊 Fitness Score: 1.847
```

### Best Results Summary
```
🏆 Best Trial #23
   Fitness Score: 1.847
   
🎯 Best Parameters:
   invalid_penalty: 0.347
   portfolio_scaling: 0.089
   
📊 Best Performance:
   Average Return: 0.58%
   Win Rate: 64%
   Invalid Actions: 45.7
```

## 🔄 Integration Workflow

1. **Run Optimization**: Find best parameters with Optuna
2. **Analyze Results**: Review parameter importance and performance
3. **Generate Code**: Use `apply_optimized_rewards.py` to create config
4. **Update Model**: Apply optimized parameters to your `TradingConfig`
5. **Full Training**: Train your final model with optimized parameters
6. **Validate**: Test on out-of-sample data

## ⚠️ Important Notes

### Optimization Best Practices
- Start with quick optimization (10-20 trials) to test the system
- Use more episodes per trial for final optimization (50-100+)
- Consider multi-phase optimization (coarse -> fine)
- Validate results on out-of-sample data

### Parameter Interactions
- `invalid_penalty` and `portfolio_scaling` interact strongly
- Higher learning rates may need lower `epsilon_decay`
- Batch size affects training stability and speed
- Hidden size impacts model capacity and overfitting

### Computational Requirements
- Each trial trains a DQN for 25-100 episodes
- 50 trials × 50 episodes = 2,500 training episodes total
- GPU recommended for reasonable optimization time
- Consider running overnight for thorough optimization

## 🎯 Troubleshooting

### Common Issues
- **NaN rewards**: Check data preprocessing and feature scaling
- **All negative scores**: Reduce penalty terms in fitness function  
- **No improvement**: Increase trial count or episode count
- **Memory errors**: Reduce batch size or hidden size

### Performance Issues
- **Slow optimization**: Use fewer episodes per trial for initial testing
- **Poor convergence**: Check learning rate and epsilon decay ranges
- **Overfitting**: Validate on different time periods

Happy optimizing! 🚀 