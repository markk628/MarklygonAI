"""
Apply Optimized Reward Parameters to DQN v5
===========================================

This script helps apply the optimized parameters from Optuna optimization
to the DQN v5 step function.

Usage:
    python apply_optimized_rewards.py --config best_dqn_v5_config.json
"""

import json
import argparse
from typing import Dict, Any


def load_optimized_config(config_path: str) -> Dict[str, Any]:
    """Load optimized configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data


def generate_step_function_code(config_data: Dict[str, Any]) -> str:
    """Generate the optimized step function code"""
    
    best_params = config_data['best_parameters']
    
    # Extract optimized parameters
    invalid_penalty = best_params.get('invalid_penalty', 0.1)
    portfolio_scaling = best_params.get('portfolio_scaling', 0.1)
    
    code = f'''
# Optimized step function for DQN v5
# Generated from optimization study: {config_data.get('study_name', 'unknown')}
# Best trial: #{config_data.get('best_trial_number', 'unknown')}
# Best score: {config_data.get('best_score', 'unknown'):.3f}

def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
    """Execute action and return next state, reward, done, info
    
    OPTIMIZED REWARD SYSTEM - Parameters found via Optuna optimization:
    - Portfolio scaling: {portfolio_scaling:.3f}
    - Invalid penalty: {invalid_penalty:.3f}
    """
    
    current_price = self.data.iloc[self.current_step]['close']        
    reward = 0
    trade_executed = False
    invalid_action = self._is_invalid_action(action)
    
    # Store current portfolio value for comparison
    if not hasattr(self, 'last_portfolio_value'):
        self.last_portfolio_value = self.balance + (self.position * current_price)
    
    # Calculate current portfolio value
    current_portfolio_value = self.balance + (self.position * current_price)
    
    # OPTIMIZED PORTFOLIO TRACKING: Reward = portfolio value change
    portfolio_change = current_portfolio_value - self.last_portfolio_value
    reward = portfolio_change * {portfolio_scaling:.3f}  # Optimized scaling
    
    # Update last portfolio value for next step
    self.last_portfolio_value = current_portfolio_value
    
    # Execute the action (portfolio change + optimized invalid penalty)
    if invalid_action:
        self.invalid_actions += 1
        self.consecutive_invalid_actions += 1
        # Optimized penalty for invalid actions
        reward -= {invalid_penalty:.3f}  # Optimized penalty
    else:
        self.consecutive_invalid_actions = 0 
        
        if action == 1:  # Buy
            position_value = self.balance * self.config.max_position_size
            shares_to_buy = position_value / current_price
            cost = shares_to_buy * current_price * (1 + self.config.transaction_fee_percent)
            
            self.position = shares_to_buy
            self.balance -= cost
            self.entry_price = current_price
            self.position_entry_step = self.current_step
            trade_executed = True
            self.last_action = 1
            self.consecutive_holds = 0
            
            # Update trade tracking
            self.last_trade_step = self.current_step
            self.steps_since_last_loss += 1
                
        elif action == 2:  # Sell
            revenue = self.position * current_price * (1 - self.config.transaction_fee_percent)
            cost_basis = self.position * self.entry_price * (1 + self.config.transaction_fee_percent)
            profit = revenue - cost_basis
            
            self.balance += revenue
            self.position = 0
            trade_executed = True
            self.total_trades += 1
            self.last_action = 2
            self.consecutive_holds = 0
            
            # Update trade tracking
            self.last_trade_step = self.current_step
            self.last_trade_was_loss = profit <= 0
            if profit <= 0:
                self.steps_since_last_loss = 0
            else:
                self.steps_since_last_loss += 1
            
            if profit > 0:
                self.winning_trades += 1
                self.total_profit += profit
            else:
                self.losing_trades += 1
                self.total_loss += abs(profit)
            
            self.position_entry_step = -1
            
        else:  # Hold (action == 0)
            self.last_action = 0
            self.consecutive_holds += 1
    
    # ... rest of step function remains the same ...
    
    return next_state, reward, done, info
'''
    
    return code


def print_optimization_summary(config_data: Dict[str, Any]):
    """Print summary of optimization results"""
    print("🎯 OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Study: {config_data.get('study_name', 'unknown')}")
    print(f"Best Trial: #{config_data.get('best_trial_number', 'unknown')}")
    print(f"Best Score: {config_data.get('best_score', 'unknown'):.3f}")
    print(f"Timestamp: {config_data.get('timestamp', 'unknown')}")
    
    print("\n🏆 BEST PARAMETERS:")
    best_params = config_data.get('best_parameters', {})
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    
    print("\n📊 PERFORMANCE METRICS:")
    results = config_data.get('detailed_results', {})
    if results:
        print(f"   Average Return: {results.get('avg_return', 0):.2%}")
        print(f"   Win Rate: {results.get('win_rate', 0):.1%}")
        print(f"   Invalid Actions: {results.get('avg_invalid_actions', 0):.1f}")
        print(f"   Invalid Rate: {results.get('invalid_rate', 0):.1%}")
        print(f"   Avg Trades: {results.get('avg_trades', 0):.1f}")
        print(f"   Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")


def generate_config_update_code(config_data: Dict[str, Any]) -> str:
    """Generate code to update TradingConfig with optimized parameters"""
    
    best_params = config_data['best_parameters']
    
    code = f'''
# Update TradingConfig with optimized parameters
def create_optimized_config() -> TradingConfig:
    """Create TradingConfig with Optuna-optimized parameters"""
    config = TradingConfig()
    
    # Optimized reward parameters
    config.invalid_penalty = {best_params.get('invalid_penalty', 0.1):.3f}
    config.portfolio_scaling = {best_params.get('portfolio_scaling', 0.1):.3f}
    
    # Optimized trading parameters
    config.min_profit_threshold = {best_params.get('min_profit_threshold', 0.015):.4f}
    config.transaction_fee_percent = {best_params.get('transaction_fee_percent', 0.001):.4f}
    
    # Optimized learning parameters
    config.learning_rate = {best_params.get('learning_rate', 0.0001):.2e}
    config.epsilon_decay = {best_params.get('epsilon_decay', 2000)}
    
    # Optimized network parameters
    config.batch_size = {best_params.get('batch_size', 64)}
    config.hidden_size = {best_params.get('hidden_size', 512)}
    
    return config
'''
    
    return code


def main():
    """Main application script"""
    parser = argparse.ArgumentParser(description='Apply optimized reward parameters')
    parser.add_argument('--config', required=True, help='Path to optimized config JSON file')
    parser.add_argument('--output', help='Output file for generated code')
    
    args = parser.parse_args()
    
    # Load configuration
    config_data = load_optimized_config(args.config)
    
    # Print summary
    print_optimization_summary(config_data)
    
    # Generate code
    step_function_code = generate_step_function_code(config_data)
    config_update_code = generate_config_update_code(config_data)
    
    # Output code
    full_code = f'''
"""
OPTIMIZED DQN v5 PARAMETERS
Generated from Optuna optimization study: {config_data.get('study_name', 'unknown')}
Best trial: #{config_data.get('best_trial_number', 'unknown')}
Best score: {config_data.get('best_score', 'unknown'):.3f}
"""

{config_update_code}

{step_function_code}
'''
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(full_code)
        print(f"\n💾 Generated code saved to: {args.output}")
    else:
        print("\n📝 GENERATED CODE:")
        print("="*50)
        print(full_code)
    
    print("\n✅ To apply these optimizations:")
    print("   1. Copy the optimized parameters to your TradingConfig")
    print("   2. Update the step function in dqn_v5.py with the new reward scaling")
    print("   3. Test the optimized configuration")


if __name__ == "__main__":
    main() 