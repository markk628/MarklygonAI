import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import statistics
import time
import torch
from datetime import datetime

from src.config.config import (
    DATA_DIR, 
    MODELS_DIR, 
    RESULTS_DIR, 
    TRAIN_RATIO, 
    VALID_RATIO,
    NUM_EPISODES,
    EVALUATE_INTERVAL,
    CORE_FEATURES,
    AUXILIARY_FEATURES,
    STOCK_FEATURES
)
from src.models.mark.dqn.agent.DQNAgent import DQNAgent
from src.models.mark.dqn.env.StockTradingEnv import StockTradingEnv
from src.preprocessing.data_processor import RollingWindowFeatureProcessor
from src.utils.utils import format_duration
from src.web.marklygon_web.models import app, db, BacktestHistory, ModelType, MarklygonModel

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class DQNTrainer:

    def load_stock_data(self, ticker: str, cutoff: pd.Timestamp | None=None, cols_to_keep: list[str]=STOCK_FEATURES) -> tuple[pd.DataFrame, datetime, datetime]:
        """
        Get saved csv data
        """
        print(f'Loading {ticker}...')
        file_path = DATA_DIR / f'feature_engineered/{ticker.lower()}.csv'
        df = pd.read_csv(file_path)
        start_date = cutoff.to_pydatetime()
        end_date = pd.to_datetime(df['timestamp'].iloc[-1]).to_pydatetime()

        if cutoff:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[df['timestamp'] >= cutoff]
            
        return df[cols_to_keep], start_date, end_date


    def split_data(self, data: pd.DataFrame, train_ratio: float=TRAIN_RATIO, val_ratio: float=VALID_RATIO):
        """
        Split data chronologically into train, validation, and test sets
        """
        print(f'Splitting data...')
        
        train_end = int(len(data) * train_ratio)
        val_end = train_end + int(len(data) * val_ratio)
        train_data = data.iloc[:train_end].copy().reset_index(drop=True)
        val_data = data.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_data = data.iloc[val_end:].copy().reset_index(drop=True)
        
        print(f"Data split: Train {len(train_data)}, Validation {len(val_data)}, Test {len(test_data)}")
        
        return train_data, val_data, test_data
    
    
    def save_backtest_results_to_db(self,
                                    model_type: ModelType,
                                    ticker: str,
                                    info: dict[str, float]) -> tuple[int, str]:
        backtest_date = info['backtest_date']
        backtest_date_for_path = backtest_date.strftime(r'%Y-%m-%d_%H-%M-%S')
        return_rate = info['return_rate'] * 100
        
        with app.app_context():
            db.create_all()
            model = MarklygonModel(
                model=model_type,
                ticker=ticker
            )
            db.session.add(model)
            db.session.flush()
            
            model_id = model.id
            model_path = f'{MODELS_DIR}/dqn/dqn_{model.id}_{ticker}_{backtest_date_for_path}_{return_rate:.4f}.pth'
            model.model_path = model_path

            backtest = BacktestHistory(
                model=model,
                backtest_date=backtest_date,
                start_date=info['start_date'],
                end_date=info['end_date'],
                initial_balance=info['initial_balance'],
                final_balance=info['final_balance'],
                net_profit=info['net_profit'],
                total_trades=info['total_trades'],
                winning_trades=info['winning_trades'],
                losing_trades=info['losing_trades'],
                return_rate=return_rate,
                max_drawdown=info['max_drawdown'],
                sharpe_ratio=info['sharpe_ratio'],
                calmar_ratio=info['calmar_ratio'],
                invalid_actions=info['invalid_actions'],
            )

            db.session.add(backtest)
            db.session.commit()
        return model_id, model_path
    
    
    def plot_training_results(self, 
                              ticker: str,
                              metrics: dict[str, float | int], 
                              model_name: str="DQN", 
                              validation_frequency: int=5,):
        """
        Visualize training and performance metrics
        """
        plt.figure(figsize=(15, 10))
        
        # plot scores
        plt.subplot(2, 3, 1)
        plt.plot(metrics['scores'], label='Training Score')
        validation_episodes = [i*validation_frequency for i in range(len(metrics['validation_scores']))]
        plt.plot(validation_episodes, metrics['validation_scores'], 'r-', label='Validation Score')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Score')
        plt.title(f'{model_name} Learning Curve - Cumulative Score')
        plt.legend()
        
        # plot balances
        plt.subplot(2, 3, 2)
        plt.plot(metrics['balances'], label='Training Balance')
        validation_balances = [i*validation_frequency for i in range(len(metrics['validation_balances']))]
        plt.plot(validation_balances, metrics['validation_balances'], 'r-', label='Validation Balance')
        plt.xlabel('Episode')
        plt.ylabel('Final Balance ($)')
        plt.title('Portfolio Value at End of Episode')
        
        # plot invalid action counts
        plt.subplot(2, 3, 3)
        plt.plot(metrics['invalid_action_counts'], label='Training Invalid Actions')
        validation_invalid_action_counts = [i*validation_frequency for i in range(len(metrics['validation_invalid_action_counts']))]
        plt.plot(validation_invalid_action_counts, metrics['validation_invalid_action_counts'], 'r-', label='Validation Invalid Actions')
        plt.xlabel('Episode')
        plt.ylabel('Invalid Actions')
        plt.title('Average Invalid Actions During Training')
        
        # plot loss history
        plt.subplot(2, 3, 4)
        plt.plot(metrics['loss_history'])
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        
        # plot average Q-values
        plt.subplot(2, 3, 5)
        plt.plot(metrics['avg_q_values'])
        plt.xlabel('Action Selection')
        plt.ylabel('Average Q-Value')
        plt.title('Average Q-Values During Training')
        
        plt.suptitle(f'{ticker} DQN Training Results')
        plt.tight_layout()
        return plt


    def plot_backtest_results(self, 
                              portfolio_values: list[int], 
                              price_history: list[float], 
                              action_history: list[int], 
                              ticker: str):
        """
        Visualize backtesting results including price chart, portfolio value, and buy/sell actions
        """
        plt.figure(figsize=(15, 10))
        
        # Plot stock price
        plt.subplot(2, 1, 1)
        plt.plot(price_history, label=f'{ticker} Price')
        
        # Mark buy and sell actions
        buy_indices = [i for i, a in enumerate(action_history) if a == 2]
        sell_indices = [i for i, a in enumerate(action_history) if a == 0]
        
        if buy_indices:
            plt.scatter(buy_indices, [price_history[i] for i in buy_indices], 
                    color='green', marker='^', s=60, label='Buy')
        if sell_indices:
            plt.scatter(sell_indices, [price_history[i] for i in sell_indices], 
                    color='red', marker='v', s=60, label='Sell')
        
        plt.xlabel('Trading Step')
        plt.ylabel('Price ($)')
        plt.title(f'{ticker} Price and Trading Actions')
        plt.legend()
        
        # Plot portfolio value
        plt.subplot(2, 1, 2)
        plt.plot(portfolio_values, label='Portfolio Value')
        
        # Calculate and plot buy-and-hold strategy for comparison
        initial_balance = portfolio_values[0]
        initial_price = price_history[0]
        shares_bought = initial_balance / initial_price
        buy_hold_values = [shares_bought * price for price in price_history]
        plt.plot(buy_hold_values, '--', label='Buy & Hold Strategy')
        
        plt.xlabel('Trading Step')
        plt.ylabel('Portfolio Value ($)')
        plt.title('Portfolio Value Comparison')
        plt.legend()
        
        plt.tight_layout()
        return plt
    
    
    def train_agent(self,
                    env: StockTradingEnv, 
                    agent: DQNAgent, 
                    episodes: int=NUM_EPISODES, 
                    validation_env: StockTradingEnv | None=None,
                    validation_frequency: int=EVALUATE_INTERVAL,
                    early_stopping_patience: int=10):
        """
        Train the agent with optional validation and early stopping
        """
        scores = []
        balances = []
        invalid_action_counts = []
        validation_scores = []
        validation_balances = []
        validation_invalid_action_counts = []
        
        best_validation_score = float('-inf')
        patience_counter = 0
        best_model_state = None
        start_time = time.time()
        
        for e in range(episodes):
            agent.current_episode = e
            state = env.reset()
            score = 0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.train()
                state = next_state
                score += reward

            print('training action history length', len(info['trade_info']['action_history']))
            agent.scheduler.step(score)
            scores.append(score)
            balances.append(env.balance)
            invalid_action_counts.append(env.invalid_actions)
            # with open(f'./reward_episode_{e}.json', "w") as file:
            #     json.dump(env.reward_components, file, indent=4, cls=NumpyEncoder)
            
            # if (e + 1) % validation_frequency == 0:
            print(f"\nT. Episode: {e+1}/{episodes} | "
                  f"Steps: {env.current_step} | "
                  f"Avg. Score: {score / env.current_step:.4f} | "
                  f"Balance: {env.balance:.2f} | "
                  f"P. Trades: {env.winning_trades} | "
                  f"L. Trades: {env.losing_trades} | "
                  f"P/L: {env.balance - env.initial_balance:.2f} | "
                  f"Invalid Actions: {env.invalid_actions} | "
                  f"Epsilon: {agent.epsilon:.4f}")
            
            # validation if provided
            if validation_env is not None and (e + 1) % validation_frequency == 0:
                validation_metrics = self.evaluate_agent(validation_env, agent, episodes=1, verbose=True)
                validation_scores += validation_metrics['scores']
                validation_balances += validation_metrics['balances']
                validation_invalid_action_counts += validation_metrics['invalid_action_counts']
                
                validation_score = statistics.mean(validation_metrics['scores'])
                # early stopping logic
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    patience_counter = 0
                    # save best model state
                    best_model_state = {k: v.cpu() for k, v in agent.main_network.state_dict().items()}
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at episode {e+1}. Best validation score: {best_validation_score:.4f}")
                    # restore best model
                    if best_model_state:
                        agent.main_network.load_state_dict(best_model_state)
                        agent.target_network.load_state_dict(best_model_state)
                    break
                
        print(f'Training Time: {format_duration(time.time() - start_time)}')
        
        training_metrics = {
            'scores': scores,
            'balances': balances,
            'invalid_action_counts': invalid_action_counts,
            'validation_scores': validation_scores,
            'validation_balances': validation_balances,
            'validation_invalid_action_counts': validation_invalid_action_counts,
            'loss_history': agent.loss_history,
            'avg_q_values': agent.avg_q_values,
        }
        
        return training_metrics
    

    def evaluate_agent(self,
                       env: StockTradingEnv, 
                       agent: DQNAgent, 
                       episodes: int=10, 
                       verbose: bool=True):
        """
        Evaluate the agent's performance
        """
        scores = []
        balances = []
        invalid_action_counts = []
        start_time = time.time()
        
        for e in range(episodes):
            state = env.reset()
            score = 0
            done = False

            while not done:
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                state = next_state
                score += reward
                
            scores.append(score)
            balances.append(env.balance)
            invalid_action_counts.append(env.invalid_actions)
            
            if verbose:
                print(f"V. Episode: {e+1}/{episodes} | "
                      f"Steps: {env.current_step} | "
                      f"Avg. Score: {score / env.current_step:.4f} | "
                      f"Balance: {env.balance:.2f} | "
                      f"P. Trades: {env.winning_trades} | "
                      f"L. Trades: {env.losing_trades} | "
                      f"P/L: {env.balance - env.initial_balance:.2f} | "
                      f"Invalid Actions: {env.invalid_actions} | "
                      f"Epsilon: {agent.epsilon:.4f}")
            
        if verbose:
            print(f'Evaluation Time: {format_duration(time.time() - start_time)}')

        evaluation_metrics = {
            'scores': scores,
            'balances': balances,
            'invalid_action_counts': invalid_action_counts
        }

        return evaluation_metrics
    
    
    def train_and_evaluate(self):
        print("Let's get this bread")
        # parameters
        ticker = 'AAPL'
        cutoff = pd.Timestamp('2025-04-29 08:00:00', tz='UTC')
        
        # load and prepare data
        print(f'Preparing data starting from {cutoff}...')
        data, start_date, end_date = self.load_stock_data(ticker, cutoff)
        train_data, val_data, test_data = self.split_data(data)
        
        print('Preprocessing data...')
        window_processsor = RollingWindowFeatureProcessor()
        window_processsor.fit(train_data[CORE_FEATURES + AUXILIARY_FEATURES])
        use_dueling = True
        use_hierarchical = True
        use_prioritized = True
        use_vram = True
        
        # create environments
        train_env = StockTradingEnv(
            train_data, 
            window_processsor,
            start_date,
            end_date,
            mode='train',
            use_hierarchical=use_hierarchical
        )
        
        val_env = StockTradingEnv(
            val_data, 
            window_processsor,
            start_date,
            end_date,
            mode='validation',
            use_hierarchical=use_hierarchical
        )
        
        test_env = StockTradingEnv(
            test_data, 
            window_processsor,
            start_date,
            end_date,
            mode='test',
            use_hierarchical=use_hierarchical
        )
        
        # initialize agent
        agent = DQNAgent(
            sizes=train_env.get_branch_sizes(),
            epsilon_decay_rate=10,
            target_update_frequency=200,
            use_dueling=use_dueling,
            use_hierarchical=use_hierarchical,
            use_prioritized=use_prioritized,
            use_vram=use_vram
        )
        
        print('Training start time:', datetime.today())
        
        # train agent with validation-based early stopping
        training_metrics = self.train_agent(
            train_env, 
            agent, 
            validation_env=val_env,
            early_stopping_patience=15
        )
        
        # plot training results
        train_date = datetime.today().strftime(r'%Y-%m-%d_%H-%M-%S')
        training_plot = self.plot_training_results(ticker, training_metrics)
        training_plot.savefig(f'{RESULTS_DIR}/dqn/dqn_{ticker}_{train_date}_training_results.png')
        
        # evaluate on test set
        print("\nEvaluating on test set...")
        self.evaluate_agent(test_env, agent, episodes=1)
        
        state = test_env.reset()
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            next_state, _, done, info = test_env.step(action)
            state = next_state
        
        info = info['trade_info']
        model_id, model_path = self.save_backtest_results_to_db(ModelType.DQN, ticker, info)
        
        # Plot backtest results
        backtest_plot = self.plot_backtest_results(info['portfolio_values'], 
                                                   info['price_history'], 
                                                   info['action_history'], 
                                                   ticker=ticker)
        backtest_plot.savefig(f"{RESULTS_DIR}/dqn/dqn_{model_id}_{ticker}_{train_date}_{info['return_rate']*100:.2f}_backtest_results.png")
            
        # Save model
        agent.save(model_path)
        
        # Print final metrics
        print(f"\nTest Results for {ticker}:")
        print(f"Final Balance: ${info['final_balance']:.2f}")
        print(f"Return Rate: {info['return_rate']:.4f} ({info['return_rate']*100:.4f}%)")
        print(f"Max Drawdown: {info['max_drawdown']:.4f} ({info['max_drawdown']*100:.2f}%)")
        print(f"Sharpe Ratio: {info['sharpe_ratio']:.4f}")
        print(f"Calmar Ratio: {info['calmar_ratio']:.4f}")
        print(f"Invalid Actions: {info['invalid_actions']}")
        print(f"winning Trades: {info['winning_trades']}")
        print(f"losing Trades: {info['losing_trades']}")
        
def main():
    trainer = DQNTrainer()
    trainer.train_and_evaluate()
    
if __name__=='__main__':
    main()