from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List

from src.config.config import MODELS_DIR
from src.utils.utils import create_directory
from src.web.models import app, db, BacktestHistory, ModelType, MarklygonModel


def save_backtest_results_to_db(model_type: ModelType,
                                ticker: str,
                                info: dict,
                                preprocessor_path: Optional[str] = None) -> Tuple[int, str, str]:
    """Save backtest results to database and create model entry"""
    backtest_date = info['backtest_date']
    return_rate = info['return_rate']
    
    with app.app_context():
        db.create_all()
        model = MarklygonModel(
            model=model_type,
            ticker=ticker
        )
        db.session.add(model)
        db.session.flush()
        
        model_id = model.id
        # Create directory for this model - use absolute path
        model_dir = str(MODELS_DIR / 'dqn_v2' / str(model_id))
        create_directory(model_dir)
        
        # Set model path within the model's directory - use absolute path
        model_path = str(Path(model_dir) / 'model.pth')
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
            max_drawdown=abs(info['max_drawdown']),
            sharpe_ratio=info['sharpe_ratio'],
            invalid_actions=info['invalid_actions'],
            preprocessor_path=preprocessor_path
        )

        db.session.add(backtest)
        db.session.commit()
    return model_id, model_path, model_dir


def create_model_in_db(model_type: ModelType, ticker: str) -> Tuple[int, str, str]:
    """
    Create a model entry in the database and return model info
    
    Returns:
        tuple: (model_id, model_path, model_dir)
    """
    with app.app_context():
        db.create_all()
        model = MarklygonModel(
            model=model_type,
            ticker=ticker
        )
        db.session.add(model)
        db.session.flush()
        
        model_id = model.id
        # Create directory for this model - use absolute path
        model_dir = str(MODELS_DIR / 'dqn_v2' / str(model_id))
        create_directory(model_dir)
        
        # Set model path within the model's directory - use absolute path
        model_path = str(Path(model_dir) / 'model.pth')
        model.model_path = model_path
        
        db.session.commit()
    
    return model_id, model_path, model_dir


def save_backtest_to_existing_model(model_id: int, 
                                   info: dict, 
                                   preprocessor_path: Optional[str] = None) -> int:
    """
    Save a backtest result to an existing model
    
    Args:
        model_id: ID of existing model
        info: Backtest information dictionary
        preprocessor_path: Optional path to preprocessor
        
    Returns:
        backtest_id: ID of created backtest entry
    """
    backtest_date = info['backtest_date']
    return_rate = info['return_rate']
    
    with app.app_context():
        # Get the existing model
        model = MarklygonModel.query.get(model_id)
        if not model:
            raise ValueError(f"Model with ID {model_id} not found")
        
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
            max_drawdown=abs(info['max_drawdown']),
            sharpe_ratio=info['sharpe_ratio'],
            invalid_actions=info['invalid_actions'],
            preprocessor_path=preprocessor_path
        )

        db.session.add(backtest)
        db.session.commit()
        
        return backtest.id


def save_multi_day_backtest_to_db(model_type: ModelType,
                                 ticker: str,
                                 multi_day_results: dict,
                                 start_date,
                                 end_date,
                                 initial_balance: float,
                                 preprocessor_path: Optional[str] = None) -> Tuple[int, str, str, List[int]]:
    """
    Save multi-day backtest results: one model with multiple backtest entries
    
    Args:
        model_type: Type of model (e.g., ModelType.DQN)
        ticker: Stock ticker symbol
        multi_day_results: Results from multi-day testing
        start_date: Start date of testing period
        end_date: End date of testing period  
        initial_balance: Initial trading balance
        preprocessor_path: Optional path to preprocessor
        
    Returns:
        tuple: (model_id, model_path, model_dir, backtest_ids)
    """
    # Create the model first
    model_id, model_path, model_dir = create_model_in_db(model_type, ticker)
    
    individual_days = multi_day_results['individual_days']
    aggregate_stats = multi_day_results['aggregate_stats']
    
    backtest_ids = []
    
    # Save aggregate summary backtest
    aggregate_info = {
        'backtest_date': datetime.now(timezone.utc),
        'start_date': start_date,
        'end_date': end_date,
        'initial_balance': initial_balance,
        'final_balance': aggregate_stats['avg_final_value'],
        'net_profit': aggregate_stats['avg_final_value'] - initial_balance,
        'total_trades': int(aggregate_stats['avg_trades']),
        'winning_trades': int(aggregate_stats['avg_winning_trades']),
        'losing_trades': int(aggregate_stats['avg_losing_trades']),
        'return_rate': aggregate_stats['avg_return'], 
        'max_drawdown': aggregate_stats['avg_max_drawdown'],
        'sharpe_ratio': aggregate_stats['avg_sharpe_ratio'],
        'invalid_actions': int(aggregate_stats['avg_invalid_actions']),
    }
    
    aggregate_backtest_id = save_backtest_to_existing_model(model_id, aggregate_info, preprocessor_path)
    backtest_ids.append(aggregate_backtest_id)
    
    # Save individual day backtests
    for i, day_result in enumerate(individual_days, 1):
        day_info = {
            'backtest_date': datetime.now(timezone.utc),
            'start_date': start_date,
            'end_date': end_date,
            'initial_balance': initial_balance,
            'final_balance': day_result['final_value'],
            'net_profit': day_result['final_value'] - initial_balance,
            'total_trades': day_result['total_trades'],
            'winning_trades': day_result['winning_trades'],
            'losing_trades': day_result['losing_trades'],
            'return_rate': day_result['total_return'], 
            'max_drawdown': day_result['max_drawdown'],
            'sharpe_ratio': day_result['sharpe_ratio'],
            'invalid_actions': day_result['invalid_actions'],
        }
        
        day_backtest_id = save_backtest_to_existing_model(model_id, day_info)
        backtest_ids.append(day_backtest_id)
    
    return model_id, model_path, model_dir, backtest_ids 