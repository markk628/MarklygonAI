"""
SAC Factory - Complete System Creation
=====================================

Factory functions to create complete SAC trading systems with your choice of:
- Environment type (Basic, Weighted Average, Lot-Based)
- Network type (Original, Simplified)
- Configuration parameters

Provides a clean, modular interface for all SAC components.
"""

import pandas as pd
import torch
from typing import Tuple, Optional, Dict
from datetime import datetime, time

from src.config.config import DEVICE
from src.models.jeawan.sac.sac_config import (
    SACConfig, 
    TradingMode, 
    EnvironmentType, 
    NetworkType,
    create_basic_sac_config,
    create_weighted_average_sac_config,
    create_lot_based_sac_config
)
from src.models.jeawan.sac.sac_agent import SAC
from src.models.jeawan.sac.sac_environments import (
    create_environment,
    BasicTradingEnvironment,
    WeightedAverageTradingEnvironment,
    LotBasedTradingEnvironment
)


def create_sac_system(data: pd.DataFrame,
                     scaled_data: pd.DataFrame,
                     environment_type: EnvironmentType = EnvironmentType.BASIC,
                     network_type: NetworkType = NetworkType.SIMPLIFIED,
                     device: torch.device = DEVICE,
                     **config_overrides) -> Tuple[SAC, object, SACConfig]:
    """
    Create a complete SAC trading system with chosen environment and network types.
    
    Args:
        data: Raw stock data DataFrame
        scaled_data: Preprocessed/scaled stock data DataFrame  
        environment_type: Type of trading environment to use
        network_type: Type of networks to use (original vs simplified)
        device: PyTorch device
        **config_overrides: Override any configuration parameters
    
    Returns:
        Tuple of (SAC agent, training environment, config)
    """
    
    # Create configuration based on environment type
    if environment_type == EnvironmentType.WEIGHTED_AVERAGE:
        config = create_weighted_average_sac_config(**config_overrides)
    elif environment_type == EnvironmentType.LOT_BASED:
        config = create_lot_based_sac_config(**config_overrides)
    else:  # BASIC
        config = create_basic_sac_config(**config_overrides)
    
    # Set network type
    config.network_type = network_type
    config.environment_type = environment_type
    
    # Create agent
    agent = SAC(config, device)
    
    # Create training environment
    train_env = create_environment(data, scaled_data, config, TradingMode.TRAIN, device)
    
    return agent, train_env, config


def create_evaluation_environments(data: pd.DataFrame,
                                  scaled_data: pd.DataFrame, 
                                  config: SACConfig,
                                  device: torch.device = DEVICE) -> Tuple[object, object]:
    """
    Create validation and test environments matching the training environment.
    
    Args:
        data: Raw stock data DataFrame
        scaled_data: Preprocessed/scaled stock data DataFrame
        config: SAC configuration
        device: PyTorch device
    
    Returns:
        Tuple of (validation environment, test environment)
    """
    val_env = create_environment(data, scaled_data, config, TradingMode.VAL, device)
    test_env = create_environment(data, scaled_data, config, TradingMode.TEST, device)
    
    return val_env, test_env


def create_complete_sac_suite(data: pd.DataFrame,
                             scaled_data: pd.DataFrame,
                             environment_type: EnvironmentType = EnvironmentType.BASIC,
                             network_type: NetworkType = NetworkType.SIMPLIFIED,
                             device: torch.device = DEVICE,
                             **config_overrides) -> Dict:
    """
    Create a complete SAC suite with agent and all environments.
    
    Args:
        data: Raw stock data DataFrame
        scaled_data: Preprocessed/scaled stock data DataFrame
        environment_type: Type of trading environment
        network_type: Type of networks to use
        device: PyTorch device
        **config_overrides: Override any configuration parameters
    
    Returns:
        Dictionary containing:
        - 'agent': SAC agent
        - 'train_env': Training environment
        - 'val_env': Validation environment  
        - 'test_env': Test environment
        - 'config': Configuration used
    """
    
    # Create main system
    agent, train_env, config = create_sac_system(
        data, scaled_data, environment_type, network_type, device, **config_overrides
    )
    
    # Create evaluation environments
    val_env, test_env = create_evaluation_environments(data, scaled_data, config, device)
    
    return {
        'agent': agent,
        'train_env': train_env,
        'val_env': val_env,
        'test_env': test_env,
        'config': config
    }

# Data loading utilities (same as original)
def filter_to_regular_hours(df):
    """Filter dataframe to regular market hours using UTC timestamps"""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Regular market hours filtering
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    eastern_times = df['timestamp'].dt.tz_convert('US/Eastern')
    market_open = eastern_times.dt.time >= time(9, 30)
    market_close = eastern_times.dt.time < time(16, 0)
    
    filtered_df = df[market_open & market_close].reset_index(drop=True)
    filtered_df['timestamp'] = filtered_df['timestamp'].dt.tz_localize(None)
    
    print(f"Data filtered: {len(df)} → {len(filtered_df)} rows ({len(filtered_df)/len(df)*100:.1f}%)")
    return filtered_df


def load_stock_data(data_path: str, 
                   cutoff: Optional[pd.Timestamp] = None, 
                   cols_to_keep: Optional[list] = None) -> Tuple[pd.DataFrame, datetime, datetime]:
    """Load and filter stock data"""
    from src.config.config import STOCK_FEATURES_V2
    
    if cols_to_keep is None:
        cols_to_keep = STOCK_FEATURES_V2
    
    df = pd.read_csv(data_path)
    
    # Apply cutoff first if specified
    if cutoff:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if cutoff.tz is not None:
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert(cutoff.tz)
        else:
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        df = df[df['timestamp'] >= cutoff]
        
    # Filter to regular market hours
    df = filter_to_regular_hours(df)
    
    # Get date range after filtering
    start_date = pd.to_datetime(df['timestamp'].iloc[0]).to_pydatetime()
    end_date = pd.to_datetime(df['timestamp'].iloc[-1]).to_pydatetime()
        
    return df[cols_to_keep], start_date, end_date


if __name__ == "__main__":
    pass