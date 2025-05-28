-- MarklygonAI Database Schema
-- PostgreSQL Database for MarklygonAI Trading Platform

-- Drop existing tables if they exist (for development purposes)
DROP TABLE IF EXISTS trading_signals CASCADE;
DROP TABLE IF EXISTS performance_metrics CASCADE;
DROP TABLE IF EXISTS training_history CASCADE;
DROP TABLE IF EXISTS portfolio_holdings CASCADE;
DROP TABLE IF EXISTS backtest_results CASCADE;
DROP TABLE IF EXISTS tickers CASCADE;
DROP TABLE IF EXISTS models CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Users table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    trading_level VARCHAR(20) DEFAULT 'beginner' CHECK (trading_level IN ('beginner', 'intermediate', 'advanced', 'expert')),
    risk_tolerance VARCHAR(20) DEFAULT 'moderate' CHECK (risk_tolerance IN ('conservative', 'moderate', 'aggressive')),
    preferred_models JSONB DEFAULT '[]',
    notification_settings JSONB DEFAULT '{"email": true, "push": true, "trading": true, "reports": true}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Models table
CREATE TABLE models (
    model_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'signal_generator', 'auxiliary', etc.
    version VARCHAR(20) NOT NULL,
    description TEXT,
    parameters INTEGER, -- number of model parameters
    architecture JSONB, -- model architecture details
    training_status VARCHAR(20) DEFAULT 'not_started' CHECK (training_status IN ('not_started', 'training', 'completed', 'failed', 'paused')),
    training_phase INTEGER DEFAULT 1,
    performance_metrics JSONB, -- accuracy, sharpe_ratio, max_drawdown, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(user_id)
);

-- Tickers table
CREATE TABLE tickers (
    ticker_id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    company_name VARCHAR(200),
    sector VARCHAR(100),
    market_cap BIGINT,
    is_active BOOLEAN DEFAULT TRUE,
    trading_phase INTEGER DEFAULT 1, -- which phase this ticker is included in
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Backtest Results table
CREATE TABLE backtest_results (
    result_id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) REFERENCES models(model_id),
    ticker_id INTEGER REFERENCES tickers(ticker_id),
    return_rate DECIMAL(10, 4), -- return rate in percentage
    max_drawdown DECIMAL(10, 4), -- max drawdown in percentage
    sharpe_ratio DECIMAL(10, 4), -- sharpe ratio
    invalid_actions INTEGER DEFAULT 0, -- number of invalid actions
    total_trades INTEGER, -- total number of trades
    win_rate DECIMAL(5, 2), -- win rate in percentage
    profit_factor DECIMAL(10, 4), -- profit factor
    start_date DATE,
    end_date DATE,
    initial_capital DECIMAL(15, 2),
    final_capital DECIMAL(15, 2),
    volatility DECIMAL(10, 4), -- portfolio volatility
    sortino_ratio DECIMAL(10, 4), -- sortino ratio
    calmar_ratio DECIMAL(10, 4), -- calmar ratio
    test_metadata JSONB, -- additional test metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, ticker_id) -- one backtest result per model-ticker pair
);

-- Portfolio Holdings table
CREATE TABLE portfolio_holdings (
    holding_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    model_id VARCHAR(50) REFERENCES models(model_id),
    ticker_id INTEGER REFERENCES tickers(ticker_id),
    quantity DECIMAL(15, 4),
    average_price DECIMAL(10, 2),
    current_price DECIMAL(10, 2),
    market_value DECIMAL(15, 2),
    unrealized_pnl DECIMAL(15, 2),
    weight DECIMAL(5, 2), -- portfolio weight in percentage
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, model_id, ticker_id)
);

-- Trading Signals table
CREATE TABLE trading_signals (
    signal_id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) REFERENCES models(model_id),
    ticker_id INTEGER REFERENCES tickers(ticker_id),
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD')),
    confidence DECIMAL(5, 4) CHECK (confidence >= 0 AND confidence <= 1), -- signal confidence 0-1
    price DECIMAL(10, 2),
    quantity DECIMAL(15, 4),
    timestamp TIMESTAMP NOT NULL,
    executed BOOLEAN DEFAULT FALSE,
    execution_price DECIMAL(10, 2),
    execution_time TIMESTAMP,
    signal_metadata JSONB, -- additional signal information
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training History table
CREATE TABLE training_history (
    history_id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) REFERENCES models(model_id),
    epoch INTEGER,
    train_loss DECIMAL(12, 8),
    val_loss DECIMAL(12, 8),
    learning_rate DECIMAL(12, 8),
    batch_size INTEGER,
    metrics JSONB, -- additional training metrics
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance Metrics table (time series data)
CREATE TABLE performance_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) REFERENCES models(model_id),
    ticker_id INTEGER REFERENCES tickers(ticker_id),
    user_id INTEGER REFERENCES users(user_id),
    date DATE,
    daily_return DECIMAL(10, 6),
    cumulative_return DECIMAL(10, 4),
    portfolio_value DECIMAL(15, 2),
    volatility DECIMAL(10, 6),
    benchmark_return DECIMAL(10, 6), -- benchmark comparison
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, ticker_id, user_id, date)
);

-- Indexes for better performance
CREATE INDEX idx_backtest_results_model_id ON backtest_results(model_id);
CREATE INDEX idx_backtest_results_ticker_id ON backtest_results(ticker_id);
CREATE INDEX idx_backtest_results_return_rate ON backtest_results(return_rate DESC);
CREATE INDEX idx_backtest_results_sharpe_ratio ON backtest_results(sharpe_ratio DESC);

CREATE INDEX idx_portfolio_holdings_user_id ON portfolio_holdings(user_id);
CREATE INDEX idx_portfolio_holdings_model_id ON portfolio_holdings(model_id);

CREATE INDEX idx_trading_signals_model_id ON trading_signals(model_id);
CREATE INDEX idx_trading_signals_timestamp ON trading_signals(timestamp DESC);

CREATE INDEX idx_performance_metrics_model_id ON performance_metrics(model_id);
CREATE INDEX idx_performance_metrics_date ON performance_metrics(date DESC);

CREATE INDEX idx_training_history_model_id ON training_history(model_id);
CREATE INDEX idx_training_history_timestamp ON training_history(timestamp DESC);

-- Insert sample data
INSERT INTO users (username, email, password_hash, full_name, trading_level, risk_tolerance) VALUES
('aitrader', 'aitrader@marklygon.ai', '$2b$12$example_hash', 'AI Trader', 'advanced', 'moderate');

INSERT INTO models (model_id, model_name, model_type, version, description, parameters, training_status, training_phase, performance_metrics) VALUES
('eddie-001', 'Eddie', 'signal_generator', '1.0.0', 'Main Eddie System with Signal Generator and Pattern Analyzer', 3300000, 'training', 1, '{"accuracy": 0.67, "sharpe_ratio": 1.45, "max_drawdown": -0.12}'),
('mark-001', 'Mark', 'auxiliary', '1.0.0', 'Mark auxiliary model for enhanced signal processing', 2100000, 'completed', 1, '{"accuracy": 0.62, "sharpe_ratio": 1.23, "max_drawdown": -0.15}'),
('sugarmixcoffee-001', 'SugarMixCoffee', 'auxiliary', '1.0.0', 'SugarMixCoffee model for volatility analysis', 1800000, 'completed', 1, '{"accuracy": 0.59, "sharpe_ratio": 1.08, "max_drawdown": -0.18}'),
('mint-001', 'Mint', 'auxiliary', '1.0.0', 'Mint model for trend analysis', 2200000, 'training', 1, '{"accuracy": 0.64, "sharpe_ratio": 1.32, "max_drawdown": -0.14}'),
('jeawan-001', 'Jeawan', 'auxiliary', '1.0.0', 'Jeawan model for pattern recognition', 1950000, 'completed', 1, '{"accuracy": 0.61, "sharpe_ratio": 1.18, "max_drawdown": -0.16}'),
('bnm-001', 'BNM', 'auxiliary', '1.0.0', 'BNM model for market regime detection', 2000000, 'training', 1, '{"accuracy": 0.63, "sharpe_ratio": 1.27, "max_drawdown": -0.13}');

INSERT INTO tickers (symbol, company_name, sector, trading_phase) VALUES
('NVDA', 'NVIDIA Corporation', 'Technology', 1),
('AAPL', 'Apple Inc.', 'Technology', 1),
('MSFT', 'Microsoft Corporation', 'Technology', 1),
('AMZN', 'Amazon.com Inc.', 'Consumer Discretionary', 1),
('JPM', 'JPMorgan Chase & Co.', 'Financial Services', 1),
('BAC', 'Bank of America Corporation', 'Financial Services', 1),
('GS', 'The Goldman Sachs Group Inc.', 'Financial Services', 1),
('JNJ', 'Johnson & Johnson', 'Healthcare', 1),
('MCD', 'McDonald''s Corporation', 'Consumer Discretionary', 1),
('KO', 'The Coca-Cola Company', 'Consumer Staples', 1);

-- Insert sample backtest results
INSERT INTO backtest_results (model_id, ticker_id, return_rate, max_drawdown, sharpe_ratio, invalid_actions, total_trades, win_rate, profit_factor, start_date, end_date, initial_capital, final_capital) VALUES
('eddie-001', 1, 24.67, -12.34, 1.85, 3, 156, 67.31, 2.14, '2024-01-01', '2024-12-31', 100000.00, 124670.00),
('eddie-001', 2, 18.45, -8.92, 1.67, 2, 142, 63.38, 1.89, '2024-01-01', '2024-12-31', 100000.00, 118450.00),
('eddie-001', 3, 21.32, -10.15, 1.74, 1, 134, 65.67, 2.01, '2024-01-01', '2024-12-31', 100000.00, 121320.00),
('mark-001', 1, 19.23, -15.67, 1.42, 7, 178, 61.24, 1.67, '2024-01-01', '2024-12-31', 100000.00, 119230.00),
('mark-001', 2, 14.78, -11.23, 1.28, 5, 165, 58.79, 1.45, '2024-01-01', '2024-12-31', 100000.00, 114780.00),
('sugarmixcoffee-001', 5, 12.45, -18.34, 1.08, 12, 203, 54.19, 1.23, '2024-01-01', '2024-12-31', 100000.00, 112450.00),
('mint-001', 4, 16.89, -13.56, 1.39, 4, 149, 62.42, 1.73, '2024-01-01', '2024-12-31', 100000.00, 116890.00),
('jeawan-001', 6, 11.67, -16.78, 0.98, 9, 187, 56.15, 1.18, '2024-01-01', '2024-12-31', 100000.00, 111670.00),
('bnm-001', 7, 15.34, -14.21, 1.26, 6, 158, 60.13, 1.52, '2024-01-01', '2024-12-31', 100000.00, 115340.00);

-- Create a view for easy backtest results querying
CREATE VIEW backtest_results_view AS
SELECT 
    br.result_id,
    br.model_id,
    m.model_name,
    t.symbol as ticker,
    t.company_name,
    br.return_rate,
    br.max_drawdown,
    br.sharpe_ratio,
    br.invalid_actions,
    br.total_trades,
    br.win_rate,
    br.profit_factor,
    br.start_date,
    br.end_date,
    br.initial_capital,
    br.final_capital,
    br.created_at
FROM backtest_results br
JOIN models m ON br.model_id = m.model_id
JOIN tickers t ON br.ticker_id = t.ticker_id
ORDER BY br.return_rate DESC;

-- Create a view for portfolio summary
CREATE VIEW portfolio_summary AS
SELECT 
    ph.user_id,
    ph.model_id,
    m.model_name,
    COUNT(ph.ticker_id) as holdings_count,
    SUM(ph.market_value) as total_value,
    SUM(ph.unrealized_pnl) as total_unrealized_pnl,
    AVG(ph.weight) as avg_weight
FROM portfolio_holdings ph
JOIN models m ON ph.model_id = m.model_id
GROUP BY ph.user_id, ph.model_id, m.model_name;

-- Function to update portfolio values (would be called by backend)
CREATE OR REPLACE FUNCTION update_portfolio_values()
RETURNS VOID AS $$
BEGIN
    -- This function would be implemented to update current prices and calculate market values
    -- For now, it's a placeholder
    UPDATE portfolio_holdings 
    SET last_updated = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE users IS 'User accounts and preferences';
COMMENT ON TABLE models IS 'AI trading models and their metadata';
COMMENT ON TABLE tickers IS 'Stock symbols and company information';
COMMENT ON TABLE backtest_results IS 'Historical backtest performance results';
COMMENT ON TABLE portfolio_holdings IS 'Current portfolio positions for each user/model combination';
COMMENT ON TABLE trading_signals IS 'Generated trading signals from AI models';
COMMENT ON TABLE training_history IS 'Model training progress and metrics';
COMMENT ON TABLE performance_metrics IS 'Time series performance data';

-- Ensure proper permissions (adjust as needed for your environment)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO marklygon_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO marklygon_app; 