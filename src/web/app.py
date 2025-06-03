# from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
# from functools import wraps
# import os
# from datetime import datetime

# from src.web.marklygon_web.models import *
# from src.config.config import DATABASE_WEB
# from src.utils.database import DatabaseManager
# from src.web.marklygon_web.extensions import *

# # app = Flask(__name__)
# # app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
# # app.config['SQLALCHEMY_DATABASE_URI'] = WEB_DATABASE_URI
# # app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# # Initialize the database
# db.init_app(app)

# def login_required(f):
#     """Decorator to require login for certain routes"""
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         if 'user_id' not in session:
#             flash('Please log in to access this page.', 'warning')
#             return redirect(url_for('login'))
#         return f(*args, **kwargs)
#     return decorated_function

# @app.route('/')
# def home():
#     """Home page"""
#     return render_template('home.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     """Login page"""
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')

#         if not username or not password:
#             flash('Both username and password are required.', 'error')
#             return render_template('login.html')

#         user = Profile.query.filter_by(username=username).first()

#         if user and user.check_password(password):
#             session['user_id'] = user.id
#             session['username'] = user.username
#             flash(f'Welcome back, {user.first_name}!', 'success')
#             return redirect(url_for('portfolio'))
#         else:
#             flash('Invalid credentials. Please try again.', 'error')

#     return render_template('login.html')


# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     """Sign up page"""
#     if request.method == 'POST':
#         username = request.form.get('username')
#         first_name = request.form.get('first_name')
#         last_name = request.form.get('last_name')
#         password = request.form.get('password')

#         if not all([username, first_name, last_name, password]):
#             flash('All fields are required.', 'error')
#             return render_template('signup.html')

#         existing_user = Profile.query.filter_by(username=username).first()
#         if existing_user:
#             flash('Username already exists. Please choose a different one.', 'error')
#             return render_template('signup.html')

#         try:
#             new_user = Profile(
#                 username=username,
#                 first_name=first_name,
#                 last_name=last_name,
#                 password=password  # Triggers hashing via setter
#             )
#             db.session.add(new_user)
#             db.session.commit()

#             # default_portfolio = Portfolio(
#             #     name=f"{first_name}'s Portfolio",
#             #     current_balance=10000.0,
#             #     initial_balance=10000.0,
#             #     is_live_trading=False,
#             #     profile_id=new_user.id
#             # )
#             # db.session.add(default_portfolio)
#             # db.session.commit()

#             flash('Account created successfully! You can now log in.', 'success')
#             return redirect(url_for('login'))

#         except Exception as e:
#             db.session.rollback()
#             flash('An error occurred while creating your account. Please try again.', 'error')
#             app.logger.error(f"Signup error: {e}")

#     return render_template('signup.html')


# @app.route('/portfolio')
# @login_required
# def portfolio():
#     """Portfolio page - shows user's portfolios and trading history"""
#     user_id = session.get('user_id')
#     user = session.get(Profile, user_id)
    
#     if not user:
#         flash('User not found.', 'error')
#         return redirect(url_for('login'))
    
#     # Get user's portfolios
#     portfolios = Portfolio.query.filter_by(profile_id=user_id).all()
    
#     # Get recent trades across all portfolios
#     recent_trades = []
#     for portfolio in portfolios:
#         trades = TradeHistory.query.filter_by(portfolio_id=portfolio.id)\
#                                  .order_by(TradeHistory.timestamp.desc())\
#                                  .limit(10).all()
#         recent_trades.extend(trades)
    
#     # Sort all trades by timestamp
#     recent_trades.sort(key=lambda x: x.timestamp, reverse=True)
#     recent_trades = recent_trades[:20]  # Show last 20 trades
    
#     return render_template('portfolio.html', 
#                          user=user, 
#                          portfolios=portfolios, 
#                          recent_trades=recent_trades)

# @app.route('/create_portfolio', methods=['POST'])
# @login_required
# def create_portfolio():
#     """Create a new portfolio"""
#     user_id = session.get('user_id')
#     portfolio_name = request.form.get('portfolio_name')
#     initial_balance = float(request.form.get('initial_balance', 0))
    
#     if not portfolio_name:
#         flash('Portfolio name is required.', 'error')
#         return redirect(url_for('portfolio'))
    
#     try:
        # new_portfolio = Portfolio(
        #     name=portfolio_name,
        #     current_balance=initial_balance,
        #     initial_balance=initial_balance,
        #     is_live_trading=False,
        #     profile_id=user_id
        # )
#         db.session.add(new_portfolio)
#         db.session.commit()
#         flash(f'Portfolio "{portfolio_name}" created successfully!', 'success')
#     except Exception as e:
#         db.session.rollback()
#         flash('Error creating portfolio. Please try again.', 'error')
#         app.logger.error(f"Portfolio creation error: {e}")
    
#     return redirect(url_for('portfolio'))

# @app.route('/api/portfolio/<int:portfolio_id>/trades')
# @login_required
# def get_portfolio_trades(portfolio_id):
#     """API endpoint to get trades for a specific portfolio"""
#     user_id = session.get('user_id')
    
#     # Verify the portfolio belongs to the current user
#     portfolio = Portfolio.query.filter_by(id=portfolio_id, profile_id=user_id).first()
#     if not portfolio:
#         return jsonify({'error': 'Portfolio not found'}), 404
    
#     trades = TradeHistory.query.filter_by(portfolio_id=portfolio_id)\
#                               .order_by(TradeHistory.timestamp.desc())\
#                               .limit(50).all()
    
#     trades_data = [trade.to_dict() for trade in trades]
#     return jsonify(trades_data)

# @app.route('/logout')
# def logout():
#     """Logout - clear session"""
#     session.clear()
#     flash('You have been logged out successfully.', 'info')
#     return redirect(url_for('home'))

# @app.errorhandler(404)
# def not_found_error(error):
#     return render_template('404.html'), 404

# @app.errorhandler(500)
# def internal_error(error):
#     db.session.rollback()
#     return render_template('500.html'), 500


# if __name__ == '__main__':
#     db_manager = DatabaseManager(database=DATABASE_WEB)
#     with app.app_context():
#         # Create tables
#         db.create_all()
#         db_manager.execute("SELECT create_hypertable('trade_history', 'timestamp', if_not_exists => TRUE);")
        
#         # Create a sample user if none exists (for testing)
#         if not Profile.query.first():
#             sample_user = Profile(
#                 username='demo_user',
#                 first_name='Demo',
#                 last_name='User',
#                 password='password123'
#             )
#             db.session.add(sample_user)
#             db.session.commit()
            
#             # Create sample portfolio
#             sample_portfolio = Portfolio(
#                 name='Demo Portfolio',
#                 current_balance=15000.0,
#                 initial_balance=10000.0,
#                 is_live_trading=False,
#                 profile_id=sample_user.id
#             )
#             db.session.add(sample_portfolio)
#             db.session.commit()
            
#             print("Created demo user: username='demo_user'")
    
#     app.run(debug=True)

from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
import json
import random
import os

from src.web.models import *
from src.config.config import DATABASE_WEB
from src.utils.database import DatabaseManager
from src.web.extensions import *

# Initialize the database
# db.init_app(app)

# Initialize extensions
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simple User Model
# class User(UserMixin, db.Model):
#     __tablename__ = 'user'
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)Profile)
#     password_hash = db.Column(db.String(255), nullable=False)
#     created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))

@login_manager.user_loader
def load_user(user_id):
    return Profile.query.get(int(user_id))

AI_MODELS = [
    {"name": "Eddie", "performance": 24.7, "risk": 3.2, "description": "Primary AI model with advanced ML algorithms"},
    {"name": "Aurora", "performance": 19.3, "risk": 2.8, "description": "Conservative growth-focused model"},
    {"name": "Phoenix", "performance": 31.5, "risk": 4.1, "description": "High-growth aggressive trading model"},
    {"name": "Titan", "performance": 22.8, "risk": 3.0, "description": "Balanced risk-reward optimization"},
    {"name": "Vega", "performance": 18.2, "risk": 2.5, "description": "Low-risk dividend strategy model"},
    {"name": "Orion", "performance": 28.9, "risk": 3.8, "description": "Technical analysis momentum model"}
]

TICKERS = [
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "price": 891.30, "change": 4.2},
    {"symbol": "AAPL", "name": "Apple Inc.", "price": 195.50, "change": 2.3},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "price": 415.75, "change": -0.8},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "price": 178.90, "change": -1.5},
    {"symbol": "TSLA", "name": "Tesla Inc.", "price": 245.60, "change": 3.7},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "price": 152.30, "change": 1.2},
    {"symbol": "META", "name": "Meta Platforms Inc.", "price": 484.20, "change": 0.9},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "price": 198.45, "change": 1.1},
    {"symbol": "BAC", "name": "Bank of America Corp.", "price": 42.15, "change": -0.5},
    {"symbol": "GS", "name": "Goldman Sachs Group Inc.", "price": 445.30, "change": 2.8}
]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        profile = Profile.query.filter_by(username=username).first()
        
        if profile and check_password_hash(profile._password, password):
            login_user(profile)
            flash('로그인에 성공했습니다!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('아이디 또는 비밀번호가 잘못되었습니다.', 'error')
    
    return render_template('signin.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if Profile.query.filter_by(username=username).first():
            flash('이미 사용 중인 아이디입니다.', 'error')
            return render_template('signup.html')
        
        if Profile.query.filter_by(email=email).first():
            flash('이미 등록된 이메일입니다.', 'error')
            return render_template('signup.html')
        
        # Create new user using the property setter for password hashing
        profile = Profile(
            email=email,
            username=username,
        )
        profile.password = password  
        
        db.session.add(profile)
        db.session.commit()
        
        login_user(profile)
        flash('회원가입이 완료되었습니다!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('로그아웃되었습니다.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Generate portfolio data for demo
    portfolio_data = {
        'portfolio_summary': {
            'total_value': 125430.50 + random.uniform(-2000, 3000),
            'daily_change': random.uniform(-2.5, 3.5),
            'daily_change_amount': random.uniform(-500, 800),
            'total_return': random.uniform(15, 25),
            'active_positions': random.randint(8, 15),
            'win_rate': random.uniform(65, 75),
            'new_positions_today': random.randint(0, 5)
        },
        'model_performance': [
            {
                'model_id': f'model_{i}',
                'model_name': model['name'],
                'model_type': 'AI Trading Model',
                'status': 'completed',
                'current_value': 20000 + random.uniform(-2000, 5000),
                'allocation_percentage': random.uniform(15, 25),
                'total_return': model['performance'],
                'daily_change': random.uniform(-2, 4)
            } for i, model in enumerate(AI_MODELS)
        ]
    }
    return render_template('dashboard.html', portfolio=portfolio_data)

@app.route('/portfolio')
@login_required
def portfolio():
    # 포트폴리오 페이지는 대시보드로 통합됨
    return redirect(url_for('dashboard'))

@app.route('/backtest')
@login_required
def backtest():
    return render_template('backtest.html')

@app.route('/mypage')
@login_required
def mypage():
    return render_template('mypage.html')

# API Routes
@app.route('/api/models')
def api_models():
    models_data = {
        'models': [
            {
                'id': f'model_{i}',
                'model_name': model['name'],
                'model_type': 'AI Trading Model',
                'description': model['description'],
                'training_status': 'completed'
            } for i, model in enumerate(AI_MODELS)
        ]
    }
    return jsonify(models_data)

@app.route('/api/tickers')
def api_tickers():
    tickers_data = {
        'tickers': [
            {
                'symbol': ticker['symbol'],
                'company_name': ticker['name'],
                'current_price': ticker['price'],
                'change_percent': ticker['change']
            } for ticker in TICKERS
        ]
    }
    return jsonify(tickers_data)

@app.route('/api/portfolio')
@login_required
def api_portfolio():
    portfolio_data = {
        'portfolio_summary': {
            'total_value': 125430.50 + random.uniform(-2000, 3000),
            'daily_change': random.uniform(-2.5, 3.5),
            'daily_change_amount': random.uniform(-500, 800),
            'total_return': random.uniform(15, 25),
            'active_positions': random.randint(8, 15),
            'win_rate': random.uniform(65, 75)
        },
        'model_performance': [
            {
                'model_id': f'model_{i}',
                'model_name': model['name'],
                'model_type': 'AI Trading Model',
                'status': 'completed',
                'current_value': 20000 + random.uniform(-2000, 5000),
                'allocation_percentage': random.uniform(15, 25),
                'total_return': model['performance'],
                'daily_change': random.uniform(-2, 4)
            } for i, model in enumerate(AI_MODELS)
        ]
    }
    return jsonify(portfolio_data)

@app.route('/api/backtest-results')
@login_required
def api_backtest_results():
    backtest_data = {
        'backtest_results': []
    }
    
    # Generate sample backtest results
    for model in AI_MODELS[:3]:  # First 3 models
        for ticker in TICKERS[:3]:  # First 3 tickers
            result = {
                'model_id': f'model_{AI_MODELS.index(model)}',
                'model_name': model['name'],
                'ticker': ticker['symbol'],
                'company_name': ticker['name'],
                'return_rate': random.uniform(-10, 35),
                'max_drawdown': random.uniform(-20, -5),
                'sharpe_ratio': random.uniform(0.8, 2.5),
                'invalid_actions': random.randint(0, 8),
                'win_rate': random.uniform(55, 75)
            }
            backtest_data['backtest_results'].append(result)
    
    return jsonify(backtest_data)

@app.route('/api/realtime/portfolio')
@login_required
def api_realtime_portfolio():
    realtime_data = {
        'total_value': 125430.50 + random.uniform(-2000, 3000),
        'daily_change': random.uniform(-2.5, 3.5),
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(realtime_data)

@app.route('/api/user/profile', methods=['GET', 'POST'])
@login_required
def api_user_profile():
    if request.method == 'GET':
        profile_data = {
            'full_name': current_user.username,
            'email': current_user.email,
            'trading_level': 'intermediate',
            'risk_tolerance': 'moderate',
            'preferred_models': ['model_0', 'model_1', 'model_2'],
            'notification_settings': {
                'email': True,
                'push': True,
                'trading': True,
                'reports': True
            }
        }
        return jsonify(profile_data)
    
    elif request.method == 'POST':
        return jsonify({'status': 'success', 'message': 'Profile updated successfully'})

if __name__ == '__main__':
    print("MarklygonAI Flask Application Starting...")

    with app.app_context():
        db.create_all()
        print("Database initialized with sample data")

        # Create default user if not exists
        if not Profile.query.filter_by(username='demo').first():
            demo_user = Profile(
                username='demo',
                email='demo@marklygon.ai',
            )
            demo_user.password = 'demo123'  # Triggers password hash via property setter
            db.session.add(demo_user)
            db.session.commit()
            print("Demo user created: demo/demo123")

    print("Application running at: http://localhost:5000")
    print("Default login: Create a new account at /register")
    app.run(debug=True, host='0.0.0.0', port=5000)
