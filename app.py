from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import random
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'marklygon-ai-secret-key-2025'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///marklygonai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Simple User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Sample data for MarklygonAI
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
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
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
        if User.query.filter_by(username=username).first():
            flash('이미 사용 중인 아이디입니다.', 'error')
            return render_template('signup.html')
        
        if User.query.filter_by(email=email).first():
            flash('이미 등록된 이메일입니다.', 'error')
            return render_template('signup.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
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
        if not User.query.filter_by(username='demo').first():
            demo_user = User(
                username='demo',
                email='demo@marklygon.ai',
                password_hash=generate_password_hash('demo123')
            )
            db.session.add(demo_user)
            db.session.commit()
            print("Demo user created: demo/demo123")
    
    print("Application running at: http://localhost:5000")
    print("Default login: Create a new account at /register")
    app.run(debug=True, host='0.0.0.0', port=5000)