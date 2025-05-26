from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

from src.web.marklygon_web.models import db

from src.config.config import DATABASE_URI

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # avoids a warning

db.init_app(app)

@app.route('/')
def check():
    return 'Flask is working'

@app.route('/profile')
def profile():
    return render_template('')

@app.route('/profile/portfolio<portfolio_id>')
def portfolio():
    return 


@app.cli.command("create-db")
def create_db_command():
    """Creates database tables and TimescaleDB hypertable using SQLAlchemy-TimescaleDB."""
    with app.app_context():
        # Ensure the TimescaleDB extension is enabled (still often a good idea)
        try:
            db.session.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
            db.session.commit()
            print("TimescaleDB extension ensured.")
        except Exception as e:
            print(f"Warning: Could not create TimescaleDB extension (may already exist or permission issue): {e}")
            db.session.rollback()
        db.create_all()
    print("Database tables created and hypertables configured!")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run()