"""
Initialize portfolios for users
"""
from src.web.models import Profile, Portfolio, db
from src.web.extensions import app

def init_portfolios():
    """Create default portfolios for users without portfolios"""
    with app.app_context():
        # Get all profiles without portfolios
        profiles_without_portfolio = Profile.query.filter(
            ~Profile.portfolios.any()
        ).all()
        
        for profile in profiles_without_portfolio:
            # Create a default portfolio
            portfolio = Portfolio(
                name=f"{profile.username}'s Portfolio",
                initial_balance=10000.0,
                current_balance=10000.0,
                is_live_trading=False,
                owner=profile
            )
            db.session.add(portfolio)
            print(f"Created portfolio for user: {profile.username}")
        
        db.session.commit()
        print(f"Created {len(profiles_without_portfolio)} portfolios")
        
        # Display all portfolios
        all_portfolios = Portfolio.query.all()
        print(f"\nTotal portfolios in database: {len(all_portfolios)}")
        for portfolio in all_portfolios:
            print(f"  - {portfolio.name} (ID: {portfolio.id}, Owner: {portfolio.owner.username})")

if __name__ == "__main__":
    init_portfolios() 