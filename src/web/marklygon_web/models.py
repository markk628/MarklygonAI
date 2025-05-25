from enum import Enum
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import String, ForeignKey, DateTime, func, Numeric
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy_timescaledb import TimescaleDB, Hypertable
from typing import List, Optional
from datetime import datetime

class Base(SQLAlchemy.DeclarativeBase, TimescaleDB):
    pass

db = SQLAlchemy(model_class=Base)

class Profile(db.Model):
    __tablename__ = 'profiles'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped['str'] = mapped_column(String(30), unique=True, nullable=False)
    first_name = mapped_column(String(30), unique=False, nullable=False)
    last_name = mapped_column(String(30), unique=False, nullable=False)
    portfolios: Mapped[List["Portfolio"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan", # deletes portfolios if profile is deleted
        lazy="selectin" # eagerly load portfolios when a profile is queried
    )
    
    def __repr__(self):
        return f'<Profile {self.username}>'
    
    
class Portfolio(db.Model):
    __tablename__ = 'portfolios'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    current_balance: Mapped[float] = mapped_column(default=0.0)
    initial_balance: Mapped[float] = mapped_column(default=0.0)
    is_live_trading: Mapped[bool] = mapped_column(default=False)
    portfolios: Mapped[List["TradeHistory"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan", # deletes portfolios if profile is deleted
        lazy="selectin" # eagerly load portfolios when a profile is queried
    )
    
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id"), nullable=False)
    owner: Mapped["Profile"] = relationship(back_populates="portfolios")
    
    def __repr__(self):
        return f'<Portfolio {self.name} (Owner ID: {self.profile_id})>'

 
class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    
class TradeHistory(db.Model):
    __tablename__ = 'trade_history'
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), nullable=False)
    trade_type: Mapped[str] = mapped_column(TradeType, nullable=False, default=TradeType.HOLD)
    amount: Mapped[Numeric] = mapped_column(Numeric(precision=18, scale=4), nullable=False)
    price: Mapped[Numeric] = mapped_column(Numeric(precision=18, scale=4), nullable=False)
    shares: Mapped[Numeric] = mapped_column(Numeric(precision=18, scale=4), nullable=False)
    
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id"), nullable=False)
    portfolio: Mapped["Portfolio"] = relationship(back_populates="trade_history")

    # create a hypertable on 'timestamp' when db.create_all() is called.
    __table_args__ = (
        Hypertable(time_column_name=timestamp),
        db.Index('idx_trade_history_timestamp', timestamp.name), # performance boost
        db.UniqueConstraint('id', name='uq_trade_history_id')
    )

    def __repr__(self):
        return f'<Trade {self.trade_type} at {self.timestamp} (Portfolio ID: {self.portfolio_id})>'
