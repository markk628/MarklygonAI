import json
import pandas as pd
import psycopg2
from contextlib import contextmanager
from datetime import timezone
from functools import wraps
from psycopg2 import pool
from psycopg2.extras import execute_values
from psycopg2.sql import Identifier, Literal, SQL
from typing import List, Optional

from src.utils.logger import Logger
from src.config.config import DATA_DIR, TICKERS, DATABASE_HOST, DATABASE_PORT, DATABASE_DATABASE, DATABASE_USER, DATABASE_PASSWORD

class DatabaseManager:
    def __init__(
        self,
        host: str = DATABASE_HOST,
        port: int = DATABASE_PORT,
        database: str = DATABASE_DATABASE,
        user: str = DATABASE_USER,
        password: str = DATABASE_PASSWORD,
        logger: Optional[Logger] = None
    ):
        self.config = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.logger = logger
        self.connection_pool = None
        self._initialize_pool()
        
    def _initialize_pool(self) -> None:
        '''
        데이터베이스 연결 풀 초기화
        '''
        try:
            self.connection_pool = pool.SimpleConnectionPool(minconn=1, 
                                                             maxconn=10,
                                                             **self.config)
            if self.logger:
                self.logger.info(f'데이터베이스 연결 풀 초기화 완료: {self.config["host"]}:{self.config["port"]}/{self.config["database"]}')
        except pool.PoolError as e:
            if self.logger:
                self.logger.error(f'데이터베이스 연결 풀 초기화 실패: {e}')
            raise
        except psycopg2.OperationalError as e:
            if self.logger:
                self.logger.error(f'데이터베이스 연결 실패: {e}')
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(f'데이터베이스 예기치 않은 오류: {e}')
            raise
        
    @contextmanager
    def _connect(self):
        '''
        풀을 이용해 데이터배이스 연결
        '''
        connection = None
        try:
            connection = self.connection_pool.getconn()
            yield connection
        except pool.PoolError as e:
            if self.logger:
                self.logger.error(f'데이터베이스 연결 풀 연결 실패: {e}')
            raise
        except psycopg2.OperationalError as e:
            if self.logger:
                self.logger.error(f'데이터베이스 연결 실패: {e}')
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(f'데이터베이스 예기치 않은 오류: {e}')
            raise
        finally:
            if connection:
                try:
                    self.connection_pool.putconn(connection)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f'데이터베이스 연결 반환 실패: {e}')
                        
    def db_exception_handler(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            try:
                with self._connect() as connection:
                    with connection.cursor() as cursor:
                        return method(self, cursor, connection, *args, **kwargs)
            except (psycopg2.OperationalError,
                    psycopg2.ProgrammingError,
                    psycopg2.IntegrityError,
                    psycopg2.DataError,
                    psycopg2.InterfaceError,
                    psycopg2.DatabaseError,
                    Exception) as e:
                if self.logger:
                    self.logger.error(f'데이터베이스 오류: {type(e).__name__}: {e}')
                raise
        return wrapper
    
    @db_exception_handler
    def execute(self, cursor, connection, query: str, params=None, fetch: str = None) -> Optional[pd.DataFrame]:
        '''
        한 번에 하나의 행을 삽입하거나, 단일 SQL 문을 실행할 때 이상적
        '''
        cursor.execute(query, params)
        connection.commit()
        if fetch:
            records = cursor.fetchone() if fetch == 'one' else cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            if fetch == 'one':
                return pd.DataFrame([records], columns=column_names)
            else:
                return pd.DataFrame(records, columns=column_names)
        
    @db_exception_handler
    def execute_many(self, cursor, connection, query: str, params=None):
        '''
        소규모 배치 작업에는 적합
        '''
        cursor.executemany(query, params)
        connection.commit()
    
    @db_exception_handler
    def execute_values(self, cursor, connection, query: str, params=None):
        '''
        대량 삽입에 가장 적합
        '''
        execute_values(cursor, query.as_string(connection), params)
        connection.commit()
        
    def create_tables(self, tickers: List[str]):
        create_table_query = """
            CREATE TABLE IF NOT EXISTS {} (
                timestamp TIMESTAMPTZ NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                transactions INTEGER,
                volume BIGINT,
                vwap DOUBLE PRECISION,
                PRIMARY KEY (timestamp)
            );
        """
        create_hypertable_query = """
            SELECT create_hypertable({}, 'timestamp', if_not_exists => TRUE);
        """
        
        for ticker in tickers:
            query = SQL(create_table_query).format(Identifier(ticker))
            hypertable_query = SQL(create_hypertable_query).format(Literal(ticker))
            self.execute(query)
            self.execute(hypertable_query)
               
    def insert_data(self, tickers: List[str]):
        insert_data_query = '''
            INSERT INTO {} (timestamp, open, high, low, close, transactions, volume, vwap) 
            VALUES %s ON CONFLICT (timestamp) DO NOTHING;
        '''
        
        for ticker in tickers:
            if self.logger:
                self.logger.info(f'{ticker.upper()} 데이터 삽입 시도 중...')
            filepath = f'{DATA_DIR}/raw/1_minute/{ticker.upper()}.json'
            with open(filepath, 'r') as file:
                data = json.load(file)
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            data = df.to_dict(orient='records')
            formatted_data = [(
                    item["timestamp"].to_pydatetime().replace(tzinfo=timezone.utc),
                    item["open"],
                    item["high"],
                    item["low"],
                    item["close"],
                    item["transactions"],
                    item["volume"],
                    item["vwap"]
                ) for item in data
            ]
            query = SQL(insert_data_query).format(Identifier(ticker))
            self.execute_values(query, formatted_data)
            
    def get_ticker_df(self, ticker: str) -> pd.DataFrame:
        query = SQL("SELECT * FROM {}").format(Identifier(ticker))
        return self.execute(query, fetch='all')
        
if __name__ == '__main__':
    database_manager = DatabaseManager()
    tickers = [ticker.lower() for ticker in TICKERS]
    database_manager.create_tables(tickers)
    database_manager.insert_data(tickers)