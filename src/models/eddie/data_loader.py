"""
Eddie Data Loader for MarklygonAI
실제 MarklygonAI 데이터를 Eddie 시스템에 맞게 로드 및 전처리
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pickle
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from config import EddieConfig, DEFAULT_CONFIG


class MarklygonDataProcessor:
    """
    MarklygonAI 데이터를 Eddie 시스템용으로 전처리하는 클래스
    """
    
    def __init__(self, config: EddieConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger('MarklygonDataProcessor')
        
        # Scalers and transformers
        self.feature_scaler = None
        self.target_scaler = None
        self.pca = None
        
        # Feature information
        self.feature_columns = None
        self.target_columns = ['target']  # Target column from the data
        
        # Data statistics
        self.data_stats = {}
        
    def load_stock_data(self, data_path: str, symbols: List[str] = None) -> pd.DataFrame:
        """
        주식 데이터 로드
        
        Args:
            data_path: 데이터 경로 (feature_engineered 또는 processed)
            symbols: 로드할 심볼 리스트 (None이면 모든 데이터)
            
        Returns:
            combined_data: 결합된 주식 데이터
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        all_data = []
        
        # If symbols not specified, try to find all available files
        if symbols is None:
            if (data_path / "feature_engineered").exists():
                symbols = [f.stem for f in (data_path / "feature_engineered").glob("*.csv")]
            elif (data_path / "processed").exists():
                symbols = [f.stem for f in (data_path / "processed").glob("*.parquet")]
            else:
                # Look for files directly in the path
                csv_files = list(data_path.glob("*.csv"))
                parquet_files = list(data_path.glob("*.parquet"))
                
                if csv_files:
                    symbols = [f.stem for f in csv_files]
                elif parquet_files:
                    symbols = [f.stem for f in parquet_files]
                else:
                    raise FileNotFoundError("No data files found")
        
        self.logger.info(f"Loading data for symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
        
        for symbol in symbols:
            try:
                # Try different file formats and locations
                file_paths = [
                    data_path / "feature_engineered" / f"{symbol}.csv",
                    data_path / "processed" / f"{symbol}.parquet",
                    data_path / f"{symbol}.csv",
                    data_path / f"{symbol}.parquet"
                ]
                
                data = None
                for file_path in file_paths:
                    if file_path.exists():
                        if file_path.suffix == '.csv':
                            data = pd.read_csv(file_path)
                        elif file_path.suffix == '.parquet':
                            data = pd.read_parquet(file_path)
                        break
                
                if data is None:
                    self.logger.warning(f"No data file found for {symbol}")
                    continue
                
                # Add symbol column
                data['symbol'] = symbol
                
                # Convert timestamp if needed
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data = data.sort_values('timestamp')
                
                all_data.append(data)
                self.logger.debug(f"Loaded {len(data)} rows for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data could be loaded")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"Total data loaded: {len(combined_data)} rows, {len(combined_data.columns)} columns")
        
        return combined_data
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Eddie 시스템용 피처 준비
        
        Args:
            data: 원본 데이터
            
        Returns:
            features_df: 준비된 피처 데이터프레임
            feature_names: 피처 이름 리스트
        """
        # 기본적으로 제외할 컬럼들
        exclude_columns = [
            'timestamp', 'symbol', 'target',
            # 원본 OHLCV 데이터 (이미 변환된 지표들을 사용)
            'open', 'high', 'low', 'close', 'volume', 'transactions', 'vwap'
        ]
        
        # 사용할 피처 선택
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # 기술적 지표만 선택 (Eddie 시스템은 TA-Lib 지표를 사용)
        talib_indicators = []
        time_features = []
        lag_features = []
        rolling_features = []
        
        for col in feature_columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in [
                'rsi', 'macd', 'stoch', 'roc', 'ultosc', 'plusdi', 'minusdi', 
                'adx', 'cci', 'ema', 'sma', 'obv', 'mfi', 'bband', 'atr'
            ]):
                talib_indicators.append(col)
            elif any(time_feat in col_lower for time_feat in [
                'minute', 'hour', 'day', 'month', 'quarter', 'sin', 'cos'
            ]):
                time_features.append(col)
            elif 'lag' in col_lower:
                lag_features.append(col)
            elif 'rolling' in col_lower or 'diff' in col_lower or 'pct_change' in col_lower:
                rolling_features.append(col)
        
        # Eddie 시스템에서 사용할 피처 선택
        selected_features = talib_indicators + time_features + lag_features[:10] + rolling_features[:10]
        
        # 데이터에서 실제 존재하는 피처만 선택
        selected_features = [col for col in selected_features if col in data.columns]
        
        self.logger.info(f"Selected {len(selected_features)} features:")
        self.logger.info(f"  - TA-Lib indicators: {len(talib_indicators)}")
        self.logger.info(f"  - Time features: {len(time_features)}")
        self.logger.info(f"  - Lag features: {len([f for f in selected_features if 'lag' in f.lower()])}")
        self.logger.info(f"  - Rolling features: {len([f for f in selected_features if 'rolling' in f.lower()])}")
        
        # 피처 데이터 추출
        features_df = data[selected_features].copy()
        
        # 결측값 처리
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        self.feature_columns = selected_features
        
        return features_df, selected_features
    
    def create_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Eddie 시스템용 타겟 변수 생성
        
        Args:
            data: 원본 데이터
            
        Returns:
            targets_df: 타겟 변수 데이터프레임
        """
        targets = {}
        
        # 1. Sell Intensity (기존 target 컬럼이 있으면 사용, 없으면 생성)
        if 'target' in data.columns:
            # 기존 target을 sell_intensity로 사용 (범위를 [-1, 1]로 조정)
            targets['sell_intensity'] = np.tanh(data['target'].fillna(0))
        else:
            # close 가격을 이용해 미래 수익률 기반 sell_intensity 생성
            if 'close' in data.columns:
                # 5분 후 수익률 계산
                future_return = data.groupby('symbol')['close'].pct_change(5).shift(-5)
                # 수익률을 sell_intensity로 변환 (음수면 매도 신호)
                targets['sell_intensity'] = -np.tanh(future_return * 10).fillna(0)
            else:
                # 기본값으로 0 설정
                targets['sell_intensity'] = np.zeros(len(data))
        
        # 2. Market Regime (변동성 기반)
        if 'close' in data.columns:
            # 단기 변동성 계산
            volatility = data.groupby('symbol')['close'].pct_change().rolling(20).std()
            
            # 변동성을 4개 구간으로 나누어 시장 상태 분류
            vol_quantiles = volatility.quantile([0.25, 0.5, 0.75])
            regime = np.zeros(len(data))
            regime[volatility <= vol_quantiles[0.25]] = 0  # Low volatility
            regime[(volatility > vol_quantiles[0.25]) & (volatility <= vol_quantiles[0.5])] = 1  # Medium-low
            regime[(volatility > vol_quantiles[0.5]) & (volatility <= vol_quantiles[0.75])] = 2  # Medium-high
            regime[volatility > vol_quantiles[0.75]] = 3  # High volatility
            
            targets['market_regime'] = regime
        else:
            targets['market_regime'] = np.zeros(len(data))
        
        # 3. Volatility (정규화된 변동성)
        if 'atr_14_1min' in data.columns:
            # ATR을 이용한 변동성
            atr_volatility = data['atr_14_1min'].fillna(0)
            targets['volatility'] = (atr_volatility - atr_volatility.min()) / (atr_volatility.max() - atr_volatility.min() + 1e-8)
        elif 'close' in data.columns:
            # 가격 변동성 기반
            price_volatility = data.groupby('symbol')['close'].pct_change().rolling(14).std()
            targets['volatility'] = (price_volatility - price_volatility.min()) / (price_volatility.max() - price_volatility.min() + 1e-8)
        else:
            targets['volatility'] = np.zeros(len(data))
        
        # 4. Uncertainty (모델이 추정해야 할 실제 불확실성의 대리변수)
        # 가격 변동의 예측 불가능성을 측정
        if 'close' in data.columns:
            returns = data.groupby('symbol')['close'].pct_change()
            rolling_std = returns.rolling(20).std()
            rolling_mean = returns.rolling(20).mean()
            
            # 표준편차와 평균의 비율로 불확실성 측정
            uncertainty = np.abs(rolling_std / (np.abs(rolling_mean) + 1e-8))
            targets['uncertainty'] = np.tanh(uncertainty).fillna(0.5)  # 기본값 0.5
        else:
            targets['uncertainty'] = np.full(len(data), 0.5)
        
        targets_df = pd.DataFrame(targets)
        
        # 결측값 처리
        targets_df = targets_df.fillna(0)
        
        self.logger.info(f"Created targets with shapes:")
        for col in targets_df.columns:
            self.logger.info(f"  - {col}: range [{targets_df[col].min():.3f}, {targets_df[col].max():.3f}]")
        
        return targets_df
    
    def normalize_features(self, features_df: pd.DataFrame, fit_scaler: bool = True) -> np.ndarray:
        """
        피처 정규화
        
        Args:
            features_df: 피처 데이터프레임
            fit_scaler: 스케일러 학습 여부
            
        Returns:
            normalized_features: 정규화된 피처 배열
        """
        if fit_scaler or self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            normalized_features = self.feature_scaler.fit_transform(features_df.values)
            self.logger.info("Feature scaler fitted")
        else:
            normalized_features = self.feature_scaler.transform(features_df.values)
        
        return normalized_features
    
    def apply_pca(self, features: np.ndarray, fit_pca: bool = True) -> np.ndarray:
        """
        PCA 차원 축소 적용
        
        Args:
            features: 입력 피처 배열
            fit_pca: PCA 학습 여부
            
        Returns:
            pca_features: PCA 적용된 피처 배열
        """
        n_components = min(self.config.talib.pca_components, features.shape[1])
        
        if fit_pca or self.pca is None:
            self.pca = PCA(n_components=n_components)
            pca_features = self.pca.fit_transform(features)
            
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            self.logger.info(f"PCA fitted: {features.shape[1]} -> {n_components} components")
            self.logger.info(f"Explained variance: {explained_variance:.3f}")
        else:
            pca_features = self.pca.transform(features)
        
        return pca_features
    
    def create_sequences(
        self, 
        features: np.ndarray, 
        targets: pd.DataFrame,
        metadata: pd.DataFrame,
        seq_len: int = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.DataFrame]:
        """
        시퀀스 데이터 생성
        
        Args:
            features: 피처 배열
            targets: 타겟 데이터프레임
            metadata: 메타데이터 (symbol, timestamp 등)
            seq_len: 시퀀스 길이
            
        Returns:
            sequences: 시퀀스 피처 배열
            sequence_targets: 시퀀스 타겟 딕셔너리
            sequence_metadata: 시퀀스 메타데이터
        """
        if seq_len is None:
            seq_len = self.config.signal_generator.seq_len
        
        # 심볼별로 시퀀스 생성
        all_sequences = []
        all_targets = {col: [] for col in targets.columns}
        all_metadata = []
        
        if 'symbol' in metadata.columns:
            symbols = metadata['symbol'].unique()
        else:
            # 심볼 정보가 없으면 전체 데이터를 하나의 시퀀스로 처리
            symbols = ['ALL']
            metadata = metadata.copy()
            metadata['symbol'] = 'ALL'
        
        for symbol in symbols:
            if symbol == 'ALL':
                mask = np.ones(len(features), dtype=bool)
            else:
                mask = metadata['symbol'] == symbol
            
            symbol_features = features[mask]
            symbol_targets = targets.loc[mask]
            symbol_metadata = metadata.loc[mask]
            
            if len(symbol_features) < seq_len:
                self.logger.warning(f"Not enough data for {symbol}: {len(symbol_features)} < {seq_len}")
                continue
            
            # 시퀀스 생성
            for i in range(len(symbol_features) - seq_len + 1):
                # 피처 시퀀스
                seq_features = symbol_features[i:i + seq_len]
                all_sequences.append(seq_features)
                
                # 타겟 (마지막 시점)
                for col in targets.columns:
                    all_targets[col].append(symbol_targets.iloc[i + seq_len - 1][col])
                
                # 메타데이터 (마지막 시점)
                all_metadata.append(symbol_metadata.iloc[i + seq_len - 1])
        
        if not all_sequences:
            raise ValueError("No sequences could be created")
        
        # 배열로 변환
        sequences = np.array(all_sequences)
        sequence_targets = {col: np.array(values) for col, values in all_targets.items()}
        sequence_metadata = pd.DataFrame(all_metadata).reset_index(drop=True)
        
        self.logger.info(f"Created {len(sequences)} sequences of length {seq_len}")
        self.logger.info(f"Sequence shape: {sequences.shape}")
        
        return sequences, sequence_targets, sequence_metadata
    
    def train_test_split(
        self,
        sequences: np.ndarray,
        targets: Dict[str, np.ndarray],
        metadata: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[Dict, Dict, Dict]:
        """
        시간 순서를 고려한 train/validation/test 분할
        
        Args:
            sequences: 시퀀스 배열
            targets: 타겟 딕셔너리
            metadata: 메타데이터
            test_size: 테스트 비율
            val_size: 검증 비율
            
        Returns:
            train_data, val_data, test_data: 분할된 데이터 딕셔너리
        """
        total_len = len(sequences)
        
        # 시간 순서를 유지하며 분할
        train_end = int(total_len * (1 - test_size - val_size))
        val_end = int(total_len * (1 - test_size))
        
        def create_split(start_idx: int, end_idx: int) -> Dict:
            return {
                'features': sequences[start_idx:end_idx],
                'targets': {k: v[start_idx:end_idx] for k, v in targets.items()},
                'metadata': metadata.iloc[start_idx:end_idx].reset_index(drop=True)
            }
        
        train_data = create_split(0, train_end)
        val_data = create_split(train_end, val_end)
        test_data = create_split(val_end, total_len)
        
        self.logger.info(f"Data split:")
        self.logger.info(f"  Train: {len(train_data['features'])} samples")
        self.logger.info(f"  Validation: {len(val_data['features'])} samples")
        self.logger.info(f"  Test: {len(test_data['features'])} samples")
        
        return train_data, val_data, test_data
    
    def save_preprocessor(self, save_path: str):
        """전처리기 저장"""
        save_data = {
            'config': self.config,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'pca': self.pca,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'data_stats': self.data_stats
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"Preprocessor saved to {save_path}")
    
    def load_preprocessor(self, load_path: str):
        """전처리기 로드"""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.config = save_data['config']
        self.feature_scaler = save_data['feature_scaler']
        self.target_scaler = save_data.get('target_scaler')
        self.pca = save_data['pca']
        self.feature_columns = save_data['feature_columns']
        self.target_columns = save_data['target_columns']
        self.data_stats = save_data.get('data_stats', {})
        
        self.logger.info(f"Preprocessor loaded from {load_path}")


class MarklygonDataset(Dataset):
    """
    Eddie 시스템용 PyTorch 데이터셋
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: Dict[str, np.ndarray],
        metadata: pd.DataFrame = None
    ):
        self.features = torch.FloatTensor(features)
        self.targets = {k: torch.FloatTensor(v) for k, v in targets.items()}
        self.metadata = metadata
        
        # 길이 확인
        lengths = [len(self.features)] + [len(v) for v in self.targets.values()]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError("Features and targets must have the same length")
        
        self.length = len(self.features)
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'features': self.features[idx],
            **{k: v[idx] for k, v in self.targets.items()}
        }
        
        # Don't include metadata in DataLoader to avoid collation issues
        # Metadata can be accessed separately if needed
        # if self.metadata is not None:
        #     item['metadata'] = self.metadata.iloc[idx].to_dict()
        
        return item


def create_data_loaders(
    data_path: str,
    config: EddieConfig = None,
    symbols: List[str] = None,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, MarklygonDataProcessor]:
    """
    MarklygonAI 데이터로부터 Eddie 시스템용 데이터 로더 생성
    
    Args:
        data_path: 데이터 경로
        config: Eddie 설정
        symbols: 사용할 심볼 리스트
        batch_size: 배치 크기
        num_workers: 워커 수
        
    Returns:
        train_loader, val_loader, test_loader, processor
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # 데이터 프로세서 초기화
    processor = MarklygonDataProcessor(config)
    
    # 1. 데이터 로드
    logging.info("Loading stock data...")
    raw_data = processor.load_stock_data(data_path, symbols)
    
    # 2. 피처 준비
    logging.info("Preparing features...")
    features_df, feature_names = processor.prepare_features(raw_data)
    
    # 3. 타겟 생성
    logging.info("Creating targets...")
    targets_df = processor.create_targets(raw_data)
    
    # 4. 피처 정규화
    logging.info("Normalizing features...")
    normalized_features = processor.normalize_features(features_df, fit_scaler=True)
    
    # 5. PCA 적용 (설정에 따라)
    if config.talib.use_pca:
        logging.info("Applying PCA...")
        pca_features = processor.apply_pca(normalized_features, fit_pca=True)
    else:
        pca_features = normalized_features
    
    # 6. 메타데이터 준비
    metadata_columns = ['symbol']
    if 'timestamp' in raw_data.columns:
        metadata_columns.append('timestamp')
    
    metadata = raw_data[metadata_columns].reset_index(drop=True)
    
    # 7. 시퀀스 생성
    logging.info("Creating sequences...")
    sequences, sequence_targets, sequence_metadata = processor.create_sequences(
        pca_features, targets_df, metadata, config.signal_generator.seq_len
    )
    
    # 8. Train/Val/Test 분할
    logging.info("Splitting data...")
    train_data, val_data, test_data = processor.train_test_split(
        sequences, sequence_targets, sequence_metadata
    )
    
    # 9. 데이터셋 생성
    train_dataset = MarklygonDataset(
        train_data['features'], train_data['targets'], train_data['metadata']
    )
    val_dataset = MarklygonDataset(
        val_data['features'], val_data['targets'], val_data['metadata']
    )
    test_dataset = MarklygonDataset(
        test_data['features'], test_data['targets'], test_data['metadata']
    )
    
    # 10. 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    logging.info("Data loaders created successfully!")
    
    return train_loader, val_loader, test_loader, processor 