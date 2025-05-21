import os
import pandas as pd
from typing import Union
from pathlib import Path

def create_directory(directory_path: Union[str, Path]) -> None:
    """
    디렉토리가 존재하지 않으면 생성하는 함수
    
    Args:
        directory_path: 생성할 디렉토리 경로
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"디렉토리 생성됨: {directory_path}")
        
def save_to_csv(df: pd.DataFrame, file_path: Union[str, Path], index: bool = True) -> None:
    """
    데이터프레임을 CSV 파일로 저장하는 함수
    
    Args:
        df: 저장할 데이터프레임
        file_path: 저장할 파일 경로
        index: 인덱스 포함 여부
    """
    create_directory(os.path.dirname(file_path))
    df.to_csv(file_path, index=index, encoding='utf-8-sig')
    print(f"파일 저장됨: {file_path}")

def load_from_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    CSV 파일에서 데이터프레임을 로드하는 함수
    
    Args:
        file_path: 로드할 파일 경로
        
    Returns:
        로드된 데이터프레임
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없음: {file_path}")
    
    return pd.read_csv(file_path, encoding='utf-8-sig')

def format_duration(seconds: float):
    '''
    초 단위를 가장 적절한 시간 단위로 변환하여 사람이 읽기 쉬운 문자열로 반환하는 함수
    
    Args:
        seconds (float): 초 단위 시간

    Returns:
        적절한 시간 단위로 표현된 사람이 읽기 쉬운 시간 문자열
    '''
    days = int(seconds // 86400)
    seconds %= 86400
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60

    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:
        parts.append(f"{seconds:.2f} second{'s' if seconds != 1 else ''}")

    return ' '.join(parts)