import os
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