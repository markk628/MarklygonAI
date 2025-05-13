"""
로깅 모듈
"""
import os
import logging
from typing import Optional, Dict, Any


class Logger:
    """
    로깅을 담당하는 클래스
    """
    
    def __init__(
        self, 
        log_file: Optional[str] = None, 
        log_level: int = logging.INFO,
        console_output: bool = True
    ):
        """
        Logger 클래스 초기화
        
        Args:
            log_file: 로그 파일 경로 (옵션)
            log_level: 로깅 레벨
            console_output: 콘솔 출력 여부
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # 기존 핸들러 제거
        
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 파일 로거 설정
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # 콘솔 로거 설정
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str) -> None:
        """
        디버그 레벨 로깅
        
        Args:
            message: 로그 메시지
        """
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """
        정보 레벨 로깅
        
        Args:
            message: 로그 메시지
        """
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """
        경고 레벨 로깅
        
        Args:
            message: 로그 메시지
        """
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """
        오류 레벨 로깅
        
        Args:
            message: 로그 메시지
        """
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """
        심각한 오류 레벨 로깅
        
        Args:
            message: 로그 메시지
        """
        self.logger.critical(message)
    
    def log_dict(self, data: Dict[str, Any], prefix: str = "") -> None:
        """
        딕셔너리 데이터 로깅
        
        Args:
            data: 로깅할 딕셔너리 데이터
            prefix: 접두사 (옵션)
        """
        for key, value in data.items():
            if isinstance(value, dict):
                self.log_dict(value, f"{prefix}{key}.")
            else:
                self.logger.info(f"{prefix}{key}: {value}")
    
    def log_exception(self, e: Exception, message: Optional[str] = None) -> None:
        """
        예외 정보 로깅
        
        Args:
            e: 예외 객체
            message: 추가 메시지 (옵션)
        """
        if message:
            self.logger.exception(f"{message}: {str(e)}")
        else:
            self.logger.exception(str(e)) 