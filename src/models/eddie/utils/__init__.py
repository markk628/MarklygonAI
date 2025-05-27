"""
Eddie Utilities Package
Eddie 시스템을 위한 유틸리티 모듈들
"""

from .integration import (
    EddiePredictor,
    EddieDataProcessor, 
    EddieRealTimeInterface,
    load_eddie_predictor,
    quick_predict,
    create_realtime_interface
)

__all__ = [
    'EddiePredictor',
    'EddieDataProcessor',
    'EddieRealTimeInterface', 
    'load_eddie_predictor',
    'quick_predict',
    'create_realtime_interface'
]

__version__ = "1.0.0"
__author__ = "Eddie - MarklygonAI Team" 