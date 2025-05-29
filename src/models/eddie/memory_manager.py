"""
Eddie 시스템용 지능형 메모리 관리자
메모리 사용량을 모니터링하고 자동으로 최적화하여 안정적인 훈련을 보장합니다.
"""
import psutil
import torch
import gc
import time
import logging
from typing import Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class MemoryThresholds:
    """메모리 임계치 설정 (효율성과 안전성의 균형)"""
    # 시스템 메모리 임계치
    system_warning: float = 80.0      # 80% 경고
    system_critical: float = 88.0     # 88% 위험
    system_emergency: float = 93.0    # 93% 응급
    
    # GPU 메모리 임계치 (RTX 4060 Ti 16GB 로컬 환경)
    gpu_warning: float = 75.0         # 75% 경고 (12GB 사용 시)
    gpu_critical: float = 85.0        # 85% 위험 (13.6GB 사용 시)
    gpu_emergency: float = 90.0       # 90% 응급 (14.4GB 사용 시)


class MemoryManager:
    """
    지능형 메모리 관리자
    시스템과 GPU 메모리를 실시간으로 모니터링하고 최적화
    """
    
    def __init__(self, thresholds: MemoryThresholds = None):
        self.thresholds = thresholds or MemoryThresholds()
        self.logger = logging.getLogger('MemoryManager')
        
        # 모니터링 히스토리
        self.memory_history = []
        self.chunk_size_history = []
        self.batch_size_history = []
        
        # 초기 설정값 저장 (RTX 4060 Ti 16GB 로컬 환경)
        self.initial_chunk_size = 200000  # 청크 크기 유지
        self.initial_batch_size = 64      # RTX 4060 Ti에 맞게 증가
        self.min_chunk_size = 20000       # 최소 청크 크기 유지
        self.min_batch_size = 8           # 최소 배치 크기
        
        # 메모리 상태
        self.last_memory_check = time.time()
        self.memory_check_interval = 5.0  # 5초마다 체크
        
    def get_memory_status(self) -> Dict[str, float]:
        """현재 메모리 상태 조회"""
        # 시스템 메모리
        system_memory = psutil.virtual_memory()
        system_used_percent = system_memory.percent
        system_available_gb = system_memory.available / (1024**3)
        
        # GPU 메모리
        gpu_status = {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'used_percent': 0.0,
            'available_gb': 0.0
        }
        
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # 공유 GPU 환경에서는 reserved 메모리 기준으로 사용률 계산
            gpu_used_percent = (gpu_reserved / gpu_total) * 100
            gpu_available = gpu_total - gpu_reserved
            
            # RTX 4060 Ti에서 안전한 사용 가능 메모리 (전체의 85%까지)
            safe_gpu_limit = gpu_total * 0.85
            safe_available = max(0, safe_gpu_limit - gpu_reserved)
            
            gpu_status = {
                'allocated_gb': gpu_allocated,
                'reserved_gb': gpu_reserved,
                'used_percent': gpu_used_percent,
                'available_gb': gpu_available,
                'safe_available_gb': safe_available,
                'total_gb': gpu_total,
                'safe_limit_gb': safe_gpu_limit
            }
        
        status = {
            'system_used_percent': system_used_percent,
            'system_available_gb': system_available_gb,
            'timestamp': time.time(),
            **{f'gpu_{k}': v for k, v in gpu_status.items()}
        }
        
        # 히스토리에 추가
        self.memory_history.append(status)
        
        # 히스토리 크기 제한 (최근 100개만 유지)
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
        
        return status
    
    def get_memory_pressure_level(self) -> Tuple[str, int]:
        """
        메모리 압박 수준 계산
        Returns:
            level: 'safe', 'warning', 'critical', 'emergency'
            score: 0-100 압박 점수
        """
        status = self.get_memory_status()
        
        system_percent = status['system_used_percent']
        gpu_percent = status.get('gpu_used_percent', 0)
        
        # 시스템 메모리 점수 계산
        if system_percent >= self.thresholds.system_emergency:
            system_score = 100
        elif system_percent >= self.thresholds.system_critical:
            system_score = 85
        elif system_percent >= self.thresholds.system_warning:
            system_score = 70
        else:
            system_score = min(60, system_percent)
        
        # GPU 메모리 점수 계산
        if gpu_percent >= self.thresholds.gpu_emergency:
            gpu_score = 100
        elif gpu_percent >= self.thresholds.gpu_critical:
            gpu_score = 85
        elif gpu_percent >= self.thresholds.gpu_warning:
            gpu_score = 70
        else:
            gpu_score = min(60, gpu_percent)
        
        # 최종 점수 (더 높은 점수 사용)
        final_score = max(system_score, gpu_score)
        
        # 레벨 결정
        if final_score >= 95:
            level = 'emergency'
        elif final_score >= 85:
            level = 'critical'
        elif final_score >= 70:
            level = 'warning'
        else:
            level = 'safe'
        
        return level, int(final_score)
    
    def suggest_chunk_size(self, current_chunk_size: int) -> int:
        """메모리 상태에 따른 적절한 청크 크기 제안 (더 적극적)"""
        level, score = self.get_memory_pressure_level()
        
        if level == 'emergency':
            # 응급: 청크 크기를 1/4로
            new_size = max(self.min_chunk_size, current_chunk_size // 4)
            self.logger.warning(f"🚨 EMERGENCY: Reducing chunk size {current_chunk_size:,} → {new_size:,}")
            
        elif level == 'critical':
            # 위험: 청크 크기를 절반으로
            new_size = max(self.min_chunk_size, current_chunk_size // 2)
            self.logger.warning(f"⚠️ CRITICAL: Reducing chunk size {current_chunk_size:,} → {new_size:,}")
            
        elif level == 'warning':
            # 경고: 청크 크기를 75%로
            new_size = max(self.min_chunk_size, int(current_chunk_size * 0.75))
            self.logger.info(f"🔶 WARNING: Reducing chunk size {current_chunk_size:,} → {new_size:,}")
            
        else:
            # 안전: 메모리 활용도를 높이기 위해 적극적으로 증가
            if score < 60 and current_chunk_size < self.initial_chunk_size:
                new_size = min(self.initial_chunk_size, int(current_chunk_size * 1.2))
                self.logger.info(f"✅ SAFE: Increasing chunk size for efficiency {current_chunk_size:,} → {new_size:,}")
            elif score < 70:
                new_size = current_chunk_size
            else:
                new_size = current_chunk_size
        
        return new_size
    
    def suggest_batch_size(self, current_batch_size: int) -> int:
        """메모리 상태에 따른 적절한 배치 크기 제안 (RTX 4060 Ti 16GB 로컬 환경)"""
        level, score = self.get_memory_pressure_level()
        
        # GPU 메모리 상태도 고려 (RTX 4060 Ti 기준)
        gpu_pressure = 0
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
            if gpu_allocated > 75:  # 12GB 이상 사용 시
                gpu_pressure = 1
            if gpu_allocated > 85:  # 13.6GB 이상 사용 시
                gpu_pressure = 2
        
        if level == 'emergency' or gpu_pressure >= 2:
            # 응급: 배치 크기를 절반으로
            new_size = max(self.min_batch_size, current_batch_size // 2)
            self.logger.warning(f"🚨 EMERGENCY: Reducing batch size {current_batch_size} → {new_size}")
            
        elif level == 'critical' or gpu_pressure >= 1:
            # 위험: 배치 크기를 75%로
            new_size = max(self.min_batch_size, int(current_batch_size * 0.75))
            self.logger.warning(f"⚠️ CRITICAL: Reducing batch size {current_batch_size} → {new_size}")
            
        elif level == 'warning':
            # 경고: 배치 크기를 90%로
            new_size = max(self.min_batch_size, int(current_batch_size * 0.9))
            self.logger.info(f"🔶 WARNING: Reducing batch size {current_batch_size} → {new_size}")
            
        else:
            # 안전: RTX 4060 Ti에서는 적극적으로 활용
            if score < 60 and gpu_pressure == 0 and current_batch_size < self.initial_batch_size * 1.5:
                new_size = min(self.initial_batch_size * 1.5, int(current_batch_size * 1.2))
                self.logger.info(f"✅ SAFE: Increasing batch size for RTX 4060 Ti {current_batch_size} → {new_size}")
            else:
                new_size = current_batch_size
        
        return new_size
    
    def emergency_cleanup(self) -> bool:
        """응급 메모리 정리 - 강화된 버전"""
        self.logger.warning("🧹 Emergency memory cleanup initiated...")
        
        initial_status = self.get_memory_status()
        
        # 1. 다중 가비지 컬렉션 (더 철저하게)
        total_collected = 0
        for i in range(3):  # 3번 반복
            collected = gc.collect()
            total_collected += collected
            if collected == 0:
                break  # 더 이상 정리할 게 없으면 중단
        
        self.logger.info(f"🗑️ Garbage collected: {total_collected} objects (3 passes)")
        
        # 2. GPU 메모리 강제 정리 (공유 환경 고려)
        if torch.cuda.is_available():
            # 현재 GPU 메모리 상태 확인
            current_gpu = torch.cuda.memory_allocated() / (1024**3)
            reserved_gpu = torch.cuda.memory_reserved() / (1024**3)
            
            # GPU 메모리 정리
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # 동기화로 확실히 정리
            torch.cuda.ipc_collect()  # IPC 메모리도 정리
            
            # 정리 후 상태 확인
            after_gpu = torch.cuda.memory_allocated() / (1024**3)
            freed_gpu = current_gpu - after_gpu
            
            self.logger.info(f"🔥 GPU cleanup: {current_gpu:.2f}GB → {after_gpu:.2f}GB (freed: {freed_gpu:.2f}GB)")
        
        # 3. NumPy/Python 내부 메모리 정리
        try:
            import numpy as np
            # NumPy의 임시 배열들 정리
            for _ in range(5):
                temp = np.array([1])
                del temp
        except:
            pass
        
        # 4. 시스템 레벨 메모리 압축 시도
        try:
            import ctypes
            if hasattr(ctypes, 'windll'):  # Windows에서만
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        except:
            pass
        
        # 5. 추가 대기 시간 (정리 완료)
        time.sleep(2)
        
        # 결과 확인
        final_status = self.get_memory_status()
        
        system_freed = initial_status['system_used_percent'] - final_status['system_used_percent']
        gpu_freed = initial_status.get('gpu_used_percent', 0) - final_status.get('gpu_used_percent', 0)
        
        self.logger.info(f"🎯 Enhanced cleanup results:")
        self.logger.info(f"  System: {system_freed:.1f}% freed ({initial_status['system_used_percent']:.1f}% → {final_status['system_used_percent']:.1f}%)")
        self.logger.info(f"  GPU: {gpu_freed:.1f}% freed")
        self.logger.info(f"  Available: {final_status['system_available_gb']:.1f}GB")
        
        # 성공 기준: 시스템 메모리가 85% 미만으로 내려갔는지
        return final_status['system_used_percent'] < 85.0
    
    def wait_for_memory_availability(self, target_memory_mb: int = 1000, timeout: int = 30) -> bool:
        """
        지정된 메모리가 사용 가능해질 때까지 대기
        
        Args:
            target_memory_mb: 필요한 메모리 크기 (MB)
            timeout: 최대 대기 시간 (초)
            
        Returns:
            bool: 메모리 확보 성공 여부
        """
        start_time = time.time()
        target_memory_gb = target_memory_mb / 1024
        
        while time.time() - start_time < timeout:
            status = self.get_memory_status()
            
            if status['system_available_gb'] >= target_memory_gb:
                return True
            
            # 메모리 부족 시 정리 시도
            level, _ = self.get_memory_pressure_level()
            if level in ['critical', 'emergency']:
                self.emergency_cleanup()
            
            time.sleep(2)
        
        return False
    
    def monitor_and_log(self, stage: str = "Processing") -> None:
        """메모리 상태 모니터링 및 로깅"""
        current_time = time.time()
        
        # 일정 간격으로만 체크
        if current_time - self.last_memory_check < self.memory_check_interval:
            return
        
        self.last_memory_check = current_time
        
        status = self.get_memory_status()
        level, score = self.get_memory_pressure_level()
        
        # 레벨에 따른 이모지 선택
        emoji_map = {
            'safe': '💚',
            'warning': '🟡',
            'critical': '🟠',
            'emergency': '🔴'
        }
        
        emoji = emoji_map.get(level, '💚')
        
        # 로그 출력
        self.logger.info(
            f"{emoji} [{stage}] Memory Status - "
            f"System: {status['system_used_percent']:.1f}%, "
            f"GPU: {status.get('gpu_used_percent', 0):.1f}%, "
            f"Pressure: {level.upper()} ({score})"
        )
        
        # 위험 수준일 때 추가 정보
        if level in ['critical', 'emergency']:
            self.logger.warning(
                f"💾 Available - System: {status['system_available_gb']:.1f}GB, "
                f"GPU: {status.get('gpu_available_gb', 0):.1f}GB"
            )
    
    def get_memory_report(self) -> str:
        """메모리 사용 리포트 생성"""
        if not self.memory_history:
            return "No memory data available"
        
        recent_status = self.memory_history[-1]
        level, score = self.get_memory_pressure_level()
        
        report = [
            "=" * 50,
            "📊 MEMORY USAGE REPORT",
            "=" * 50,
            f"Current Status: {level.upper()} (Score: {score})",
            f"System Memory: {recent_status['system_used_percent']:.1f}%",
            f"Available System Memory: {recent_status['system_available_gb']:.1f}GB",
        ]
        
        if torch.cuda.is_available():
            report.extend([
                f"GPU Memory: {recent_status.get('gpu_used_percent', 0):.1f}%",
                f"Available GPU Memory: {recent_status.get('gpu_available_gb', 0):.1f}GB",
            ])
        
        # 히스토리 분석
        if len(self.memory_history) >= 5:
            recent_system = [h['system_used_percent'] for h in self.memory_history[-5:]]
            trend = "📈 Rising" if recent_system[-1] > recent_system[0] else "📉 Falling"
            avg_usage = np.mean(recent_system)
            
            report.extend([
                "=" * 30,
                f"Recent Trend: {trend}",
                f"Average Usage (last 5): {avg_usage:.1f}%",
            ])
        
        return "\n".join(report)


# 전역 메모리 관리자 인스턴스
_global_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """전역 메모리 관리자 인스턴스 반환"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager

def check_memory_and_suggest(current_chunk_size: int, current_batch_size: int) -> Tuple[int, int]:
    """
    메모리 상태를 확인하고 적절한 크기를 제안
    
    Returns:
        (suggested_chunk_size, suggested_batch_size)
    """
    manager = get_memory_manager()
    
    new_chunk_size = manager.suggest_chunk_size(current_chunk_size)
    new_batch_size = manager.suggest_batch_size(current_batch_size)
    
    return new_chunk_size, new_batch_size

def safe_memory_operation(operation_name: str = "Operation"):
    """
    안전한 메모리 연산을 위한 데코레이터
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_memory_manager()
            
            # 연산 전 메모리 상태 체크
            manager.monitor_and_log(f"Before {operation_name}")
            level, _ = manager.get_memory_pressure_level()
            
            # 위험 수준일 때 정리
            if level in ['critical', 'emergency']:
                manager.emergency_cleanup()
            
            try:
                result = func(*args, **kwargs)
                manager.monitor_and_log(f"After {operation_name}")
                return result
            except MemoryError as e:
                manager.logger.error(f"Memory error in {operation_name}: {e}")
                manager.emergency_cleanup()
                raise
            
        return wrapper
    return decorator 