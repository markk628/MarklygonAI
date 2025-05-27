"""
Eddie Training Monitor
훈련 진행 상황 모니터링 및 결과 확인 스크립트
"""
import os
import time
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

def find_latest_results():
    """가장 최근 훈련 결과 디렉토리 찾기"""
    results_dir = Path("results")
    if not results_dir.exists():
        print("Results directory not found!")
        return None
    
    # eddie_real_data_* 패턴의 디렉토리 찾기
    pattern = "eddie_real_data_*"
    dirs = list(results_dir.glob(pattern))
    
    if not dirs:
        print("No training results found!")
        return None
    
    # 가장 최근 디렉토리 반환
    latest_dir = max(dirs, key=lambda x: x.stat().st_mtime)
    return latest_dir

def monitor_training_logs(results_dir):
    """훈련 로그 모니터링"""
    log_file = results_dir / "training.log"
    
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return
    
    print(f"Monitoring training logs: {log_file}")
    print("=" * 60)
    
    # 로그 파일의 마지막 몇 줄 읽기
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 마지막 20줄 출력
        recent_lines = lines[-20:] if len(lines) > 20 else lines
        for line in recent_lines:
            print(line.strip())
            
    except Exception as e:
        print(f"Error reading log file: {e}")

def check_training_progress(results_dir):
    """훈련 진행 상황 확인"""
    print(f"\nChecking training progress in: {results_dir}")
    print("=" * 60)
    
    # 체크포인트 파일 확인
    checkpoint_files = list(results_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        print(f"Latest checkpoint: {latest_checkpoint.name}")
        
        # 에포크 번호 추출
        epoch_num = latest_checkpoint.stem.split('_')[-1]
        print(f"Current epoch: {epoch_num}")
    else:
        print("No checkpoints found yet...")
    
    # 결과 파일들 확인
    files_to_check = [
        "training_history.json",
        "final_model.pth", 
        "evaluation_results.json",
        "data_analysis.png"
    ]
    
    print("\nFiles status:")
    for filename in files_to_check:
        filepath = results_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            print(f"✅ {filename} ({size:,} bytes, {mtime.strftime('%H:%M:%S')})")
        else:
            print(f"❌ {filename}")

def plot_training_history(results_dir):
    """훈련 히스토리 시각화"""
    history_file = results_dir / "training_history.json"
    
    if not history_file.exists():
        print("Training history not found yet...")
        return
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # 훈련 히스토리 플롯
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Eddie Training Progress', fontsize=16)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss plot
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Signal accuracy plot
        if 'signal_accuracy' in history:
            axes[0, 1].plot(epochs, history['signal_accuracy'], 'g-', label='Signal Accuracy')
            axes[0, 1].set_title('Signal Prediction Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Pattern accuracy plot
        if 'pattern_accuracy' in history:
            axes[1, 0].plot(epochs, history['pattern_accuracy'], 'm-', label='Pattern Accuracy')
            axes[1, 0].set_title('Pattern Analysis Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate plot
        if 'learning_rate' in history:
            axes[1, 1].plot(epochs, history['learning_rate'], 'orange', label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        # 플롯 저장
        plot_path = results_dir / "training_progress.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved: {plot_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error plotting training history: {e}")

def show_evaluation_results(results_dir):
    """평가 결과 표시"""
    eval_file = results_dir / "evaluation_results.json"
    
    if not eval_file.exists():
        print("Evaluation results not found yet...")
        return
    
    try:
        with open(eval_file, 'r') as f:
            results = json.load(f)
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        for metric_name, value in results.items():
            if isinstance(value, dict):
                print(f"\n{metric_name.upper()}:")
                for sub_metric, sub_value in value.items():
                    if isinstance(sub_value, float):
                        print(f"  {sub_metric}: {sub_value:.4f}")
                    else:
                        print(f"  {sub_metric}: {sub_value}")
            else:
                if isinstance(value, float):
                    print(f"{metric_name}: {value:.4f}")
                else:
                    print(f"{metric_name}: {value}")
                    
    except Exception as e:
        print(f"Error reading evaluation results: {e}")

def main():
    """메인 모니터링 함수"""
    print("Eddie Training Monitor")
    print("=" * 60)
    
    # 최신 결과 디렉토리 찾기
    results_dir = find_latest_results()
    if results_dir is None:
        return
    
    print(f"Monitoring: {results_dir}")
    print(f"Started: {datetime.fromtimestamp(results_dir.stat().st_mtime)}")
    
    # 훈련 진행 상황 확인
    check_training_progress(results_dir)
    
    # 로그 모니터링
    monitor_training_logs(results_dir)
    
    # 훈련 히스토리 플롯 (있는 경우)
    plot_training_history(results_dir)
    
    # 평가 결과 표시 (있는 경우)
    show_evaluation_results(results_dir)
    
    print("\n" + "="*60)
    print("Monitoring complete!")
    print("Run this script again to check for updates.")

if __name__ == "__main__":
    main() 