"""
Eddie Model Comparison Utilities
다양한 모델 버전과 설정을 비교하는 유틸리티
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

try:
    from .integration import EddiePredictor, EddieDataProcessor
    from ..config import EddieConfig
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.integration import EddiePredictor, EddieDataProcessor
    from config import EddieConfig


class ModelComparator:
    """
    Eddie 모델 비교 및 벤치마킹 도구
    """
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('ModelComparator')
        
        # Comparison results storage
        self.models = {}
        self.comparison_results = {}
        
    def add_model(
        self,
        name: str,
        model_path: str,
        config: EddieConfig = None,
        description: str = ""
    ):
        """비교할 모델 추가"""
        try:
            predictor = EddiePredictor(model_path, config)
            
            self.models[name] = {
                'predictor': predictor,
                'config': config,
                'description': description,
                'model_path': model_path
            }
            
            self.logger.info(f"Model '{name}' added for comparison")
            
        except Exception as e:
            self.logger.error(f"Failed to add model '{name}': {e}")
    
    def compare_models(
        self,
        test_data: Dict[str, np.ndarray],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        모델들을 테스트 데이터로 비교
        
        Args:
            test_data: {'features': X, 'targets': y} 형식의 테스트 데이터
            metrics: 계산할 메트릭 리스트
            
        Returns:
            comparison_results: 비교 결과
        """
        if metrics is None:
            metrics = ['mse', 'mae', 'r2', 'inference_time']
        
        features = test_data['features']  # (N, seq_len, num_features)
        targets = test_data['targets']    # Dict with various targets
        
        results = {}
        
        for model_name, model_info in self.models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            try:
                # Run predictions
                model_results = self._evaluate_single_model(
                    model_info['predictor'],
                    features,
                    targets,
                    metrics
                )
                
                results[model_name] = {
                    **model_results,
                    'description': model_info['description'],
                    'config': model_info['config']
                }
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.comparison_results = results
        return results
    
    def _evaluate_single_model(
        self,
        predictor: EddiePredictor,
        features: np.ndarray,
        targets: Dict[str, np.ndarray],
        metrics: List[str]
    ) -> Dict[str, float]:
        """단일 모델 평가"""
        import time
        
        # Batch prediction with timing
        start_time = time.time()
        predictions = predictor.predict_batch(features, batch_size=32)
        inference_time = (time.time() - start_time) / len(features)  # Per sample
        
        # Extract predictions
        sell_intensity_pred = np.array([p['sell_intensity'] for p in predictions])
        volatility_pred = np.array([p['volatility'] for p in predictions])
        uncertainty_pred = np.array([p['uncertainty'] for p in predictions])
        
        results = {'inference_time': inference_time}
        
        # Calculate metrics
        if 'mse' in metrics and 'sell_intensity' in targets:
            results['sell_intensity_mse'] = mean_squared_error(
                targets['sell_intensity'], sell_intensity_pred
            )
        
        if 'mae' in metrics and 'sell_intensity' in targets:
            results['sell_intensity_mae'] = mean_absolute_error(
                targets['sell_intensity'], sell_intensity_pred
            )
        
        if 'r2' in metrics and 'sell_intensity' in targets:
            results['sell_intensity_r2'] = r2_score(
                targets['sell_intensity'], sell_intensity_pred
            )
        
        # Volatility metrics
        if 'volatility' in targets:
            results['volatility_mse'] = mean_squared_error(
                targets['volatility'], volatility_pred
            )
            results['volatility_r2'] = r2_score(
                targets['volatility'], volatility_pred
            )
        
        # Uncertainty calibration
        if 'uncertainty' in targets:
            # Simple uncertainty calibration score
            uncertainty_true = targets['uncertainty']
            uncertainty_corr = np.corrcoef(uncertainty_true, uncertainty_pred)[0, 1]
            results['uncertainty_calibration'] = uncertainty_corr if not np.isnan(uncertainty_corr) else 0.0
        
        # Market regime accuracy (if available)
        if 'market_regime' in targets:
            regime_pred = np.array([p.get('market_regime', 0) for p in predictions])
            results['regime_accuracy'] = accuracy_score(targets['market_regime'], regime_pred)
        
        return results
    
    def generate_comparison_report(
        self,
        save_plots: bool = True,
        plot_format: str = 'png'
    ) -> Dict[str, Any]:
        """비교 리포트 생성"""
        if not self.comparison_results:
            raise ValueError("No comparison results available. Run compare_models first.")
        
        # Create summary table
        summary_df = self._create_summary_table()
        
        # Generate plots
        if save_plots:
            self._generate_comparison_plots(plot_format)
        
        # Calculate rankings
        rankings = self._calculate_rankings()
        
        # Create detailed report
        report = {
            'summary_table': summary_df,
            'rankings': rankings,
            'detailed_results': self.comparison_results,
            'best_models': self._find_best_models(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = self.results_dir / 'model_comparison_report.pkl'
        with open(report_path, 'wb') as f:
            pickle.dump(report, f)
        
        # Save summary as CSV
        summary_df.to_csv(self.results_dir / 'model_comparison_summary.csv')
        
        self.logger.info(f"Comparison report saved to {self.results_dir}")
        
        return report
    
    def _create_summary_table(self) -> pd.DataFrame:
        """요약 테이블 생성"""
        data = []
        
        for model_name, results in self.comparison_results.items():
            if 'error' in results:
                continue
            
            row = {'Model': model_name}
            
            # Add all numeric metrics
            for key, value in results.items():
                if isinstance(value, (int, float)) and key != 'config':
                    row[key] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _generate_comparison_plots(self, plot_format: str = 'png'):
        """비교 플롯 생성"""
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Performance metrics comparison
        self._plot_performance_metrics(plot_format)
        
        # 2. Inference time comparison
        self._plot_inference_times(plot_format)
        
        # 3. Radar chart for overall comparison
        self._plot_radar_comparison(plot_format)
    
    def _plot_performance_metrics(self, plot_format: str):
        """성능 메트릭 비교 플롯"""
        metrics = ['sell_intensity_mse', 'sell_intensity_r2', 'volatility_mse', 'regime_accuracy']
        available_metrics = []
        
        # Check which metrics are available
        for metric in metrics:
            if any(metric in results for results in self.comparison_results.values() 
                   if 'error' not in results):
                available_metrics.append(metric)
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:4]):
            if i >= 4:
                break
            
            model_names = []
            values = []
            
            for model_name, results in self.comparison_results.items():
                if 'error' not in results and metric in results:
                    model_names.append(model_name)
                    values.append(results[metric])
            
            if values:
                axes[i].bar(model_names, values)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(available_metrics), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'performance_comparison.{plot_format}', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_inference_times(self, plot_format: str):
        """추론 시간 비교 플롯"""
        model_names = []
        inference_times = []
        
        for model_name, results in self.comparison_results.items():
            if 'error' not in results and 'inference_time' in results:
                model_names.append(model_name)
                inference_times.append(results['inference_time'] * 1000)  # Convert to ms
        
        if not inference_times:
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, inference_times)
        plt.title('Model Inference Time Comparison')
        plt.ylabel('Inference Time (ms per sample)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars, inference_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{time:.2f}ms', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'inference_time_comparison.{plot_format}',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_radar_comparison(self, plot_format: str):
        """레이더 차트 비교"""
        try:
            import numpy as np
            from math import pi
            
            # Select key metrics for radar chart
            radar_metrics = [
                'sell_intensity_r2', 'volatility_r2', 'regime_accuracy', 
                'uncertainty_calibration'
            ]
            
            # Filter available metrics and models
            available_models = []
            available_metrics = []
            
            for metric in radar_metrics:
                if any(metric in results for results in self.comparison_results.values() 
                       if 'error' not in results):
                    available_metrics.append(metric)
            
            for model_name, results in self.comparison_results.items():
                if 'error' not in results and any(metric in results for metric in available_metrics):
                    available_models.append(model_name)
            
            if len(available_models) < 2 or len(available_metrics) < 3:
                return
            
            # Create radar chart
            N = len(available_metrics)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(available_models)))
            
            for i, model_name in enumerate(available_models):
                values = []
                for metric in available_metrics:
                    value = self.comparison_results[model_name].get(metric, 0)
                    # Normalize to 0-1 scale for radar chart
                    if 'mse' in metric.lower():
                        value = max(0, 1 - value)  # Inverse for MSE (lower is better)
                    values.append(max(0, min(1, value)))
                
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title('Model Performance Radar Chart', size=16, y=1.08)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / f'radar_comparison.{plot_format}',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Could not generate radar chart: {e}")
    
    def _calculate_rankings(self) -> Dict[str, List[str]]:
        """모델 순위 계산"""
        rankings = {}
        
        # Define metrics and whether higher is better
        metric_configs = {
            'sell_intensity_r2': True,
            'sell_intensity_mse': False,
            'volatility_r2': True,
            'volatility_mse': False,
            'regime_accuracy': True,
            'uncertainty_calibration': True,
            'inference_time': False
        }
        
        for metric, higher_is_better in metric_configs.items():
            model_scores = []
            
            for model_name, results in self.comparison_results.items():
                if 'error' not in results and metric in results:
                    model_scores.append((model_name, results[metric]))
            
            if model_scores:
                # Sort based on whether higher is better
                model_scores.sort(key=lambda x: x[1], reverse=higher_is_better)
                rankings[metric] = [name for name, _ in model_scores]
        
        return rankings
    
    def _find_best_models(self) -> Dict[str, str]:
        """최고 성능 모델 찾기"""
        best_models = {}
        
        key_metrics = ['sell_intensity_r2', 'volatility_r2', 'regime_accuracy', 'inference_time']
        
        for metric in key_metrics:
            model_scores = []
            
            for model_name, results in self.comparison_results.items():
                if 'error' not in results and metric in results:
                    model_scores.append((model_name, results[metric]))
            
            if model_scores:
                if metric == 'inference_time':
                    # Lower is better for inference time
                    best_model = min(model_scores, key=lambda x: x[1])
                else:
                    # Higher is better for accuracy metrics
                    best_model = max(model_scores, key=lambda x: x[1])
                
                best_models[metric] = best_model[0]
        
        return best_models
    
    def _generate_recommendations(self) -> Dict[str, str]:
        """모델 추천 생성"""
        recommendations = {}
        
        # Overall best model (weighted score)
        overall_scores = {}
        
        for model_name, results in self.comparison_results.items():
            if 'error' in results:
                continue
            
            score = 0
            weight_sum = 0
            
            # Weighted scoring
            weights = {
                'sell_intensity_r2': 0.3,
                'volatility_r2': 0.2,
                'regime_accuracy': 0.2,
                'uncertainty_calibration': 0.15,
                'inference_time': 0.15
            }
            
            for metric, weight in weights.items():
                if metric in results:
                    value = results[metric]
                    
                    # Normalize and invert for inference time
                    if metric == 'inference_time':
                        # Convert to score (lower time = higher score)
                        max_time = max(r.get(metric, 0) for r in self.comparison_results.values() 
                                     if 'error' not in r and metric in r)
                        if max_time > 0:
                            value = 1 - (value / max_time)
                    
                    score += value * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                overall_scores[model_name] = score / weight_sum
        
        if overall_scores:
            best_overall = max(overall_scores.items(), key=lambda x: x[1])
            recommendations['overall_best'] = best_overall[0]
        
        # Speed recommendation
        speed_scores = {name: results.get('inference_time', float('inf')) 
                       for name, results in self.comparison_results.items() 
                       if 'error' not in results}
        
        if speed_scores:
            fastest_model = min(speed_scores.items(), key=lambda x: x[1])
            recommendations['fastest'] = fastest_model[0]
        
        # Accuracy recommendation
        accuracy_scores = {name: results.get('sell_intensity_r2', 0) 
                          for name, results in self.comparison_results.items() 
                          if 'error' not in results}
        
        if accuracy_scores:
            most_accurate = max(accuracy_scores.items(), key=lambda x: x[1])
            recommendations['most_accurate'] = most_accurate[0]
        
        return recommendations
    
    def print_summary(self):
        """비교 결과 요약 출력"""
        if not self.comparison_results:
            print("❌ No comparison results available.")
            return
        
        print("🔍 Eddie Model Comparison Summary")
        print("=" * 60)
        
        # Overall stats
        total_models = len(self.comparison_results)
        successful_models = sum(1 for r in self.comparison_results.values() if 'error' not in r)
        
        print(f"Total Models: {total_models}")
        print(f"Successfully Evaluated: {successful_models}")
        print()
        
        # Key metrics
        print("📊 Key Performance Metrics:")
        print("-" * 40)
        
        for model_name, results in self.comparison_results.items():
            if 'error' in results:
                print(f"{model_name}: ❌ {results['error']}")
                continue
            
            print(f"{model_name}:")
            if 'sell_intensity_r2' in results:
                print(f"  Signal R²: {results['sell_intensity_r2']:.4f}")
            if 'volatility_r2' in results:
                print(f"  Volatility R²: {results['volatility_r2']:.4f}")
            if 'regime_accuracy' in results:
                print(f"  Regime Accuracy: {results['regime_accuracy']:.4f}")
            if 'inference_time' in results:
                print(f"  Inference Time: {results['inference_time']*1000:.2f}ms")
            print()


# Utility functions for easy model comparison
def compare_eddie_models(
    model_configs: List[Dict[str, Any]],
    test_data: Dict[str, np.ndarray],
    results_dir: str = "./comparison_results"
) -> Dict[str, Any]:
    """
    Eddie 모델들을 쉽게 비교하는 함수
    
    Args:
        model_configs: [{'name': str, 'path': str, 'config': EddieConfig}, ...]
        test_data: {'features': X, 'targets': y}
        results_dir: 결과 저장 디렉토리
        
    Returns:
        comparison_results: 비교 결과
    """
    comparator = ModelComparator(results_dir)
    
    # Add models
    for config in model_configs:
        comparator.add_model(
            name=config['name'],
            model_path=config['path'],
            config=config.get('config'),
            description=config.get('description', '')
        )
    
    # Run comparison
    results = comparator.compare_models(test_data)
    
    # Generate report
    report = comparator.generate_comparison_report()
    
    # Print summary
    comparator.print_summary()
    
    return report 