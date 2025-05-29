"""
Wavelet Transform for Time Series Analysis
다중 해상도 시계열 분석을 위한 웨이블릿 변환
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import math
import scipy.signal
from scipy import signal


class WaveletTransform:
    """
    Continuous and Discrete Wavelet Transform implementation
    금융 시계열 분석을 위한 웨이블릿 변환
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        levels: int = 5,
        mode: str = 'symmetric',
        noise_threshold: float = 0.1
    ):
        self.wavelet = wavelet
        self.levels = levels
        self.mode = mode
        self.noise_threshold = noise_threshold
        
        # 웨이블릿 필터 계수 생성
        self.wavelet_filters = self._generate_wavelet_filters()
        
    def _generate_wavelet_filters(self) -> Dict[str, np.ndarray]:
        """웨이블릿 필터 계수 생성"""
        try:
            import pywt
            wavelet_obj = pywt.Wavelet(self.wavelet)
            
            return {
                'dec_lo': wavelet_obj.dec_lo,  # Low-pass decomposition
                'dec_hi': wavelet_obj.dec_hi,  # High-pass decomposition
                'rec_lo': wavelet_obj.rec_lo,  # Low-pass reconstruction
                'rec_hi': wavelet_obj.rec_hi   # High-pass reconstruction
            }
        except ImportError:
            # Fallback: Haar wavelet coefficients
            return {
                'dec_lo': np.array([0.7071067811865476, 0.7071067811865476]),
                'dec_hi': np.array([-0.7071067811865476, 0.7071067811865476]),
                'rec_lo': np.array([0.7071067811865476, 0.7071067811865476]),
                'rec_hi': np.array([0.7071067811865476, -0.7071067811865476])
            }
    
    def dwt_decompose(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        Discrete Wavelet Transform 분해
        
        Args:
            signal: 1D time series signal
            
        Returns:
            coeffs: [approximation, detail1, detail2, ..., detailN]
        """
        try:
            import pywt
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.levels, mode=self.mode)
            return coeffs
        except ImportError:
            # Simple implementation using convolution
            return self._manual_dwt_decompose(signal)
    
    def _manual_dwt_decompose(self, signal: np.ndarray) -> List[np.ndarray]:
        """Manual DWT implementation using convolution"""
        coeffs = []
        current_signal = signal.copy()
        
        for level in range(self.levels):
            # Low-pass filter (approximation)
            approx = np.convolve(current_signal, self.wavelet_filters['dec_lo'], mode='same')
            approx = approx[::2]  # Downsampling
            
            # High-pass filter (detail)
            detail = np.convolve(current_signal, self.wavelet_filters['dec_hi'], mode='same')
            detail = detail[::2]  # Downsampling
            
            coeffs.append(detail)
            current_signal = approx
        
        coeffs.append(current_signal)  # Final approximation
        return coeffs[::-1]  # Reverse to match pywt format
    
    def dwt_reconstruct(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """
        웨이블릿 계수로부터 신호 재구성
        
        Args:
            coeffs: Wavelet coefficients [approximation, detail1, detail2, ...]
            
        Returns:
            reconstructed_signal: Reconstructed 1D signal
        """
        try:
            import pywt
            return pywt.waverec(coeffs, self.wavelet, mode=self.mode)
        except ImportError:
            return self._manual_dwt_reconstruct(coeffs)
    
    def _manual_dwt_reconstruct(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """Manual DWT reconstruction"""
        approx = coeffs[0]
        
        for i in range(1, len(coeffs)):
            detail = coeffs[i]
            
            # Upsampling
            approx_up = np.zeros(len(approx) * 2)
            approx_up[::2] = approx
            
            detail_up = np.zeros(len(detail) * 2)
            detail_up[::2] = detail
            
            # Convolution with reconstruction filters
            approx_conv = np.convolve(approx_up, self.wavelet_filters['rec_lo'], mode='same')
            detail_conv = np.convolve(detail_up, self.wavelet_filters['rec_hi'], mode='same')
            
            approx = approx_conv + detail_conv
        
        return approx
    
    def denoise(
        self, 
        signal: np.ndarray, 
        threshold_mode: str = 'soft'
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        웨이블릿 기반 노이즈 제거
        
        Args:
            signal: Input signal
            threshold_mode: 'soft' or 'hard' thresholding
            
        Returns:
            denoised_signal: Denoised signal
            denoised_coeffs: Denoised wavelet coefficients
        """
        # 웨이블릿 분해
        coeffs = self.dwt_decompose(signal)
        
        # 임계값 계산 및 적용
        denoised_coeffs = []
        for i, coeff in enumerate(coeffs):
            if i == 0:  # Approximation coefficients (keep as is)
                denoised_coeffs.append(coeff)
            else:  # Detail coefficients (apply thresholding)
                threshold = self.noise_threshold * np.std(coeff)
                
                if threshold_mode == 'soft':
                    denoised_coeff = self._soft_threshold(coeff, threshold)
                else:
                    denoised_coeff = self._hard_threshold(coeff, threshold)
                
                denoised_coeffs.append(denoised_coeff)
        
        # 신호 재구성
        denoised_signal = self.dwt_reconstruct(denoised_coeffs)
        
        return denoised_signal, denoised_coeffs
    
    def _soft_threshold(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Soft thresholding"""
        return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)
    
    def _hard_threshold(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Hard thresholding"""
        return data * (np.abs(data) > threshold)
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """
        웨이블릿 기반 특징 추출
        
        Args:
            signal: Input time series
            
        Returns:
            features: Dictionary of extracted features
        """
        # 웨이블릿 분해
        coeffs = self.dwt_decompose(signal)
        
        # 노이즈 제거
        denoised_signal, denoised_coeffs = self.denoise(signal)
        
        features = {
            'approximation': coeffs[0],
            'details': coeffs[1:],
            'denoised_signal': denoised_signal,
            'denoised_coeffs': denoised_coeffs,
            
            # Energy features
            'energy_per_level': [np.sum(c**2) for c in coeffs],
            'total_energy': np.sum([np.sum(c**2) for c in coeffs]),
            'relative_energy': [np.sum(c**2) / np.sum([np.sum(c**2) for c in coeffs]) for c in coeffs],
            
            # Entropy features
            'entropy_per_level': [self._calculate_entropy(c) for c in coeffs],
            'total_entropy': sum([self._calculate_entropy(c) for c in coeffs]),
            
            # Statistical features
            'mean_per_level': [np.mean(c) for c in coeffs],
            'std_per_level': [np.std(c) for c in coeffs],
            'skewness_per_level': [self._calculate_skewness(c) for c in coeffs],
            'kurtosis_per_level': [self._calculate_kurtosis(c) for c in coeffs],
            
            # Trend extraction
            'trend': coeffs[0],  # Approximation as trend
            'volatility': np.std(coeffs[1]) if len(coeffs) > 1 else 0,  # First detail as volatility
            
            # Multi-scale variance
            'variance_per_level': [np.var(c) for c in coeffs],
            'normalized_variance': [np.var(c) / np.var(signal) for c in coeffs]
        }
        
        return features
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Shannon entropy calculation"""
        if len(data) == 0:
            return 0.0
        
        # Normalize to probability distribution
        data_abs = np.abs(data)
        if np.sum(data_abs) == 0:
            return 0.0
        
        prob = data_abs / np.sum(data_abs)
        prob = prob[prob > 0]  # Remove zeros to avoid log(0)
        
        return -np.sum(prob * np.log2(prob))
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Skewness calculation"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Kurtosis calculation"""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis


class TorchWaveletTransform(nn.Module):
    """
    PyTorch implementation of Wavelet Transform
    GPU 가속을 위한 텐서 기반 웨이블릿 변환
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        levels: int = 5,
        device: torch.device = None
    ):
        super().__init__()
        
        self.wavelet = wavelet
        self.levels = levels
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate wavelet filters and convert to torch tensors
        wt = WaveletTransform(wavelet, levels)
        filters = wt.wavelet_filters
        
        # Register filters as buffers (non-trainable parameters)
        self.register_buffer('dec_lo', torch.tensor(filters['dec_lo'], dtype=torch.float32))
        self.register_buffer('dec_hi', torch.tensor(filters['dec_hi'], dtype=torch.float32))
        self.register_buffer('rec_lo', torch.tensor(filters['rec_lo'], dtype=torch.float32))
        self.register_buffer('rec_hi', torch.tensor(filters['rec_hi'], dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with wavelet decomposition
        
        Args:
            x: (batch_size, seq_len) or (batch_size, seq_len, 1)
            
        Returns:
            wavelet_features: Dictionary of wavelet features
        """
        if x.dim() == 3:
            x = x.squeeze(-1)  # Remove last dimension if present
        
        batch_size, seq_len = x.size()
        
        # Decompose each signal in the batch
        approx_coeffs = []
        detail_coeffs = []
        
        for i in range(batch_size):
            signal = x[i].cpu().numpy()
            
            # Use numpy implementation for now (can be optimized later)
            wt = WaveletTransform(self.wavelet, self.levels)
            coeffs = wt.dwt_decompose(signal)
            
            approx_coeffs.append(torch.tensor(coeffs[0], device=x.device))
            detail_coeffs.append([torch.tensor(c, device=x.device) for c in coeffs[1:]])
        
        # Pad and stack coefficients
        approx_tensor = self._pad_and_stack(approx_coeffs)
        detail_tensors = []
        
        for level in range(self.levels):
            level_coeffs = [detail_coeffs[i][level] for i in range(batch_size)]
            detail_tensors.append(self._pad_and_stack(level_coeffs))
        
        return {
            'approximation': approx_tensor,
            'details': detail_tensors,
            'energy': self._compute_energy_features(approx_tensor, detail_tensors),
            'entropy': self._compute_entropy_features(approx_tensor, detail_tensors)
        }
    
    def _pad_and_stack(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Pad tensors to same length and stack"""
        if not tensors:
            return torch.empty(0, device=self.device)
        
        max_len = max(t.size(0) for t in tensors)
        
        padded_tensors = []
        for t in tensors:
            if t.size(0) < max_len:
                pad_size = max_len - t.size(0)
                t_padded = F.pad(t, (0, pad_size), mode='constant', value=0)
            else:
                t_padded = t[:max_len]
            padded_tensors.append(t_padded)
        
        return torch.stack(padded_tensors, dim=0)
    
    def _compute_energy_features(
        self, 
        approx: torch.Tensor, 
        details: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute energy-based features"""
        features = {}
        
        # Energy per level
        features['approx_energy'] = torch.sum(approx**2, dim=1)
        features['detail_energies'] = []
        
        total_energy = features['approx_energy'].clone()
        
        for detail in details:
            detail_energy = torch.sum(detail**2, dim=1)
            features['detail_energies'].append(detail_energy)
            total_energy += detail_energy
        
        features['total_energy'] = total_energy
        
        # Relative energy
        features['relative_approx_energy'] = features['approx_energy'] / (total_energy + 1e-8)
        features['relative_detail_energies'] = [
            de / (total_energy + 1e-8) for de in features['detail_energies']
        ]
        
        return features
    
    def _compute_entropy_features(
        self, 
        approx: torch.Tensor, 
        details: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute entropy-based features"""
        features = {}
        
        # Approximate entropy using squared coefficients
        features['approx_entropy'] = self._tensor_entropy(approx)
        features['detail_entropies'] = [self._tensor_entropy(detail) for detail in details]
        
        return features
    
    def _tensor_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute entropy for tensor (batch-wise)"""
        batch_size = x.size(0)
        entropies = []
        
        for i in range(batch_size):
            data = torch.abs(x[i])
            data_sum = torch.sum(data)
            
            if data_sum > 0:
                prob = data / data_sum
                prob = prob[prob > 0]  # Remove zeros
                entropy = -torch.sum(prob * torch.log2(prob + 1e-8))
            else:
                entropy = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            
            entropies.append(entropy)
        
        return torch.stack(entropies)


class WaveletFeatureExtractor(nn.Module):
    """
    High-level wavelet feature extractor for neural networks
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        levels: int = 5,
        feature_dim: int = 64,
        use_denoising: bool = True,
        device: torch.device = None
    ):
        super().__init__()
        
        self.wavelet = wavelet
        self.levels = levels
        self.feature_dim = feature_dim
        self.use_denoising = use_denoising
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.torch_wavelet = TorchWaveletTransform(wavelet, levels, device=self.device)
        
        # Feature projection layers
        self.approx_projection = nn.Linear(1, feature_dim // 4)
        self.detail_projections = nn.ModuleList([
            nn.Linear(1, feature_dim // 4) for _ in range(min(levels, 3))
        ])
        
        # Feature fusion
        total_input_dim = feature_dim // 4 * (1 + min(levels, 3))  # approx + top 3 details
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Energy and entropy fusion
        energy_dim = 1 + min(levels, 3)  # approx + top 3 details
        self.energy_fusion = nn.Sequential(
            nn.Linear(energy_dim, feature_dim // 4),
            nn.ReLU()
        )
        
        self.entropy_fusion = nn.Sequential(
            nn.Linear(energy_dim, feature_dim // 4),
            nn.ReLU()
        )
        
        # Final output
        self.output_projection = nn.Sequential(
            nn.Linear(feature_dim + feature_dim // 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) - Time series data
            
        Returns:
            features: (batch_size, feature_dim) - Wavelet features
        """
        # Wavelet decomposition
        wavelet_output = self.torch_wavelet(x)
        
        approx = wavelet_output['approximation']  # (batch_size, approx_len)
        details = wavelet_output['details']  # List of (batch_size, detail_len)
        energy_features = wavelet_output['energy']
        entropy_features = wavelet_output['entropy']
        
        # Project coefficients to feature space
        # Use global average pooling to get fixed-size features
        approx_pooled = torch.mean(approx, dim=1, keepdim=True)  # (batch_size, 1)
        approx_feat = self.approx_projection(approx_pooled)  # (batch_size, feature_dim//4)
        
        detail_feats = []
        for i, detail in enumerate(details[:3]):  # Use only top 3 detail levels
            detail_pooled = torch.mean(detail, dim=1, keepdim=True)  # (batch_size, 1)
            detail_feat = self.detail_projections[i](detail_pooled)  # (batch_size, feature_dim//4)
            detail_feats.append(detail_feat)
        
        # Concatenate coefficient features
        coeff_features = torch.cat([approx_feat] + detail_feats, dim=1)
        coeff_features = self.feature_fusion(coeff_features)  # (batch_size, feature_dim)
        
        # Energy features
        energy_input = torch.stack([
            energy_features['relative_approx_energy']
        ] + energy_features['relative_detail_energies'][:3], dim=1)  # (batch_size, energy_dim)
        
        energy_output = self.energy_fusion(energy_input)  # (batch_size, feature_dim//4)
        
        # Entropy features
        entropy_input = torch.stack([
            entropy_features['approx_entropy']
        ] + entropy_features['detail_entropies'][:3], dim=1)  # (batch_size, entropy_dim)
        
        entropy_output = self.entropy_fusion(entropy_input)  # (batch_size, feature_dim//4)
        
        # Combine all features
        combined_features = torch.cat([
            coeff_features,  # (batch_size, feature_dim)
            energy_output,   # (batch_size, feature_dim//4)
            entropy_output   # (batch_size, feature_dim//4)
        ], dim=1)
        
        # Final projection
        output_features = self.output_projection(combined_features)
        
        return output_features


def test_wavelet_transform():
    """웨이블릿 변환 테스트"""
    print("Testing Wavelet Transform...")
    
    # Generate test signal
    t = np.linspace(0, 1, 512)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(len(t))
    
    # Test numpy implementation
    wt = WaveletTransform(wavelet='db4', levels=5)
    
    # Decomposition
    coeffs = wt.dwt_decompose(signal)
    print(f"Number of coefficient levels: {len(coeffs)}")
    print(f"Coefficient lengths: {[len(c) for c in coeffs]}")
    
    # Reconstruction
    reconstructed = wt.dwt_reconstruct(coeffs)
    print(f"Reconstruction error: {np.mean((signal[:len(reconstructed)] - reconstructed)**2):.6f}")
    
    # Denoising
    denoised, denoised_coeffs = wt.denoise(signal)
    print(f"Denoising SNR improvement: {10 * np.log10(np.var(signal) / np.var(signal - denoised)):.2f} dB")
    
    # Feature extraction
    features = wt.extract_features(signal)
    print(f"Number of feature types: {len(features)}")
    print(f"Energy per level: {features['energy_per_level']}")
    
    # Test PyTorch implementation
    batch_size, seq_len = 4, 512
    x = torch.randn(batch_size, seq_len)
    
    torch_wt = TorchWaveletTransform(wavelet='db4', levels=5)
    torch_output = torch_wt(x)
    
    print(f"PyTorch approximation shape: {torch_output['approximation'].shape}")
    print(f"PyTorch detail shapes: {[d.shape for d in torch_output['details']]}")
    
    # Test feature extractor
    feature_extractor = WaveletFeatureExtractor(
        wavelet='db4',
        levels=5,
        feature_dim=64
    )
    
    features = feature_extractor(x)
    print(f"Extracted features shape: {features.shape}")
    assert features.shape == (batch_size, 64)
    
    print("All wavelet tests passed!")


if __name__ == "__main__":
    test_wavelet_transform() 