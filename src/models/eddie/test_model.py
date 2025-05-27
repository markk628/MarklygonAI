"""
Eddie Model Basic Test
Eddie 모델의 기본 동작 테스트
"""
import torch
import numpy as np
from config import QUICK_CONFIG
from signal_pipeline.signal_generator import SignalGenerator
from pattern_pipeline.pattern_analyzer import PatternAnalyzer

def test_signal_generator():
    """Signal Generator 테스트"""
    print("Testing Signal Generator...")
    
    # Create test data
    batch_size, seq_len, num_features = 2, 30, 20
    x = torch.randn(batch_size, seq_len, num_features)
    
    # Initialize model
    model = SignalGenerator(
        num_inputs=num_features,
        tcn_channels=[32, 64, 32],
        seq_len=seq_len,
        use_uncertainty=True,
        use_adaptive_attention=True,
        use_cross_time=True
    )
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    try:
        outputs = model(x)
        print("✅ Signal Generator forward pass successful!")
        
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
                
        return True
        
    except Exception as e:
        print(f"❌ Signal Generator failed: {e}")
        return False

def test_pattern_analyzer():
    """Pattern Analyzer 테스트"""
    print("\nTesting Pattern Analyzer...")
    
    # Create test data
    batch_size, seq_len, num_features = 2, 30, 20
    x = torch.randn(batch_size, seq_len, num_features)
    
    # Initialize model
    model = PatternAnalyzer(
        num_features=num_features,
        seq_len=seq_len,
        timesnet_config={
            'seq_len': seq_len,
            'pred_len': 1,
            'top_k': 5,
            'd_model': 32,
            'd_ff': 64,
            'num_kernels': 6,
            'num_layers': 2,
            'dropout': 0.1
        },
        wavelet_config={
            'wavelet': 'db4',
            'levels': 3,
            'feature_dim': 32,
            'use_denoising': True
        },
        feature_dim=32,
        integration_method='attention',
        use_market_regime=True
    )
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    try:
        outputs = model(x)
        print("✅ Pattern Analyzer forward pass successful!")
        
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
                
        return True
        
    except Exception as e:
        print(f"❌ Pattern Analyzer failed: {e}")
        return False

def test_combined_model():
    """Combined Model 테스트"""
    print("\nTesting Combined Model...")
    
    # Create test data
    batch_size, seq_len, num_features = 2, 30, 20
    x = torch.randn(batch_size, seq_len, num_features)
    
    # Initialize models
    signal_model = SignalGenerator(
        num_inputs=num_features,
        tcn_channels=[32, 64, 32],
        seq_len=seq_len,
        use_uncertainty=True,
        use_adaptive_attention=True,
        use_cross_time=True
    )
    
    pattern_model = PatternAnalyzer(
        num_features=num_features,
        seq_len=seq_len,
        timesnet_config={
            'seq_len': seq_len,
            'pred_len': 1,
            'top_k': 5,
            'd_model': 32,
            'd_ff': 64,
            'num_kernels': 6,
            'num_layers': 2,
            'dropout': 0.1
        },
        wavelet_config={
            'wavelet': 'db4',
            'levels': 3,
            'feature_dim': 32,
            'use_denoising': True
        },
        feature_dim=32,
        integration_method='attention',
        use_market_regime=True
    )
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    try:
        signal_outputs = signal_model(x)
        pattern_outputs = pattern_model(x)
        
        print("✅ Combined model forward pass successful!")
        
        print("Signal outputs:")
        for key, value in signal_outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        print("Pattern outputs:")
        for key, value in pattern_outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
                
        return True
        
    except Exception as e:
        print(f"❌ Combined model failed: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("Eddie Model Basic Tests")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    tests = [
        test_signal_generator,
        test_pattern_analyzer,
        test_combined_model
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! Eddie models are working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
    
    return all(results)

if __name__ == "__main__":
    main() 