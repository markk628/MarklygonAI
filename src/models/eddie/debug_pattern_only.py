"""
Debug Pattern Analyzer specific error
"""
import torch
import traceback
from pattern_pipeline.pattern_analyzer import PatternAnalyzer

def debug_pattern_step_by_step():
    """Step by step debugging"""
    print("🔍 Debugging Pattern Analyzer step by step...")
    
    # Exact same parameters as test_model.py
    batch_size, seq_len, num_features = 2, 30, 20
    x = torch.randn(batch_size, seq_len, num_features)
    
    print(f"Input shape: {x.shape}")
    
    # Same configuration as test_model.py
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
            'feature_dim': 32,  # This is the key difference!
            'use_denoising': True
        },
        feature_dim=32,  # This too!
        integration_method='attention',
        use_market_regime=True
    )
    
    try:
        print("Starting forward pass...")
        outputs = model(x)
        print("✅ SUCCESS!")
        
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Let's check what the configuration differences are
        print(f"\nDebugging info:")
        print(f"feature_dim in PatternAnalyzer: {model.feature_dim}")
        print(f"timesnet d_model: {model.timesnet.d_model}")

if __name__ == "__main__":
    debug_pattern_step_by_step() 