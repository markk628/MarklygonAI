"""
Debug test to isolate dimension issues
"""
import torch
import traceback
from pattern_pipeline.pattern_analyzer import PatternAnalyzer

def debug_pattern_analyzer():
    """Debug PatternAnalyzer to find the exact error"""
    print("Debugging PatternAnalyzer...")
    
    # Test parameters matching the failing test
    batch_size, seq_len, num_features = 2, 30, 20
    
    # Create test data
    x = torch.randn(batch_size, seq_len, num_features)
    print(f"Input shape: {x.shape}")
    
    # Create model
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
    
    try:
        print("\n--- Step-by-step debugging ---")
        
        # 1. TimesNet Analysis
        print("1. Running TimesNet...")
        timesnet_features = model.timesnet(x)
        print(f"TimesNet output shape: {timesnet_features.shape}")
        
        # 2. Wavelet Analysis
        print("2. Running Wavelet Analysis...")
        wavelet_features = []
        for i, wavelet_extractor in enumerate(model.wavelet_extractors):
            if i < num_features:
                print(f"Processing wavelet for feature {i}...")
                indicator_series = x[:, :, i]  # (batch_size, seq_len)
                print(f"Indicator series shape: {indicator_series.shape}")
                wavelet_feat = wavelet_extractor(indicator_series)  # (batch_size, feature_dim)
                print(f"Wavelet feature shape: {wavelet_feat.shape}")
                wavelet_features.append(wavelet_feat)
                
                if i >= 2:  # Only test first 3 for debugging
                    break
        
        # Average wavelet features
        if wavelet_features:
            avg_wavelet_features = torch.stack(wavelet_features, dim=1).mean(dim=1)
            print(f"Average wavelet features shape: {avg_wavelet_features.shape}")
        else:
            avg_wavelet_features = torch.zeros(batch_size, model.feature_dim, device=x.device)
        
        # 3. Scale-specific Analysis
        print("3. Running Scale-specific Analysis...")
        timesnet_transposed = timesnet_features.transpose(1, 2)
        print(f"TimesNet transposed shape: {timesnet_transposed.shape}")
        
        print("Running short analyzer...")
        short_features = model.short_analyzer(timesnet_transposed)
        print(f"Short features shape: {short_features.shape}")
        
        print("Running medium analyzer...")
        medium_features = model.medium_analyzer(timesnet_transposed)
        print(f"Medium features shape: {medium_features.shape}")
        
        print("Running long analyzer...")
        long_features = model.long_analyzer(timesnet_transposed)
        print(f"Long features shape: {long_features.shape}")
        
        # 4. Combine scale-specific features
        print("4. Combining scale features...")
        scale_features = torch.cat([short_features, medium_features, long_features], dim=1)
        print(f"Combined scale features shape: {scale_features.shape}")
        scale_features = scale_features.transpose(1, 2)
        print(f"Scale features transposed shape: {scale_features.shape}")
        
        # 5. Feature Integration
        print("5. Feature Integration...")
        if model.integration_method == 'attention':
            print("Using attention integration method")
            print(f"Temporal fusion input shapes:")
            print(f"  - scale_features: {scale_features.shape}")
            print(f"  - avg_wavelet_features: {avg_wavelet_features.shape}")
            
            # Debug temporal fusion step by step
            print("5a. Running temporal fusion...")
            integrated_features = model.temporal_fusion(
                scale_features, avg_wavelet_features
            )
            print(f"Integrated features shape: {integrated_features.shape}")
        
        # 6. Global feature aggregation
        print("6. Global feature aggregation...")
        global_features = torch.mean(integrated_features, dim=1)
        print(f"Global features shape: {global_features.shape}")
        
        print("✅ All steps completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def debug_temporal_fusion():
    """Debug TemporalPatternFusion separately"""
    print("\n\nDebugging TemporalPatternFusion separately...")
    
    from pattern_pipeline.pattern_analyzer import TemporalPatternFusion
    
    # Test parameters
    batch_size, seq_len = 2, 30
    timesnet_dim = 32
    wavelet_dim = 32
    output_dim = 32
    
    # Create test data
    timesnet_features = torch.randn(batch_size, seq_len, timesnet_dim)
    wavelet_features = torch.randn(batch_size, wavelet_dim)
    
    print(f"TimesNet features shape: {timesnet_features.shape}")
    print(f"Wavelet features shape: {wavelet_features.shape}")
    
    # Create temporal fusion
    fusion = TemporalPatternFusion(
        timesnet_dim=timesnet_dim,
        wavelet_dim=wavelet_dim,
        output_dim=output_dim,
        num_heads=8
    )
    
    try:
        output = fusion(timesnet_features, wavelet_features)
        print(f"TemporalPatternFusion output shape: {output.shape}")
        print("✅ TemporalPatternFusion works!")
        return True
    except Exception as e:
        print(f"ERROR in TemporalPatternFusion: {e}")
        traceback.print_exc()
        return False

def debug_simple_timesnet():
    """Debug simple TimesNet to isolate the issue"""
    print("Debugging simple TimesNet...")
    
    from pattern_pipeline.timesnet import TimesNet
    
    # Test parameters
    batch_size, seq_len = 2, 30
    
    # Create test data
    x = torch.randn(batch_size, seq_len, 1)
    print(f"Input shape: {x.shape}")
    
    # Create simple TimesNet
    timesnet = TimesNet(
        seq_len=seq_len,
        pred_len=1,
        top_k=5,
        d_model=32,
        d_ff=64,
        num_kernels=6,
        num_layers=2,
        dropout=0.1
    )
    
    try:
        output = timesnet(x)
        print(f"TimesNet output shape: {output.shape}")
        print("✅ Simple TimesNet works!")
        return True
    except Exception as e:
        print(f"ERROR in simple TimesNet: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_simple_timesnet()
    debug_temporal_fusion()
    debug_pattern_analyzer() 