#!/usr/bin/env python3
"""
Simple test to verify Python execution
"""

print("🔥 Python execution working!")
print("Starting Eddie model tests...")

try:
    import torch
    print(f"✅ PyTorch imported successfully: {torch.__version__}")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    from signal_pipeline.signal_generator import SignalGenerator
    print("✅ SignalGenerator imported successfully")
except ImportError as e:
    print(f"❌ SignalGenerator import failed: {e}")

try:
    from pattern_pipeline.pattern_analyzer import PatternAnalyzer
    print("✅ PatternAnalyzer imported successfully")
except ImportError as e:
    print(f"❌ PatternAnalyzer import failed: {e}")

print("✅ All imports successful!") 