"""
Quick test script to verify all modules work correctly
"""
import sys
import os

# Add src to path
sys.path.append('src')

print("="*60)
print("TESTING mmWave Radar AI Project")
print("="*60)

# Test 1: Import modules
print("\n[1/5] Testing module imports...")
try:
    from signal_processing import RadarSimulator, normalize_heatmap
    from data_generator import SyntheticDatasetGenerator
    from models import create_cnn_model, SVMClassifier
    from visualization import plot_range_profile
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize radar
print("\n[2/5] Testing RadarSimulator...")
try:
    radar = RadarSimulator()
    print(f"✓ RadarSimulator initialized")
    print(f"  - Range resolution: {radar.range_resolution:.3f} m")
    print(f"  - Max range: {radar.max_range:.2f} m")
    print(f"  - Max velocity: {radar.max_velocity:.2f} m/s")
except Exception as e:
    print(f"✗ RadarSimulator failed: {e}")
    sys.exit(1)

# Test 3: Generate signal
print("\n[3/5] Testing signal generation...")
try:
    signal = radar.generate_metal_object(range_m=2.0, velocity=1.0, rcs=1.0)
    print(f"✓ Signal generated: shape {signal.shape}")
    
    rd_map = radar.generate_range_doppler_map(signal)
    print(f"✓ Range-Doppler map created: shape {rd_map.shape}")
except Exception as e:
    print(f"✗ Signal generation failed: {e}")
    sys.exit(1)

# Test 4: Dataset generator
print("\n[4/5] Testing dataset generator...")
try:
    dataset_gen = SyntheticDatasetGenerator(radar, output_size=(64, 64))
    print("✓ DatasetGenerator initialized")
    
    # Generate a few samples
    heatmap1, label1 = dataset_gen.generate_metal_sample()
    heatmap2, label2 = dataset_gen.generate_nonmetal_sample()
    print(f"✓ Sample generation works")
    print(f"  - Metal sample: shape {heatmap1.shape}, label {label1}")
    print(f"  - Non-metal sample: shape {heatmap2.shape}, label {label2}")
except Exception as e:
    print(f"✗ Dataset generation failed: {e}")
    sys.exit(1)

# Test 5: Model creation
print("\n[5/5] Testing model creation...")
try:
    import tensorflow as tf
    print(f"  TensorFlow version: {tf.__version__}")
    
    # Suppress TF warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    model = create_cnn_model(input_shape=(64, 64), num_classes=2)
    print(f"✓ CNN model created")
    print(f"  - Total parameters: {model.count_params():,}")
    print(f"  - Input shape: (64, 64, 1)")
    print(f"  - Output shape: (2,)")
    
    # Quick inference test
    import numpy as np
    test_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
    output = model.predict(test_input, verbose=0)
    print(f"✓ Model inference works: output shape {output.shape}")
    
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\n✓ Project is ready to use")
print("✓ All core components working")
print("✓ Ready to run notebooks")
print("\nNext steps:")
print("  1. jupyter notebook notebooks/01_radar_simulation.ipynb")
print("  2. jupyter notebook notebooks/02_classification_model.ipynb")
print("  3. jupyter notebook notebooks/03_hidden_object_detection.ipynb")
print("="*60)
