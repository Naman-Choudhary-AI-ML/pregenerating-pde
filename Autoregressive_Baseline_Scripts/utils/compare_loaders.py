#!/usr/bin/env python3
"""
Simple script to demonstrate the difference between original and local normalization data loaders.
This is just for testing/comparison purposes.
"""

import os
import sys
import numpy as np
import torch

# Add the utils directory to path so we can import both loaders
utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, utils_dir)

# Import both loaders
import data_loader as original_loader
import data_loader_local as local_loader

def create_dummy_config():
    """Create a minimal config for testing."""
    return {
        "data": {"batch_size": 2},
        "model": {"output_dim": 3}
    }

def create_dummy_data(filepath, n_sims=5, timesteps=10, height=32, width=32):
    """Create dummy simulation data for testing."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Create realistic-ish fluid dynamics data
    data = np.zeros((n_sims, timesteps, height, width, 6), dtype=np.float32)
    
    for sim in range(n_sims):
        # Create a circular obstacle in the center
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 6
        
        obstacle_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        for t in range(timesteps):
            # Simulate some flow field (very simplified)
            # Ux (horizontal velocity)
            data[sim, t, :, :, 0] = np.random.normal(1.0 + sim * 0.5, 0.5, (height, width))
            # Uy (vertical velocity)  
            data[sim, t, :, :, 1] = np.random.normal(0.1, 0.3, (height, width))
            # P (pressure)
            data[sim, t, :, :, 2] = np.random.normal(10.0 + sim * 2.0, 2.0, (height, width))
            # Re (Reynolds number)
            data[sim, t, :, :, 3] = 1000.0 + sim * 100.0
            # Mask (1 = obstacle)
            data[sim, t, :, :, 4] = obstacle_mask.astype(np.float32)
            # SDF (signed distance field)
            data[sim, t, :, :, 5] = np.random.normal(0.0, 1.0, (height, width))
    
    np.save(filepath, data)
    print(f"Created dummy data: {filepath} with shape {data.shape}")
    return data

def compare_normalization_effects():
    """Compare the statistical properties of data from both loaders."""
    
    # Create dummy data
    easy_path = "/tmp/test_easy.npy"
    hard_path = "/tmp/test_hard.npy"
    
    dummy_easy = create_dummy_data(easy_path, n_sims=3)
    dummy_hard = create_dummy_data(hard_path, n_sims=3)
    
    config = create_dummy_config()
    
    print("\n" + "="*80)
    print("COMPARING ORIGINAL vs LOCAL NORMALIZATION DATA LOADERS")
    print("="*80)
    
    try:
        # Original loader
        print("\n[ORIGINAL LOADER]")
        orig_loaders = original_loader.get_data_loaders(
            config=config, easy_train=2, hard_train=2,
            easy_path=easy_path, hard_path=hard_path
        )
        orig_train = orig_loaders[0]
        
        # Local normalization loader
        print("\n[LOCAL NORMALIZATION LOADER]")
        local_loaders = local_loader.get_data_loaders(
            config=config, easy_train=2, hard_train=2,
            easy_path=easy_path, hard_path=hard_path,
            apply_local_norm=True
        )
        local_train = local_loaders[0]
        
        # Compare first batch from each
        print("\n" + "-"*60)
        print("COMPARING FIRST BATCH STATISTICS")
        print("-"*60)
        
        orig_batch = next(iter(orig_train))
        local_batch = next(iter(local_train))
        
        # Analyze pixel_values (seed data)
        orig_x = orig_batch["pixel_values"]  # shape: (B, H, W, 6)
        local_x = local_batch["pixel_values"]
        
        print(f"\nBatch shape: {orig_x.shape}")
        print(f"Channels: [Ux, Uy, P, Re, SDF, ValidMask]")
        
        for ch, name in enumerate(["Ux", "Uy", "P", "Re", "SDF", "ValidMask"]):
            orig_ch = orig_x[:, :, :, ch]
            local_ch = local_x[:, :, :, ch]
            
            # Only compute stats for valid (fluid) regions for physical channels
            if ch < 3:  # Physical channels
                vmask = orig_x[:, :, :, 5] > 0.5  # ValidMask
                orig_fluid = orig_ch[vmask]
                local_fluid = local_ch[vmask]
            else:
                orig_fluid = orig_ch.flatten()
                local_fluid = local_ch.flatten()
            
            print(f"\n{name:>10s}:")
            print(f"  Original  - mean: {orig_fluid.mean():.6f}, std: {orig_fluid.std():.6f}, range: [{orig_fluid.min():.3f}, {orig_fluid.max():.3f}]")
            print(f"  Local Norm- mean: {local_fluid.mean():.6f}, std: {local_fluid.std():.6f}, range: [{local_fluid.min():.3f}, {local_fluid.max():.3f}]")
        
        print("\n" + "-"*60)
        print("KEY OBSERVATIONS:")
        print("-"*60)
        print("1. Physical channels (Ux, Uy, P) in local normalized version should have")
        print("   means closer to 0 and more standardized ranges across samples.")
        print("2. Re, SDF, and ValidMask should be identical between both loaders.")
        print("3. Local normalization preserves the valid mask structure.")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        for path in [easy_path, hard_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Cleaned up: {path}")

if __name__ == "__main__":
    compare_normalization_effects()


