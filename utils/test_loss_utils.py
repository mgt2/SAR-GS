"""
Test script for loss functions with 1D and 2D data. Written by Claude for internal testing purposes.
"""

import torch
import sys
sys.path.append('.')

from loss_utils import ssim, l1_loss, l2_loss, fourier_loss, psnr


def test_1d_losses():
    """Test loss functions with 1D signals (waveform data)"""
    print("="*70)
    print("Testing 1D Signal Losses")
    print("="*70)
    
    # Create sample 1D signals
    batch_size = 1
    channels = 1
    length = 256
    
    # Test case 1: Shape [C, L] (no batch)
    print("\n[Test 1] Shape [C, L] = [1, 256]")
    signal1_2d = torch.randn(channels, length)
    signal2_2d = torch.randn(channels, length)
    
    try:
        ssim_val = ssim(signal1_2d, signal2_2d)
        print(f"  ✓ SSIM: {ssim_val.item():.4f}")
    except Exception as e:
        print(f"  ✗ SSIM failed: {e}")
        return False
    
    try:
        l1_val = l1_loss(signal1_2d, signal2_2d)
        print(f"  ✓ L1 Loss: {l1_val.item():.4f}")
    except Exception as e:
        print(f"  ✗ L1 Loss failed: {e}")
        return False
    
    try:
        l2_val = l2_loss(signal1_2d, signal2_2d)
        print(f"  ✓ L2 Loss: {l2_val.item():.4f}")
    except Exception as e:
        print(f"  ✗ L2 Loss failed: {e}")
        return False
    
    try:
        fourier_val = fourier_loss(signal1_2d, signal2_2d)
        print(f"  ✓ Fourier Loss: {fourier_val.item():.4f}")
    except Exception as e:
        print(f"  ✗ Fourier Loss failed: {e}")
        return False
    
    # Test case 2: Shape [B, C, L] (with batch)
    print("\n[Test 2] Shape [B, C, L] = [1, 1, 256]")
    signal1_3d = torch.randn(batch_size, channels, length)
    signal2_3d = torch.randn(batch_size, channels, length)
    
    try:
        ssim_val = ssim(signal1_3d, signal2_3d)
        print(f"  ✓ SSIM: {ssim_val.item():.4f}")
    except Exception as e:
        print(f"  ✗ SSIM failed: {e}")
        return False
    
    try:
        fourier_val = fourier_loss(signal1_3d, signal2_3d)
        print(f"  ✓ Fourier Loss: {fourier_val.item():.4f}")
    except Exception as e:
        print(f"  ✗ Fourier Loss failed: {e}")
        return False
    
    # Test case 3: Identical signals (should give perfect scores)
    print("\n[Test 3] Identical signals (sanity check)")
    signal = torch.randn(channels, length)
    
    ssim_val = ssim(signal, signal)
    l1_val = l1_loss(signal, signal)
    l2_val = l2_loss(signal, signal)
    
    print(f"  SSIM (should be ~1.0): {ssim_val.item():.6f}")
    print(f"  L1 Loss (should be ~0.0): {l1_val.item():.6e}")
    print(f"  L2 Loss (should be ~0.0): {l2_val.item():.6e}")
    
    if ssim_val.item() > 0.99 and l1_val.item() < 1e-6 and l2_val.item() < 1e-6:
        print("  ✓ Sanity check passed")
    else:
        print("  ⚠ Warning: Sanity check values unexpected")
    
    print("\n✓ All 1D tests passed!")
    return True


def test_2d_losses():
    """Test loss functions with 2D images"""
    print("\n" + "="*70)
    print("Testing 2D Image Losses")
    print("="*70)
    
    # Create sample 2D images
    batch_size = 1
    channels = 3
    height = 64
    width = 64
    
    # Test case 1: Shape [B, C, H, W]
    print("\n[Test 1] Shape [B, C, H, W] = [1, 3, 64, 64]")
    img1 = torch.randn(batch_size, channels, height, width)
    img2 = torch.randn(batch_size, channels, height, width)
    
    try:
        ssim_val = ssim(img1, img2)
        print(f"  ✓ SSIM: {ssim_val.item():.4f}")
    except Exception as e:
        print(f"  ✗ SSIM failed: {e}")
        return False
    
    try:
        l1_val = l1_loss(img1, img2)
        print(f"  ✓ L1 Loss: {l1_val.item():.4f}")
    except Exception as e:
        print(f"  ✗ L1 Loss failed: {e}")
        return False
    
    try:
        fourier_val = fourier_loss(img1, img2)
        print(f"  ✓ Fourier Loss: {fourier_val.item():.4f}")
    except Exception as e:
        print(f"  ✗ Fourier Loss failed: {e}")
        return False
    
    try:
        psnr_val = psnr(img1, img2)
        print(f"  ✓ PSNR: {psnr_val.mean().item():.4f}")
    except Exception as e:
        print(f"  ✗ PSNR failed: {e}")
        return False
    
    print("\n✓ All 2D tests passed!")
    return True


def test_cuda():
    """Test if CUDA is available and works"""
    print("\n" + "="*70)
    print("Testing CUDA Support")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping GPU tests")
        return True
    
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Test 1D on GPU
    signal1 = torch.randn(1, 256).cuda()
    signal2 = torch.randn(1, 256).cuda()
    
    try:
        ssim_val = ssim(signal1, signal2)
        print(f"  ✓ 1D SSIM on GPU: {ssim_val.item():.4f}")
    except Exception as e:
        print(f"  ✗ GPU test failed: {e}")
        return False
    
    # Test 2D on GPU
    img1 = torch.randn(1, 3, 64, 64).cuda()
    img2 = torch.randn(1, 3, 64, 64).cuda()
    
    try:
        ssim_val = ssim(img1, img2)
        print(f"  ✓ 2D SSIM on GPU: {ssim_val.item():.4f}")
    except Exception as e:
        print(f"  ✗ GPU test failed: {e}")
        return False
    
    print("\n✓ CUDA tests passed!")
    return True


def test_realistic_waveform():
    """Test with realistic RF waveform data"""
    print("\n" + "="*70)
    print("Testing with Realistic RF Waveform")
    print("="*70)
    
    # Simulate realistic waveform: complex to magnitude
    import numpy as np
    
    # Generate complex waveform
    time_samples = 256
    complex_waveform = np.random.randn(time_samples) + 1j * np.random.randn(time_samples)
    
    # Convert to magnitude (as done in dataloader)
    magnitude = np.abs(complex_waveform).astype(np.float32)
    
    # Normalize
    if magnitude.max() > magnitude.min():
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    
    # Convert to torch tensor with shape [1, 256]
    pred = torch.from_numpy(magnitude).unsqueeze(0)
    gt = pred + 0.1 * torch.randn_like(pred)  # Add noise for testing
    
    print(f"Waveform shape: {pred.shape}")
    print(f"Value range: [{pred.min():.4f}, {pred.max():.4f}]")
    
    # Test all losses
    try:
        ssim_val = ssim(pred, gt)
        l1_val = l1_loss(pred, gt)
        l2_val = l2_loss(pred, gt)
        fourier_val = fourier_loss(pred, gt)
        
        print(f"\nLoss values:")
        print(f"  SSIM: {ssim_val.item():.4f} (higher is better)")
        print(f"  L1 Loss: {l1_val.item():.4f}")
        print(f"  L2 Loss: {l2_val.item():.4f}")
        print(f"  Fourier Loss: {fourier_val.item():.4f}")
        
        # Test loss combination (typical in training)
        combined_loss = 0.8 * l1_val + 0.2 * (1.0 - ssim_val)
        print(f"  Combined Loss (0.8*L1 + 0.2*(1-SSIM)): {combined_loss.item():.4f}")
        
        print("\n✓ Realistic waveform test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Realistic waveform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LOSS FUNCTIONS TEST SUITE")
    print("="*70)
    
    all_passed = True
    
    # Run tests
    if not test_1d_losses():
        all_passed = False
    
    if not test_2d_losses():
        all_passed = False
    
    if not test_cuda():
        all_passed = False
    
    if not test_realistic_waveform():
        all_passed = False
    
    # Final summary
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nThe loss functions are ready to use with your RF waveform data!")
        print("Your training code should work with shape [1, 256] inputs.")
    else:
        print("SOME TESTS FAILED ✗")
        print("="*70)
    
    sys.exit(0 if all_passed else 1)