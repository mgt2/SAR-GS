# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #

# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# from math import exp
# import torch.fft


# def l1_loss(network_output, gt):
#     return torch.abs((network_output - gt)).mean()


# def l2_loss(network_output, gt):
#     return ((network_output - gt) ** 2).mean()


# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()


# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window


# def ssim(img1, img2, window_size=11, size_average=True):
#     channel = img1.size(-3)
#     window = create_window(window_size, channel)

#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)

#     return _ssim(img1, img2, window, window_size, channel, size_average)


# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)


# def mse(img1, img2):
#     return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


# def psnr(img1, img2):
#     mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))


# # def fourier_loss(pred, gt):
# #     """
# #     Compute Fourier-domain L2 loss between pred and gt.
# #     pred, gt: Tensors of shape [H, W] or [B, H, W]
# #     """
# #     # Add batch dim if missing
# #     if pred.dim() == 2:
# #         pred = pred.unsqueeze(0)
# #         gt = gt.unsqueeze(0)

# #     # Apply 2D FFT
# #     pred_fft = torch.fft.fft2(pred)
# #     gt_fft = torch.fft.fft2(gt)

# #     # Compute L2 difference in frequency domain
# #     diff = pred_fft - gt_fft
# #     loss = torch.mean(torch.abs(diff) ** 2)

# #     return loss

# def fourier_loss(pred, gt):
#     pred_fft = torch.fft.fft2(pred)
#     gt_fft = torch.fft.fft2(gt)

#     diff = torch.abs(pred_fft - gt_fft) ** 2
#     total_energy = torch.abs(gt_fft) ** 2

#     return torch.mean(diff / (total_energy + 1e-8))

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.fft


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_1d(window_size, channel):
    """Create 1D window for 1D SSIM"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(0).unsqueeze(0)
    window = Variable(_1D_window.expand(channel, 1, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    """
    SSIM that automatically handles both 1D and 2D inputs
    
    For 1D inputs: shape [B, C, L] where L is length
    For 2D inputs: shape [B, C, H, W]
    """
    # Determine dimensionality
    if img1.dim() == 2:
        # Shape [C, L] - 1D signal, add batch dimension
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    if img1.dim() == 3:
        # Shape [B, C, L] - 1D signal
        return ssim_1d(img1, img2, window_size, size_average)
    elif img1.dim() == 4:
        # Shape [B, C, H, W] - 2D image
        return ssim_2d(img1, img2, window_size, size_average)
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D input, got {img1.dim()}D")


def ssim_1d(signal1, signal2, window_size=11, size_average=True):
    """
    SSIM for 1D signals (e.g., time series, waveforms)
    signal1, signal2: shape [B, C, L] where L is signal length
    """
    channel = signal1.size(1)
    window = create_window_1d(window_size, channel)

    if signal1.is_cuda:
        window = window.cuda(signal1.get_device())
    window = window.type_as(signal1)

    return _ssim_1d(signal1, signal2, window, window_size, channel, size_average)


def ssim_2d(img1, img2, window_size=11, size_average=True):
    """
    SSIM for 2D images
    img1, img2: shape [B, C, H, W]
    """
    channel = img1.size(1)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_2d(img1, img2, window, window_size, channel, size_average)


def _ssim_1d(signal1, signal2, window, window_size, channel, size_average=True):
    """1D SSIM implementation using conv1d"""
    mu1 = F.conv1d(signal1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv1d(signal2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv1d(signal1 * signal1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv1d(signal2 * signal2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv1d(signal1 * signal2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1)


def _ssim_2d(img1, img2, window, window_size, channel, size_average=True):
    """2D SSIM implementation using conv2d"""
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def fourier_loss(pred, gt):
    """
    Compute Fourier-domain loss
    Works with both 1D and 2D inputs
    
    For 1D: shape [B, C, L] -> uses fft
    For 2D: shape [B, C, H, W] -> uses fft2
    """
    if pred.dim() == 2:
        # Add batch dimension if needed
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    
    if pred.dim() == 3:
        # 1D signal: [B, C, L]
        pred_fft = torch.fft.fft(pred, dim=-1)
        gt_fft = torch.fft.fft(gt, dim=-1)
    elif pred.dim() == 4:
        # 2D image: [B, C, H, W]
        pred_fft = torch.fft.fft2(pred)
        gt_fft = torch.fft.fft2(gt)
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D input, got {pred.dim()}D")

    diff = torch.abs(pred_fft - gt_fft) ** 2
    total_energy = torch.abs(gt_fft) ** 2

    return torch.mean(diff / (total_energy + 1e-8))
