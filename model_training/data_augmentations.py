import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100,  padding='same'):
    """
    Applies a 1D Gaussian smoothing operation with PyTorch to smooth the data along the time axis.
    Args:
        inputs (tensor : B x T x N): A 3D tensor with batch size B, time steps T, and number of features N.
                                     Assumed to already be on the correct device (e.g., GPU).
        kernelSD (float): Standard deviation of the Gaussian smoothing kernel.
        padding (str): Padding mode, either 'same' or 'valid'.
        device (str): Device to use for computation (e.g., 'cuda' or 'cpu').
    Returns:
        smoothed (tensor : B x T x N): A smoothed 3D tensor with batch size B, time steps T, and number of features N.
    """
    # Get Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gaussKernel = gaussian_filter1d(inp, smooth_kernel_std)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    # Convert to tensor
    gaussKernel = torch.tensor(gaussKernel, dtype=torch.float32, device=device)
    gaussKernel = gaussKernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # Prepare convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gaussKernel = gaussKernel.repeat(C, 1, 1)  # [C, 1, kernel_size]

    # Perform convolution
    smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)
    return smoothed.permute(0, 2, 1)  # [B, T, C]

def random_mask(inputs, max_mask_width = 40):
    """
    Applies a random mask along the time dimension of the tensor
    Args:
        inputs (tensor B x T x N): A 3D tensor with batch size B, time steps T, and number of features N.
        device (str): device used for computation (e.g., 'cude' or 'cpu')
    """
    B, T, C = inputs.shape
    max_mask_width = min(max_mask_width, T)
    # width of mask
    mask_width = torch.randint(0, max_mask_width + 1, (1,)).item()
    # starting point of mask
    mask_start = torch.randint(0, T - mask_width + 1, (1,)).item()

    # appl mask
    masked_inputs = inputs.clone()
    masked_inputs[:, mask_start:mask_start+mask_width, :] = 0

    return masked_inputs

def frequency_mask(inputs, max_freq_mask_width = 15, max_time_mask_width = 40):
    masked_inputs = inputs.clone()
    
    # freq mask
    num_freq_bins = inputs.shape[2]
    max_freq_mask_width = min(max_freq_mask_width, num_freq_bins)
    freq_mask_width = torch.randint(0, max_freq_mask_width + 1, (1,)).item()
    freq_mask_start = torch.randint(0, num_freq_bins - max_freq_mask_width + 1, (1,)).item()
    masked_inputs = masked_inputs[:, :, freq_mask_start:freq_mask_width] = 0

    # time mask
    num_time_bins = inputs.shape[1]
    max_time_mask_width = min(max_time_mask_width, inputs.shape[1])
    time_mask_width = torch.randint(0, max_time_mask_width + 1, (1,)).item()
    time_mask_start = torch.randint(0, num_time_bins - max_time_mask_width + 1, (1,)).item()
    masked_inputs = masked_inputs[:, time_mask_start:time_mask_width, :] = 0

    return masked_inputs