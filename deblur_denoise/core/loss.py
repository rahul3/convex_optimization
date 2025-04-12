import torch
import torch.nn.functional as F
from ..utils.logging_utils import logger

def l1_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the L1 loss between two tensors.
    """
    logger.info(f"L1 loss: {torch.norm(x - y, p=1)}")
    return torch.norm(x - y, p=1)

def l2_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2 loss between two tensors.
    """
    logger.info(f"L2 loss: {torch.norm(x - y, p=2)}")
    return torch.norm(x - y, p=2)

def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Compute the PSNR between two tensors.
    
    Args:
        x: Input tensor (predicted image).
        y: Target tensor (ground truth image).
        max_val: Maximum possible value of the signal (default: 1.0 for normalized tensors).
    
    Returns:
        PSNR value as a tensor.
    """
    mse = torch.mean((x - y) ** 2)
    if mse == 0:  
        logger.info(f"PSNR: {float('inf')}")
        return torch.tensor(float('inf'))
    logger.info(f"PSNR: {10 * torch.log10(max_val ** 2 / mse)}")
    return 10 * torch.log10(max_val ** 2 / mse)


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> torch.Tensor:
    """
    Compute the SSIM between two tensors (single- or multi-channel images).
    
    Args:
        x: Input tensor (predicted image, shape [B, C, H, W] or [H, W] or [C, H, W]).
        y: Target tensor (ground truth image, same shape as x).
        window_size: Size of the Gaussian window.
        max_val: Maximum possible value of the signal (e.g., 1.0 for normalized, 255.0 for uint8).
    
    Returns:
        SSIM value as a tensor (shape [B] for per-image SSIM, averaged over channels and spatial dimensions).
    """
    # Validate shapes
    if x.shape != y.shape:
        raise ValueError(f"Input tensors must have the same shape, got {x.shape} and {y.shape}")

    # Ensure inputs are 4D tensors with shape [B, C, H, W]
    if x.dim() < 4:
        if x.dim() == 2:  # [H, W]
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif x.dim() == 3:  # [C, H, W]
            x = x.unsqueeze(0)  # [1, C, H, W]
    
    if y.dim() < 4:
        if y.dim() == 2:  # [H, W]
            y = y.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif y.dim() == 3:  # [C, H, W]
            y = y.unsqueeze(0)  # [1, C, H, W]
    
    # Constants to stabilize division
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Create Gaussian window
    def gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g[:, None] * g[None, :]
    
    window = gaussian_window(window_size, sigma=1.5).to(x.device)
    window = window[None, None, :, :]  # Shape [1, 1, window_size, window_size]
    
    # Number of channels
    num_channels = x.shape[1]
    
    # Compute means (luminance)
    mu_x = F.conv2d(x, window, padding=window_size//2, groups=num_channels)
    mu_y = F.conv2d(y, window, padding=window_size//2, groups=num_channels)
    
    # Compute variances and covariance (contrast and structure)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.conv2d(x ** 2, window, padding=window_size//2, groups=num_channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, window, padding=window_size//2, groups=num_channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=window_size//2, groups=num_channels) - mu_xy
    
    # SSIM formula
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / denominator
    
    # Average over spatial dimensions and channels
    ssim_value = ssim_map.mean(dim=(2, 3))  # Shape [B, C]
    ssim_value = ssim_value.mean(dim=1)  # Shape [B], average over channels
    
    logger.info(f"SSIM: {ssim_value.mean().item()}")
    return ssim_value

def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the MSE between two tensors.
    """
    logger.info(f"MSE: {torch.mean((x - y) ** 2)}")
    return torch.mean((x - y) ** 2)

