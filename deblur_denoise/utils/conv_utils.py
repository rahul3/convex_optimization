import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
from pathlib import Path
from datetime import datetime

from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
from torch.nn import functional as F

from deblur_denoise.utils.logging_utils import logger

# Increase PIL limit if working with large TIFF files
Image.MAX_IMAGE_PIXELS = None  # Remove decompression bomb protection

def read_image(
    path: str,
    device: str | torch.device | None = None,
    shape: tuple[int, int] | None = (100, 100),
    grayscale: bool = True
) -> torch.Tensor:
    """
    Read an image from a file path and convert it to grayscale in PyTorch tensor format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    file_extension = Path(path).suffix
    file_extension = file_extension.lower() if file_extension else file_extension

    if file_extension == '.tiff':
        image = Image.open(path)
        img_tensor = ToTensor()(image)
    else:
        img_tensor = torchvision.io.read_image(path)

    # Read image as tensor (returns a tensor in [C, H, W] format)
    img_tensor = img_tensor.float() / 255  # normalize to [0, 1]
    
    # Convert to grayscale and transform to tensor
    if grayscale:
        img_gray = TF.rgb_to_grayscale(img_tensor)
    else:
        img_gray = img_tensor

    if shape is not None:
        img_gray = F.interpolate(img_gray.unsqueeze(0), size=shape, mode='bilinear', align_corners=False).squeeze(0)
    
    # Move to specified device if provided
    if device is not None:
        img_gray = img_gray.to(device)
    
    return img_gray


# Function to display tensor images
def display_images(original, processed, title1="Original", title2="After Convolution", save_path=None, img_id=None):
    plt.figure(figsize=(12, 6))
    
    # Display original image
    plt.subplot(1, 2, 1)
    
    # Convert tensor to numpy for plotting
    if torch.is_tensor(original):
        if original.is_complex():
            # For complex tensors, use magnitude
            img_np = torch.abs(original).cpu().detach().numpy()
        else:
            img_np = original.cpu().detach().numpy()
        
        # Handle different dimensions
        if img_np.ndim == 3:
            if img_np.shape[0] == 1:  # [1, H, W]
                img_np = img_np[0]  # Extract the 2D array
            elif img_np.shape[0] == 3:  # [3, H, W] (RGB)
                img_np = np.transpose(img_np, (1, 2, 0))  # Convert to [H, W, 3]
    else:
        img_np = original
        
    plt.imshow(img_np, cmap='gray')
    plt.title(title1)
    plt.axis('off')
    
    # Display processed image
    plt.subplot(1, 2, 2)
    
    # Convert tensor to numpy for plotting
    if torch.is_tensor(processed):
        if processed.is_complex():
            # For complex tensors, use magnitude
            proc_np = torch.abs(processed).cpu().detach().numpy()
        else:
            proc_np = processed.cpu().detach().numpy()
        
        # Handle different dimensions
        if proc_np.ndim == 3:
            if proc_np.shape[0] == 1:  # [1, H, W]
                proc_np = proc_np[0]  # Extract the 2D array
            elif proc_np.shape[0] == 3:  # [3, H, W] (RGB)
                proc_np = np.transpose(proc_np, (1, 2, 0))  # Convert to [H, W, 3]
        elif proc_np.ndim == 4:  # [1, 1, H, W]
            proc_np = proc_np[0, 0]  # Extract the 2D array
    else:
        proc_np = processed
        
    # Normalize for better visualization, especially after convolution
    plt.imshow(proc_np, cmap='gray')
    plt.title(title2)
    plt.axis('off')
    
    plt.tight_layout(pad=2.0)  # Increase padding to avoid cutting off titles
    if save_path is not None:
        if img_id is None:
            img_id = f"display_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save combined image
        plt.savefig(os.path.join(save_path, f"{img_id}.png"), bbox_inches='tight')
        logger.info(f"Saved combined image to {os.path.join(save_path, f'{img_id}.png')}")
        
        # Save original image separately
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np, cmap='gray')
        plt.title(title1)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{img_id}_original.png"), bbox_inches='tight')
        
        # Save processed image separately
        plt.figure(figsize=(6, 6))
        plt.imshow(proc_np, cmap='gray')
        plt.title(title2)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{img_id}_processed.png"), bbox_inches='tight')
        
        logger.info(f"Saved individual images to {os.path.join(save_path, f'{img_id}_original.png')} and {os.path.join(save_path, f'{img_id}_processed.png')}")

    plt.show()

def display_complex_output(output):
    """
    Display the real and imaginary parts of a complex tensor.

    Look at the bottom of the code for an example of how to use this function.
    """
    plt.figure(figsize=(12, 6))
    
    # Real part
    plt.subplot(1, 2, 1)
    real_np = output.real.cpu().detach().numpy()
    
    # Handle dimensions
    if real_np.ndim == 3 and real_np.shape[0] == 1:
        real_np = real_np[0]  # Extract the 2D array
    elif real_np.ndim == 4:
        real_np = real_np[0, 0]  # Extract the 2D array
        
    plt.imshow(real_np, cmap='gray')
    plt.title("Real Part")
    plt.axis('off')
    
    # Imaginary part
    plt.subplot(1, 2, 2)
    imag_np = output.imag.cpu().detach().numpy()
    
    # Handle dimensions
    if imag_np.ndim == 3 and imag_np.shape[0] == 1:
        imag_np = imag_np[0]  # Extract the 2D array
    elif imag_np.ndim == 4:
        imag_np = imag_np[0, 0]  # Extract the 2D array
        
    plt.imshow(imag_np, cmap='gray')
    plt.title("Imaginary Part")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
