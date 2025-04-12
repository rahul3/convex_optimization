import torch
import numpy as np

def create_motion_blur_kernel(size=15, angle=0) -> torch.Tensor:
    """
    Create a motion blur kernel.
    
    Args:
        size: Size of the kernel (odd number)
        angle: Angle of motion in degrees
    """
    # Ensure size is odd
    if size % 2 == 0:
        size += 1
    
    # Create a horizontal line
    kernel = torch.zeros((size, size))
    kernel[size//2, :] = 1
    
    # Rotate the kernel
    if angle != 0:
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Create rotation matrix
        center = size // 2
        rotation_matrix = torch.tensor([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        # Apply rotation
        for i in range(size):
            for j in range(size):
                # Translate to origin
                x = i - center
                y = j - center
                
                # Apply rotation
                new_x = int(center + x * rotation_matrix[0, 0] + y * rotation_matrix[0, 1])
                new_y = int(center + x * rotation_matrix[1, 0] + y * rotation_matrix[1, 1])
                
                # Check bounds and update kernel
                if 0 <= new_x < size and 0 <= new_y < size:
                    kernel[new_x, new_y] = 1
    
    # Normalize the kernel
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_filter(size, sigma):
    """
    Create a Gaussian filter.
    Args:
        size: filter size (e.g., [3, 3]), sigma: standard deviation
    """
    x, y = np.meshgrid(np.linspace(-(size[0]//2), size[0]//2, size[0]),
                       np.linspace(-(size[1]//2), size[1]//2, size[1]))
    d = x**2 + y**2
    H = np.exp(-d / (2 * sigma**2))
    return H / H.sum()  # Normalize to sum to 1