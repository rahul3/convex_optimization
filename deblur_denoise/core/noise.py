import torch
import numpy as np

def add_gaussian_noise(image, mean=0.0, std=0.1) -> torch.Tensor:
    """Add Gaussian noise to a tensor image."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01) -> torch.Tensor:
    """Add salt and pepper noise to a tensor image."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    noisy_image = image.clone()
    random_values = torch.rand_like(image)
    
    # Add salt noise (white pixels)
    salt_mask = random_values < salt_prob
    noisy_image[salt_mask] = 1.0
    
    # Add pepper noise (black pixels)
    pepper_mask = (random_values >= salt_prob) & (random_values < salt_prob + pepper_prob)
    noisy_image[pepper_mask] = 0.0
    
    return noisy_image

def add_poisson_noise(image, scale=1.0) -> torch.Tensor:
    """Add Poisson noise to a tensor image."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    scaled_image = image * 255.0 * scale
    noisy_image = torch.poisson(scaled_image) / (255.0 * scale)
    return torch.clamp(noisy_image, 0, 1)


def add_speckle_noise(image: torch.Tensor | np.ndarray, std=0.1) -> torch.Tensor:
    """Add speckle noise to a tensor image.
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    # Generate noise
    noise = torch.randn_like(image) * std
    
    # Multiply noise with image (characteristic of speckle noise)
    noisy_image = image + image * noise
    noisy_image = torch.clamp(noisy_image, 0, 1)
    
    return noisy_image

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