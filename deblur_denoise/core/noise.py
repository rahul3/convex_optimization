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
