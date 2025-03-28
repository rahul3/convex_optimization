import sys
import os
import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt


# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.conv_utils import read_image
from noise import add_gaussian_noise, add_salt_pepper_noise, add_poisson_noise, add_speckle_noise, create_motion_blur_kernel


def circular_convolve2d(image, kernel):
    """ Perform 2D circular convolution using FFT. """

    _, H, W = image.shape
    if len(kernel.shape) == 2:
        kh, kw = kernel.shape
    else:
        kh, kw = kernel.shape[0], kernel.shape[1]

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if isinstance(kernel, np.ndarray):
        kernel = torch.from_numpy(kernel)

    # Zero-pad the kernel to match image size
    padded_kernel = torch.zeros((H, W), dtype=torch.float32)
    padded_kernel[:kh, :kw] = kernel
    padded_kernel = torch.roll(padded_kernel, shifts=(-kh//2, -kw//2), dims=(0, 1))  # Center kernel

    # Compute FFTs
    image_fft = torch.fft.fft2(image)
    kernel_fft = torch.fft.fft2(padded_kernel)

    # Perform element-wise multiplication in frequency domain
    blurred_fft = image_fft * kernel_fft

    # Compute inverse FFT to get the circularly convolved image
    blurred_image = torch.fft.ifft2(blurred_fft).real

    return blurred_image





def test_circular_convolve2d_with_noise_and_motion(img_path: str | None = None):
    """
    Test the circular_convolve2d function with different types of noise and motion blur.
    """
    if img_path is None:
        image = read_image("/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/other_images/grok_generated_image.jpg", grayscale=True, shape=(100, 100))
    else:
        print(f"Loading image from {img_path} ...")
        image = read_image(img_path, grayscale=True, shape=(100, 100))

    # Create different blur kernels
    gaussian_kernel = torch.tensor([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]], dtype=torch.float32)
    gaussian_kernel /= gaussian_kernel.sum()

    # Create motion blur kernels with different angles
    motion_kernel_0 = create_motion_blur_kernel(size=15, angle=0)  # horizontal
    motion_kernel_45 = create_motion_blur_kernel(size=15, angle=45)  # diagonal
    motion_kernel_90 = create_motion_blur_kernel(size=15, angle=90)  # vertical

    # Apply different types of blur
    gaussian_blurred = circular_convolve2d(image, gaussian_kernel)
    
    # Apply motion blur
    motion_blurred_0 = circular_convolve2d(image, motion_kernel_0)
    motion_blurred_45 = circular_convolve2d(image, motion_kernel_45)
    motion_blurred_90 = circular_convolve2d(image, motion_kernel_90)

    # Add different types of noise to the blurred images
    noisy_gaussian = add_gaussian_noise(gaussian_blurred, mean=0.0, std=0.05)
    noisy_salt_pepper = add_salt_pepper_noise(gaussian_blurred, salt_prob=0.01, pepper_prob=0.01)
    noisy_poisson = add_poisson_noise(gaussian_blurred, scale=2.0)
    noisy_speckle = add_speckle_noise(gaussian_blurred, std=0.1)
    noisy_motion = add_gaussian_noise(motion_blurred_45, mean=0.0, std=0.05)

    # Display results
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(3, 3, 1)
    plt.imshow(image.squeeze().numpy(), cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    # Gaussian blur
    plt.subplot(3, 3, 2)
    plt.imshow(gaussian_blurred.squeeze().numpy(), cmap="gray")
    plt.title("Gaussian Blur")
    plt.axis("off")

    # Gaussian blur with Gaussian noise
    plt.subplot(3, 3, 3)
    plt.imshow(noisy_gaussian.squeeze().numpy(), cmap="gray")
    plt.title("Gaussian Blur + Gaussian Noise")
    plt.axis("off")

    # Motion blur (0 degrees)
    plt.subplot(3, 3, 4)
    plt.imshow(motion_blurred_0.squeeze().numpy(), cmap="gray")
    plt.title("Motion Blur (0°)")
    plt.axis("off")

    # Motion blur (45 degrees)
    plt.subplot(3, 3, 5)
    plt.imshow(motion_blurred_45.squeeze().numpy(), cmap="gray")
    plt.title("Motion Blur (45°)")
    plt.axis("off")

    # Motion blur (90 degrees)
    plt.subplot(3, 3, 6)
    plt.imshow(motion_blurred_90.squeeze().numpy(), cmap="gray")
    plt.title("Motion Blur (90°)")
    plt.axis("off")

    
    plt.subplot(3, 3, 7)
    plt.imshow(motion_blurred_0.squeeze().numpy(), cmap="gray")
    plt.title("Motion Blur (0°)")
    plt.axis("off")

    plt.subplot(3, 3, 8)
    plt.imshow(motion_blurred_45.squeeze().numpy(), cmap="gray")
    plt.title("Motion Blur (45°)")
    plt.axis("off")

    plt.subplot(3, 3, 9)
    plt.imshow(noisy_motion.squeeze().numpy(), cmap="gray")
    plt.title("Motion Blur (45°) + Noise")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Visualize the kernels
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(motion_kernel_0.numpy(), cmap="gray")
    plt.title("Motion Kernel (0°)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(motion_kernel_45.numpy(), cmap="gray")
    plt.title("Motion Kernel (45°)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(motion_kernel_90.numpy(), cmap="gray")
    plt.title("Motion Kernel (90°)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_circular_convolve2d_with_noise_and_motion()