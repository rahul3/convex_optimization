from deblur_denoise.admm.algorithm import admm_solver
from deblur_denoise.chambolle_pock.algorithm import chambolle_pock
from deblur_denoise.primal_dual_dr.algorithm import primal_dual_dr_splitting
from deblur_denoise.primal_dr.algorithm import primal_dr_splitting

from deblur_denoise.core.convolution import circular_convolve2d
from deblur_denoise.core.noise import add_salt_pepper_noise, add_poisson_noise, add_speckle_noise, gaussian_filter, create_motion_blur_kernel
from deblur_denoise.utils.conv_utils import display_images, read_image

import torch

def blur_image(image_path: str,
               blur_type: str="gaussian",
               image_shape: tuple=(100, 100),
               blur_kernel_size: int=None,
               blur_kernel_sigma: float=None,
               blur_kernel_angle: float=None,
               salt_prob: float=None,
               pepper_prob: float=None,
               mean: float=None,
               std: float=None,
               scale: float=None,
               display: bool=False) -> torch.Tensor:
    """
    Blur an image based on the specified blur type and parameters.
    
    Args:
        image_path (str): The path to the image to blur.
        blur_type (str): The type of blur to apply.
        blur_kernel_size (int): The size of the blur kernel.
        blur_kernel_sigma (float): The sigma of the blur kernel.
        blur_kernel_angle (float): The angle of the blur kernel.
    """
    try:
        image = read_image(image_path, shape=image_shape)
    except Exception as e:
        raise ValueError(f"Error reading image: {e}")

    if blur_type == "gaussian":
        kernel = gaussian_filter([blur_kernel_size, blur_kernel_size], blur_kernel_sigma)
        noisy_image = circular_convolve2d(image, kernel)
    elif blur_type == "motion":
        kernel = create_motion_blur_kernel(blur_kernel_size, blur_kernel_angle)
        noisy_image = circular_convolve2d(image, kernel)
    elif blur_type == "salt_pepper":
        noisy_image = add_salt_pepper_noise(image, salt_prob, pepper_prob)
    elif blur_type == "poisson":
        noisy_image = add_poisson_noise(image, scale)
    elif blur_type == "speckle":
        noisy_image = add_speckle_noise(image, std)
    else:
        raise ValueError(f"Invalid blur type: {blur_type}")
    
    if display:
        display_images(image, noisy_image, title1="Original Image", title2=f"Noisy Image - {blur_type}")
    
    return noisy_image.squeeze()


def deblur_image(noisy_image: torch.Tensor,
                 algorithm: str="admm",
                 display: bool=False,
                 **kwargs) -> torch.Tensor:
    """
    Deblur an image based on the specified algorithm.
    """
    t = kwargs.get("t", 18)
    rho = kwargs.get("rho", 0.001)
    gamma = kwargs.get("gamma", 0.5)
    niters = kwargs.get("niters", 1000)
    s = kwargs.get("s", 0.7)
    kernel = kwargs.get("kernel", None)

    if algorithm == "admm":
        deblurred_image = admm_solver(b=noisy_image, kernel=kernel, t=t, rho=rho, gamma=gamma, niters=niters)
    elif algorithm == "chambolle_pock":
        deblurred_image = chambolle_pock(b=noisy_image.unsqueeze(0), kernel=kernel, t=t, s=s, gamma=gamma, max_iter=niters)
    elif algorithm == "primal_dual_dr":
        deblurred_image = primal_dual_dr_splitting(noisy_image)
    elif algorithm == "primal_dr":
        deblurred_image = primal_dr_splitting(noisy_image)
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not implemented")
    
    if display:
        display_images(noisy_image, deblurred_image.squeeze(), title1="Noisy Image", title2=f"Deblurred Image - {algorithm}")

    return deblurred_image


def blur_and_deblur_image(image_path: str,
                          blur_type: str="gaussian",
                          blur_kernel_size: int=5,
                          blur_kernel_sigma: float=0.8,
                          display: bool=False,
                          kernel: torch.Tensor=None,
                          algorithm: str="chambolle_pock",
                          t: float=0.4,
                          s: float=0.7,
                          gamma: float=0.01,
                          max_iter: int=1000) -> torch.Tensor:
    """
    Blur and deblur an image.
    """
    if kernel is None:
        kernel = gaussian_filter([blur_kernel_size, blur_kernel_size], blur_kernel_sigma)
        kernel = torch.from_numpy(kernel)
    blurred_image = blur_image(image_path=image_path,
                              image_shape=(300, 300),
                              blur_type=blur_type,
                              blur_kernel_size=blur_kernel_size,
                              blur_kernel_sigma=blur_kernel_sigma,
                              display=display)
    deblurred_image = deblur_image(blurred_image,
                                  kernel=kernel,
                                  algorithm=algorithm,
                                  t=t,
                                  s=s,
                                  gamma=gamma,
                                  max_iter=max_iter,
                                  display=display)
    return deblurred_image

if __name__ == "__main__":
    blur_and_deblur_image(image_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/final_report/mcgill.jpg",
                          blur_type="gaussian",
                          blur_kernel_size=5,
                          blur_kernel_sigma=0.8,
                          display=True)
