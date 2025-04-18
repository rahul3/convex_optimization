import torch
import numpy as np
from datetime import datetime
from typing import Callable

from deblur_denoise.admm.algorithm import admm_solver
from deblur_denoise.chambolle_pock.algorithm import chambolle_pock
from deblur_denoise.primal_dual_dr.algorithm import primal_dual_dr_splitting
from deblur_denoise.primal_dr.algorithm import primal_dr_splitting

from deblur_denoise.core.convolution import circular_convolve2d
from deblur_denoise.core.noise import add_salt_pepper_noise, add_poisson_noise, add_speckle_noise, add_gaussian_noise
from deblur_denoise.core.blur import gaussian_filter, create_motion_blur_kernel 
from deblur_denoise.utils.conv_utils import display_images, read_image
from deblur_denoise.core.loss import ssim, psnr, mse, l1_loss, l2_loss
from deblur_denoise.utils.logging_utils import logger, log_execution_time


def blur_image(image_path: str,
               kernel: torch.Tensor=None,
               blur_type: str="gaussian",
               noise_type: str="gaussian",
               image_shape: tuple=(100, 100),
               blur_kernel_size: int=None,
               blur_kernel_sigma: float=None,
               blur_kernel_angle: float=None,
               salt_prob: float=None,
               pepper_prob: float=None,
               mean: float=None,
               std: float=None,
               scale: float=None,
               display: bool=False,
               save_path: str=None,
               img_id: str=None) -> torch.Tensor:
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
        if kernel is None:
            kernel = gaussian_filter([blur_kernel_size, blur_kernel_size], blur_kernel_sigma)
        blurred_image = circular_convolve2d(image, kernel)
    elif blur_type == "motion":
        if kernel is None:
            kernel = create_motion_blur_kernel(blur_kernel_size, blur_kernel_angle)
        blurred_image = circular_convolve2d(image, kernel)
    elif blur_type == "none":
        blurred_image = image
    else:
        raise NotImplementedError(f"Blur type {blur_type} not implemented")

    if noise_type == "gaussian":
        noisy_image = add_gaussian_noise(blurred_image, mean, std)
    elif noise_type == "salt_pepper":
        noisy_image = add_salt_pepper_noise(blurred_image, salt_prob, pepper_prob)
    elif noise_type == "poisson":
        noisy_image = add_poisson_noise(blurred_image, scale)
    elif noise_type == "speckle":
        noisy_image = add_speckle_noise(blurred_image, std)
    elif noise_type == "none":
        noisy_image = blurred_image
    else:
        raise NotImplementedError(f"Noise type {noise_type} not implemented")
    
    if display:
        display_images(image,
                       noisy_image,
                       title1="Original Image", title2=f"Blur: {blur_type} - Noise: {noise_type}", save_path=save_path, img_id=img_id)
    
    return noisy_image.squeeze()


def deblur_image(noisy_image: torch.Tensor,
                 algorithm: str="admm",
                 display: bool=False,
                 loss_function: Callable=ssim,
                 save_loss: bool=False,
                 save_path: str=None,
                 img_id: str=None,
                 **kwargs) -> torch.Tensor:
    """
    Deblur an image based on the specified algorithm.
    """
    if img_id is None:
        img_id = f'{algorithm}_{loss_function.__name__}'

    t = kwargs.get("t", 18)
    rho = kwargs.get("rho", 0.001)
    gamma = kwargs.get("gamma", 0.5)
    niters = kwargs.get("niters", 1000)
    tol = kwargs.get("tol", 1e-6)
    s = kwargs.get("s", 0.7)
    kernel = kwargs.get("kernel", None)

    if algorithm == "admm":
        deblurred_image = admm_solver(b=noisy_image, kernel=kernel, t=t, rho=rho, gamma=gamma, niters=niters, loss_function=loss_function, save_loss=save_loss)
    elif algorithm == "chambolle_pock":
        deblurred_image = chambolle_pock(b=noisy_image.unsqueeze(0), kernel=kernel, t=t, s=s, gamma=gamma, max_iter=niters, loss_function=loss_function, save_loss=save_loss)
    elif algorithm == "primal_dual_dr":
        deblurred_image = primal_dual_dr_splitting(b=noisy_image, kernel=kernel, niters=niters, t=t, rho=rho, gamma=gamma, tol=tol, loss_function=loss_function, save_loss=save_loss)
    elif algorithm == "primal_dr":
        deblurred_image = primal_dr_splitting(problem="primal_dr", b=noisy_image, kernel=kernel, t=t, gamma=gamma, rho=rho, tol=tol, niters=niters, loss_function=loss_function, save_loss=save_loss) 
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not implemented")
    
    if display:
        display_images(noisy_image,
                       deblurred_image.squeeze(),
                       title1="Noisy Image",
                       title2=f"Deblurred Image - {algorithm}",
                       save_path=save_path,
                       img_id=img_id)

    return deblurred_image


@log_execution_time(logger)
def blur_and_deblur_image(image_path: str,
                          image_shape: tuple=(300, 300),
                          experiment_id: str=None,
                          blur_type: str="gaussian",
                          noise_type: str="gaussian",
                          blur_kernel_size: int=5,
                          blur_kernel_sigma: float=0.8,
                          blur_kernel_angle: float=45,
                          display: bool=False,
                          kernel: torch.Tensor=None,
                          algorithm: str="chambolle_pock",
                          t: float=0.4,
                          s: float=0.7,
                          rho: float=0.4,
                          gamma: float=0.01,
                          salt_prob: float=0.15,
                          pepper_prob: float=0.15,
                          mean: float=0.0,
                          std: float=0.1,
                          scale: float=1.0,
                          niters: int=1000,
                          save_loss: bool=False,
                          save_path: str=None,
                          img_id: str=None,
                          loss_function: Callable=ssim) -> torch.Tensor:
    """
    Blur and deblur an image.
    Parameters:
    -----------
    image_path: str 
        Path to the input image file
    image_shape: tuple 
        Desired shape to resize the image (width, height)
    experiment_id: str 
        Unique identifier for the experiment
    blur_type: str 
        Type of blur to apply ("gaussian" or "motion")
    noise_type: str 
        Type of noise to add ("gaussian", "salt_pepper", "poisson", or "speckle")
    algorithm: str 
        Deblurring algorithm to use ("chambolle_pock", "admm", "primal_dr", or "primal_dual_dr")
    kernel: torch.Tensor 
        Optional pre-defined blur kernel (if None, one will be generated)
    blur_kernel_size: int 
        Size of the blur kernel
    blur_kernel_sigma: float 
        Sigma parameter for Gaussian blur
    blur_kernel_angle: float 
        Angle parameter for motion blur (in degrees)
    display: bool 
        Whether to display the images during processing
    t: float 
        Step size parameter used in optimization algorithms
    s: float 
        Step size parameter used in Chambolle-Pock algorithm
    rho: float 
        Regularization parameter used in ADMM and DR algorithms
    gamma: float 
        Regularization parameter for TV norm
    salt_prob: float 
        Probability of salt noise (white pixels) for salt_pepper noise
    pepper_prob: float 
        Probability of pepper noise (black pixels) for salt_pepper noise
    mean: float 
        Mean of Gaussian noise
    std: float 
        Standard deviation of Gaussian noise
    scale: float 
        Scale parameter for Poisson and speckle noise
    niters: int 
        Maximum number of iterations for optimization algorithms
    save_loss: bool 
        Whether to save the loss values during optimization
    save_path: str 
        Path to save the images and loss values
    img_id: str 
        Unique identifier for the image
    loss_function: Callable 
        Loss function to evaluate convergence (e.g., ssim, psnr, mse)
    
    Returns:
    --------
    torch.Tensor
        The deblurred image
    """
    if experiment_id is None:
        experiment_id = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    logger.info("*" * 100)
    logger.info(f"  experiment_id: {experiment_id}")
    logger.info("*" * 100)

    # Log all parameters
    logger.info("Running blur_and_deblur_image with parameters:")
    logger.info(f"  image_path: {image_path}")
    logger.info(f"  image_shape: {image_shape}")
    logger.info(f"  blur_type: {blur_type}")
    logger.info(f"  noise_type: {noise_type}")
    logger.info(f"  algorithm: {algorithm}")
    logger.info(f"  kernel provided: {kernel is not None}")
    logger.info(f"  blur_kernel_size: {blur_kernel_size}")
    logger.info(f"  blur_kernel_sigma: {blur_kernel_sigma}")
    logger.info(f"  blur_kernel_angle: {blur_kernel_angle}")
    logger.info(f"  display: {display}")
    logger.info(f"  t: {t}")
    logger.info(f"  s: {s}")
    logger.info(f"  rho: {rho}")
    logger.info(f"  gamma: {gamma}")
    logger.info(f"  salt_prob: {salt_prob}")
    logger.info(f"  pepper_prob: {pepper_prob}")
    logger.info(f"  mean: {mean}")
    logger.info(f"  std: {std}")
    logger.info(f"  scale: {scale}")
    logger.info(f"  niters: {niters}")
    logger.info(f"  loss_function: {loss_function.__name__ if hasattr(loss_function, '__name__') else str(loss_function)}")
    logger.info(f"  save_loss: {save_loss}")
    logger.info(f"  save_path: {save_path}")
    logger.info(f"  img_id: {img_id}")
    logger.info("*" * 100)

    if img_id is None:
        img_id = f'{algorithm}_{loss_function.__name__}_{experiment_id}'

    # Generate a kernel if not provided
    if kernel is None:
        if blur_type == "gaussian":
            kernel = gaussian_filter(size=[blur_kernel_size, blur_kernel_size], sigma=blur_kernel_sigma)
        elif blur_type == "motion":
            kernel = create_motion_blur_kernel(size=blur_kernel_size, angle=blur_kernel_angle)
        else:
            raise ValueError(f"Unsupported blur type: {blur_type}")
        
        if type(kernel) == np.ndarray:
            kernel = torch.from_numpy(kernel)


    # Blur the image
    blurred_image = blur_image(image_path=image_path,
                              image_shape=image_shape,
                              blur_type=blur_type,
                              noise_type=noise_type,
                              blur_kernel_size=blur_kernel_size,
                              blur_kernel_sigma=blur_kernel_sigma,
                              blur_kernel_angle=blur_kernel_angle,
                              salt_prob=salt_prob,
                              pepper_prob=pepper_prob,
                              mean=mean,
                              std=std,
                              scale=scale,
                              display=display)
    
    # Deblur the image
    deblurred_image = deblur_image(blurred_image,
                                  kernel=kernel,
                                  algorithm=algorithm,
                                  t=t,
                                  s=s,
                                  rho=rho,
                                  gamma=gamma,
                                  niters=niters,
                                  display=display,
                                  loss_function=loss_function,
                                  save_loss=save_loss,
                                  save_path=save_path,
                                  img_id=img_id)
    
    original_image = read_image(image_path, shape=image_shape)

    # make both images the same shape   
    if deblurred_image.shape != original_image.shape:
        if deblurred_image.shape[0] == 1:
            deblurred_image = deblurred_image.squeeze(0)
        elif original_image.shape[0] == 1:
            original_image = original_image.squeeze(0)
        else:
            raise ValueError(f"Deblurred image and original image have different shapes: {deblurred_image.shape} and {original_image.shape}")

    
    psnr_val = psnr(deblurred_image, original_image).item()
    ssim_val = ssim(deblurred_image, original_image).squeeze().item()
    mse_val = mse(deblurred_image, original_image).item()
    l1_loss_val = l1_loss(deblurred_image, original_image).item()
    l2_loss_val = l2_loss(deblurred_image, original_image).item()
    logger.info("*" * 100)
    logger.info("*" * 20 + " " * 5 + "Loss metrics w.r.t. original image" + " " * 5 + "*" * 20)
    logger.info(f"PSNR: {psnr_val}")
    logger.info(f"SSIM: {ssim_val}")
    logger.info(f"MSE: {mse_val}")
    logger.info(f"L1 Loss: {l1_loss_val}")
    logger.info(f"L2 Loss: {l2_loss_val}")
    logger.info("*" * 100)
    return deblurred_image

if __name__ == "__main__":
    from deblur_denoise.core.loss import ssim, psnr

    # This is how the professor can call our code.

    # Example 1:
    # Gaussian blur and gaussian noise with admm with ssim loss function (default)
    blur_and_deblur_image(image_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg",
                          image_shape=(300, 300),
                          blur_type="gaussian",
                          noise_type="gaussian",
                          blur_kernel_size=5,
                          blur_kernel_sigma=0.8,
                          niters=500,
                          algorithm="admm",
                          display=True,
                          save_loss=True,
                          save_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/results")
    # Example 2:
    # Motion blue with gaussian noise and chambolle_pock with ssim loss function (default   )
    # blur_and_deblur_image(image_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg",
    #                       image_shape=(300, 300),
    #                       blur_type="motion",
    #                       noise_type="gaussian",
    #                       blur_kernel_size=5,
    #                       blur_kernel_angle=45,
    #                       algorithm="primal_dr",
    #                       niters=100,
    #                       save_loss=True,
    #                       save_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/results",
    #                       display=True)
    
    # Example 3:
    # Salt and pepper noise and deblur with chambolle_pock (which is the default algorithm) with PSNR loss function (default)
    # blur_and_deblur_image(image_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg",
    #                       image_shape=(300, 300),
    #                       blur_type="gaussian",
    #                       noise_type="gaussian",
    #                       blur_kernel_size=5,
    #                       blur_kernel_sigma=0.8,
    #                       niters=100,
    #                       save_loss=True,
    #                       save_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/results",
    #                       display=True)
    
    # Example 4:
    # Motion blur and salt and pepper noise with chambolle_pock with ssim loss function
    # blur_and_deblur_image(image_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg",
    #                       image_shape=(100, 100),
    #                       blur_type="motion",
    #                       noise_type="salt_pepper",
    #                       blur_kernel_size=3,
    #                       blur_kernel_angle=30,
    #                       salt_prob=0.05,
    #                       pepper_prob=0.05,
    #                       algorithm="chambolle_pock",
    #                       loss_function=ssim,
    #                       niters=100,
    #                       save_loss=True,
    #                       save_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/results",
    #                       display=True)
    
    # Example 5:
    # Gaussian blur and gaussian noise with admm with ssim loss function
    # blur_and_deblur_image(image_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg",
    #                       image_shape=(300, 300),
    #                       blur_type="gaussian",
    #                       noise_type="gaussian",
    #                       blur_kernel_size=5,
    #                       blur_kernel_sigma=0.8,
    #                       algorithm="admm",
    #                       display=True,
    #                       save_loss=True,
    #                       save_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/results",
    #                       loss_function=ssim)

    # Example 6:
    # Motion blur and salt and pepper noise with chambolle_pock with ssim loss function
    # Running a grid search for the best parameters
    # blur_and_deblur_image(image_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg",
    #                     image_shape=(100, 100),
    #                     blur_type="motion",
    #                     noise_type="salt_pepper",
    #                     blur_kernel_size=3,
    #                     blur_kernel_angle=30,
    #                     salt_prob=0.05,
    #                     pepper_prob=0.05,
    #                     algorithm="chambolle_pock",
    #                     loss_function=ssim,
    #                     save_loss=True,
    #                     save_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/results",
    #                     display=True)

    # Example 7:
    # Gaussian blur and gaussian noise with admm with psnr loss function
    # blur_and_deblur_image(image_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg",
    #                     image_shape=(300, 300),
    #                     t=15,
    #                     gamma=0.3,
    #                     blur_type="gaussian",
    #                     noise_type="gaussian",
    #                     blur_kernel_size=5,
    #                     blur_kernel_sigma=0.8,
    #                     algorithm="admm",
    #                     display=True,
    #                     save_loss=True,
    #                     save_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/results",
    #                     loss_function=psnr)

    # Example 8:
    # Gaussian blur and gaussian noise with primal_dr with psnr loss function
    # blur_and_deblur_image(image_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg",
    #                 image_shape=(300, 300),
    #                 t=15,
    #                 gamma=0.3,
    #                 niters=500,
    #                 blur_type="gaussian",
    #                 noise_type="gaussian",
    #                 blur_kernel_size=5,
    #                 blur_kernel_sigma=0.8,
    #                 algorithm="primal_dr",
    #                 display=True,
    #                 save_loss=True,
    #                 save_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/results",
    #                 loss_function=psnr)
    
    # Example 9:
    # Gaussian blur and gaussian noise with primal_dr with psnr loss function
    # blur_and_deblur_image(image_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg",
    #                 image_shape=(300, 300),
    #                 t=15,
    #                 gamma=0.3,
    #                 niters=500,
    #                 blur_type="gaussian",
    #                 noise_type="gaussian",
    #                 blur_kernel_size=5,
    #                 blur_kernel_sigma=0.8,
    #                 algorithm="primal_dr",
    #                 display=True,
    #                 save_loss=True,
    #                 save_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/results",
    #                 loss_function=psnr)
    
    # Example 10:
    # Gaussian blur and gaussian noise with primal_dual_dr with psnr loss function
    # blur_and_deblur_image(image_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/sample_images/dog.jpg",
    #             image_shape=(300, 300),
    #             t=1.4,
    #             rho=0.9,
    #             gamma=0.02,
    #             niters=500,
    #             blur_type="gaussian",
    #             noise_type="gaussian",
    #             blur_kernel_size=5,
    #             blur_kernel_sigma=0.8,
    #             algorithm="primal_dual_dr",
    #             display=True,
    #             save_loss=True,
    #             save_path="/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/results",
    #             loss_function=psnr)
