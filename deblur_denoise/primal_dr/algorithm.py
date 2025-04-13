"""
Implementation of Primal Douglas-Rachford Splitting Algorithm
"""

import torch
import numpy as np
import time
from scipy import ndimage
from typing import Callable

from ..core.convolution import circular_convolve2d
from ..core.blur import create_motion_blur_kernel, gaussian_filter
from ..core.proximal_operators import prox_l1, prox_box, prox_iso, prox_l2_squared
from ..utils.conv_utils import read_image, display_images, display_complex_output
from ..op_math.python_code.multiplying_matrix import DeblurDenoiseOperators
from ..utils.logging_utils import log_execution_time, logger, save_loss_data
from ..core.loss import psnr, l1_loss, l2_loss, mse, ssim


@log_execution_time(logger)
def primal_dr_splitting(problem: str, kernel: torch.Tensor, 
                        b: torch.Tensor,
                        image_path: str = None, 
                        shape: tuple = (100, 100),
                        loss_function: Callable=ssim,
                        # old code below
                        # i: dict = {
                        #     'maxiter': 500, 
                        #     'gammal1': 0.049,
                        #     'gammal2': 0.049,
                        #     'tprimaldr': 2.0,
                        #     'rhoprimaldr': 0.1,
                        #     'tol': 10**-6
                        #   },
                        niters: int=500,
                        gamma: float=0.049,
                        t: float=2.0,
                        rho: float=0.1,
                        tol: float=10**-6,
                        display: bool=False,
                        save_loss: bool=False,
                        **kwargs) -> torch.Tensor:
    """
    Primal Douglas-Rachford Splitting Algorithm.

    problem: "l1" or "l2", indicating which version of the problem we're solving
    kernel: the kernel k used to do the convolution
    b: the blurry image as a matrix
    image_path: the path to the image
    shape: the shape of the image
    i: a dictionary with additional parameters. For primal_dr_splitting, it uses
        the following parameters (as keys in the dictionary):
        * maxiter (max allowed number of iterations)
        * gammal1 (the gamma value used for the iso norm in the l1 version of the problem)
        * gammal2 (the gamma value used for the iso norm in the l2 version of the problem)
        * tprimaldr (the value of t used in the algorithm)
        * rhoprimaldr (the value of rho used in the algorithm)
        * tol: stop when consecutive iterations are less than this distance from each other

    Returns the resulting deblurred image as a tensor.
    """
    start_time = time.time()
    loss_list = []
    psnr_list = []
    ssim_list = []
    mse_list = []
    l1_loss_list = []
    l2_loss_list = []

    # initialize all paramters to match old ones
    tprimaldr = t
    gammal1 = gamma
    gammal2 = gamma
    rhoprimaldr = rho
    maxiter = niters
    loss_fn_name = getattr(loss_function, '__name__', str(loss_function))

    logger.info(f"Running Primal Douglas-Rachford Splitting with parameters:")
    logger.info(f"  t: {t}")
    logger.info(f"  gamma: {gamma}")
    logger.info(f"  rho: {rho}")
    logger.info(f"  tol: {tol}")
    logger.info(f"  niters: {niters}")
    logger.info(f"  save_loss: {save_loss}")
    logger.info("*" * 100)

    # Initialize z1 and z2
    z1 = b
    z2 = torch.stack((b, b, b))
    z1prev = z1.detach().clone()
    op = DeblurDenoiseOperators(kernel, b, tprimaldr)

    exited_via_break = False
    for j in range(niters):
        # Update x
        x = prox_box(z1, tprimaldr)
        
        # Update y
        iso = None
        norm = None
        if problem == 'l1':
            # iso norm on y2, y3 parts
            iso = prox_iso(z2[[1,2],:,:], tprimaldr * gammal1)
            # l1 norm on y1-b part
            norm = b + prox_l1(z2[0,:,:] - b, tprimaldr)
        else:
            # iso norm on y2, y3 parts
            iso = prox_iso(z2[[1,2],:,:], tprimaldr * gammal2)
            # l2 norm squared on y1-b part
            norm = b + prox_l2_squared(z2[0,:,:] - b, tprimaldr)
        y = torch.stack((norm, iso[0,:,:], iso[1,:,:]))

        # Update u
        # A^T matrix multiplication part
        arg = 2 * y - z2
        A_transpose_arg = op.apply_KTrans(arg[0,:,:]) + op.apply_D1Trans(arg[1,:,:]) + op.apply_D2Trans(arg[2,:,:])
        A_transpose_arg = A_transpose_arg.real
        # (I+A^TA)^{-1} matrix multiplication part
        arg = 2 * x - z1 + A_transpose_arg
        u = op.invert_matrix(arg).real

        # Update v
        v = torch.stack((op.apply_K(u), op.apply_D1(u), op.apply_D2(u)))
        v = v.real

        # Update z1 and z2
        z1prev = z1.detach().clone()
        z1 = z1 + rhoprimaldr * (u - x)
        z2 = z2 + rhoprimaldr * (v - y)
        # real part only (imaginary part should be 0)
        #z1 = z1.real
        #z2 = z2.real

        if j == 0 or j == 15: # debug
            logger.info(loss_function(z1, b))

        if j % 50 == 0:
            loss_val = loss_function(z1, b)
            if type(loss_val) == torch.Tensor:
                loss_val = loss_val.item()
            logger.info(f"Iteration {j} completed. {loss_fn_name} loss : {loss_val}")
        
        if j > 1:
            loss_val = loss_function(z1, z1prev)
            psnr_val = psnr(z1, b)
            ssim_val = ssim(z1, b)
            mse_val = mse(z1, b)
            l1_loss_val = l1_loss(z1, b)
            l2_loss_val = l2_loss(z1, b)
            if type(loss_val) == torch.Tensor:
                loss_val = loss_val.item()
            if save_loss:
                loss_list.append(loss_val)
                psnr_list.append(psnr_val.item() if type(psnr_val) == torch.Tensor else psnr_val)
                ssim_list.append(ssim_val.item() if type(ssim_val) == torch.Tensor else ssim_val)
                mse_list.append(mse_val.item() if type(mse_val) == torch.Tensor else mse_val)
                l1_loss_list.append(l1_loss_val.item() if type(l1_loss_val) == torch.Tensor else l1_loss_val)
                l2_loss_list.append(l2_loss_val.item() if type(l2_loss_val) == torch.Tensor else l2_loss_val)
            if loss_val < tol:
                exited_via_break = True
                break
    
    if not exited_via_break:
        logger.info(f"Warning: maxiter reached ({niters}), primal_dr did not converge")
        logger.info(loss_function(z1, z1prev))

    if save_loss and loss_list:
        parameters = {
            't': t,
            'gamma': gamma,
            'rho': rho,
            'niters': niters,
            'tol': tol,
            'psnr': psnr_list,
            'ssim': ssim_list,
            'mse': mse_list,
            'l1_loss': l1_loss_list,
            'l2_loss': l2_loss_list
        }
        df = save_loss_data(loss_list, 'primal_dr_splitting', loss_fn_name, parameters, start_time)
        timestamp = f'{int(start_time)}'
        # df.to_csv(f'primal_dr_splitting_{loss_fn_name}_{timestamp}.csv', index=False)

    solution = prox_box(z1, tprimaldr)
    if display:
        display_images(solution, b)
    
    
    return prox_box(z1, tprimaldr)


def _l2_norm(x: torch.Tensor, y: torch.Tensor) -> float:
    '''
    Returns the l2 norm of the elements of x-y. That is, if we were to treat x and y as flattened
    vectors, it returns ||x-y||, using the l2 norm.
    '''
    return float(np.sqrt(torch.sum((x-y) * (x-y))))

# test
def primal_dr_splitting_test(image_path: str,
                             blur_type: str="gaussian",
                             blur_kernel_size: int=10,
                             blur_kernel_sigma: float=0.8,
                             blur_kernel_angle: float=45,
                             image_shape: tuple=(500, 500)):
    import matplotlib.pyplot as plt
    from ..core.blur import create_motion_blur_kernel, gaussian_filter
    from ..core.convolution import circular_convolve2d
    from ..utils.conv_utils import read_image

    image = read_image(image_path, shape=image_shape)

    if blur_type == "motion":
        motion_kernel = create_motion_blur_kernel(size=blur_kernel_size, angle=blur_kernel_angle)
    elif blur_type == "gaussian":
        motion_kernel = gaussian_filter(size=[blur_kernel_size, blur_kernel_size], sigma=blur_kernel_sigma)
    motion_blurred = circular_convolve2d(image, motion_kernel)

    res = primal_dr_splitting('l2', create_motion_blur_kernel(), 
                            motion_blurred.squeeze(),
                            t=1.5,
                            gamma=0.049,
                            rho=0.05,
                            niters=500,
                            save_loss=True,
                            save_path='./results',
                            img_id='primal_dr_splitting_test',
                            loss_function=ssim,
                            tol=10**-6)

    plt.subplot(1,3,1)
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(motion_blurred.squeeze().numpy(), cmap='gray')
    plt.title('Blurred image')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(res.squeeze().numpy(), cmap='gray')
    plt.title('Deblurred image')
    plt.axis('off')
    plt.show()
