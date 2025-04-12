"""
Implementation of Chambolle-Pock Algorithm
"""

import torch
import numpy as np
from scipy import ndimage
from typing import Callable
from ..core.convolution import circular_convolve2d
from ..core.noise import create_motion_blur_kernel, gaussian_filter
from ..core.loss import l1_loss, l2_loss, mse, psnr, ssim
from ..core.proximal_operators import prox_l1, prox_l2_squared, prox_box, prox_iso
from ..utils.conv_utils import read_image, display_images, display_complex_output
from ..op_math.python_code.multiplying_matrix import DeblurDenoiseOperators
from ..utils.logging_utils import log_execution_time, logger

@log_execution_time(logger)
def chambolle_pock(b: torch.Tensor,
                   kernel: torch.Tensor,
                   objective_function: str="l1",
                   t: float=0.4,
                   s: float=0.7,
                   gamma: float=0.01,
                   max_iter: int=500,
                   loss_function: Callable=psnr,
                   **kwargs) -> torch.Tensor:
    """
    Chambolle-Pock algorithm for deblurring and denoising
    """
    logger.info(f'Running Chambolle-Pock with values: t={t}, s={s}, gamma={gamma}')
    dd_ops = DeblurDenoiseOperators(kernel, b.squeeze(), 1, 1)

    n_rows, n_cols = b.squeeze().shape

    x_prev = torch.rand(size=(n_rows, n_cols))
    z_prev = torch.rand(size=(n_rows, n_cols))
    y_prev = torch.rand(size=(3, n_rows, n_cols)) 

    x_next = x_prev.clone()
    y_next = y_prev.clone()
    z_next = z_prev.clone()

    for k in range(1, max_iter):
        # Computing A_z_prev
        K_z_prev = dd_ops.apply_K(z_prev.squeeze())
        D_z_prev = dd_ops.apply_D(z_prev.squeeze())

        A_z_prev = torch.cat((K_z_prev.unsqueeze(0), D_z_prev.permute(2,0,1)), dim = 0)

        # Input to prox of g^\ast
        g_ast_input = torch.real(y_prev + s * A_z_prev)

        # Computing the proximal of g^\ast
        if objective_function == "l1":
            c1 = g_ast_input[0] - s * b - s * prox_l1(g_ast_input[0]/s - b, 1/s)
        elif objective_function == "l2":
            c1 = g_ast_input[0] - s * b - s * prox_l2_squared(g_ast_input[0]/s - b, 1/s)
        else:
            raise ValueError("Invalid objective function. Choose 'l1' or 'l2'.")
        c2 = g_ast_input[1:] - s *  prox_iso(g_ast_input[1:]/s, 1/s * gamma)
        y_next = torch.cat((c1, c2), dim =0).clone()

        # A^Ty
        A_T_y = dd_ops.apply_KTrans(y_next[0]) + dd_ops.apply_DTrans(y_next[1:].permute(1,2,0))


        # Computing input to prox_box
        prox_box_input = torch.real(x_prev - t * A_T_y.unsqueeze(0))

        # Computing prox_box
        x_next = prox_box(prox_box_input, 1)
        z_next = (2 * x_next - x_prev).clone() 

        # Updating vars
        x_prev = x_next.clone()
        y_prev = y_next.clone()
        z_prev = z_next.clone()

        if k % 50 == 0:
            logger.info(f"Iteration {k} completed.")
            loss = loss_function(x_next, b)
            if type(loss) == torch.Tensor:
                loss = loss.item()

    x_sol = x_next
    logger.info(f"{x_sol.shape=}")

    return x_sol


def chambolle_pock_test(image_path: str,
                        image_shape: tuple=(200, 200),
                        blur_type: str="gaussian",
                        blur_kernel_size: int=5,
                        blur_kernel_sigma: float=0.8,
                        blur_kernel_angle: float=45):
    """
    Test the Chambolle-Pock algorithm
    """
    img = read_image(image_path, shape=image_shape)

    # Generate a kernel
    if blur_type == "gaussian":
        kernel = gaussian_filter(size=[blur_kernel_size, blur_kernel_size], sigma=blur_kernel_sigma)
    elif blur_type == "motion":
        kernel = create_motion_blur_kernel(size=[blur_kernel_size, blur_kernel_size], angle=blur_kernel_angle)
    kernel = torch.from_numpy(kernel)

    # Blur the image
    b = circular_convolve2d(img, kernel)


    logger.info(f"{b.shape=}")

    x_sol = chambolle_pock(b, kernel)

    # display
    display_images(b, x_sol, title1=f"Blurred Image - {blur_type}", title2="Deblurred Image")