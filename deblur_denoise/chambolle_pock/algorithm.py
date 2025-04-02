"""
Implementation of Chambolle-Pock Method
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dev.python_code.multiplying_matrix import * 
from utils.conv_utils import *
from core.convolution import circular_convolve2d
from core.noise import gaussian_filter
from core.proximal_operators import prox_box, prox_l1, prox_iso


def chambolle_pock(b: torch.Tensor,
                   kernel: torch.Tensor,
                   t: float=0.4,
                   s: float=0.7,
                   gamma: float=0.01,
                   max_iter: int=1000,
                   **kwargs) -> torch.Tensor:
    """
    Chambolle-Pock algorithm for deblurring and denoising
    """
    print(f'Running Chambolle-Pock with values: t={t}, s={s}, gamma={gamma}')
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
        c1 = g_ast_input[0] - s * b - s * prox_l1(g_ast_input[0]/s - b, 1/s)
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


    x_sol = x_next
    print(f"{x_sol.shape=}")
    return x_sol


def chambolle_pock_test():
    IMG_PATH = '/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/other_images/grok_generated_image.jpg'
    img = read_image(IMG_PATH, shape=(200,200))

    # Generate a kernel
    kernel = gaussian_filter([5,5], 0.8)
    kernel = torch.from_numpy(kernel)

    # Blur the image
    b = circular_convolve2d(img, kernel)


    print(f"{b.shape=}")

    x_sol = chambolle_pock(b, kernel)

    # display
    display_images(b, x_sol, title1="Blurred Image", title2="Deblurred Image")

