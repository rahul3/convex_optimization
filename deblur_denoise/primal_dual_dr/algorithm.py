"""
Implementation of Primal-Dual Douglas-Rachford Splitting Algorithm
"""

import torch
from dev.python_code.multiplying_matrix import * 
from utils.conv_utils import *
from core.convolution import circular_convolve2d
from core.proximal_operators import prox_box, prox_l1, prox_iso

import numpy as np
from scipy import ndimage

def primal_dual_dr_splitting(b: torch.Tensor , kernel : torch.Tensor, 
                              max_iter=1000, t = 3, rho = 0.4, gamma = 0.03, tol=1e-6):
    """
    Primal-Dual Douglas-Rachford Splitting Algorithm
    
    b - blurred image
    kernel - kernel used to convolve the original image
    x0 - initialized vector (for p)
    y0 - initialized vector (for q)
    
    t - step size
    rho - relaxation parameter
    gamma - parameter multiplying the penalty norm
  
    """

    # Initializing vectors
    n_rows, n_cols = b.squeeze().shape
    p = torch.rand(size=(n_rows, n_cols))
    q = torch.rand(size=(3, n_rows, n_cols))

    # Intializing the deblur denoise operators
    dd_ops = DeblurDenoiseOperators(kernel, b.squeeze(), t, t)
    x = p
    z = q

    for k in range(1, max_iter):
        x = prox_box(p, 1) # lambda value doesn't matter

        # computing proximal operator of g^\ast
        z1 = q[0] - t * b - t * prox_l1((1/t) * q[0] - b, (1/t))
        z2 = q[1:] - t  * prox_iso( (1/t)  *  q[1:], (1/t) * gamma)

        z = torch.cat((z1, z2), dim = 0)

        # Resolvent of B computation (using the given formula for the matrix inverse in the document)
        # Input to the matrix multiplication
        i1 = 2 * x - p
        i2 = 2 * z - q

        # [0 & 0 \\ 0 & I][i1 i2] 
        o = i2.clone()

        # I(i1) -tA^T(i2)
        i1 = i1 - t * dd_ops.apply_KTrans(i2[0]) - t * dd_ops.apply_DTrans(torch.stack((i2[1], i2[2]), dim = 0).permute(1,2,0))

        # (1 + t^2 A^TA)^{-1}(i1)
        i1 = dd_ops.invert_matrix(i1)

        # [ I tA] i1

        i2 = t * dd_ops.apply_K(i1.squeeze())
        i3 = t * dd_ops.apply_D(i1.squeeze())

        i2 = torch.cat((i2.unsqueeze(0), i3.permute(2,0,1)), dim = 0)

        # [ 0   0 \\ 0  I ] + i
        w  = i1.clone()

        v = i2 + o

        p = torch.real(p + rho * (w - x))
        q = torch.real(q + rho * (v - z))

    x_sol = prox_box(p, 1)
    return x_sol


if __name__=="__main__":
    #Testing
    
    print(0)

