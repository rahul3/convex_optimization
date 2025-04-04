"""
Implementation of Primal Douglas-Rachford Splitting Algorithm
"""

import torch
import numpy as np
from scipy import ndimage

from ..core.convolution import circular_convolve2d
from ..core.noise import add_gaussian_noise, create_motion_blur_kernel
from ..core.proximal_operators import prox_l1, prox_box, prox_iso
from ..utils.conv_utils import read_image, display_images, display_complex_output
from deblur_denoise.op_math.python_code.multiplying_matrix import DeblurDenoiseOperators

def primal_dr_splitting(problem: str, kernel: torch.Tensor, b: torch.Tensor,
                        image_path: str = None, shape: tuple = (100, 100),
                        i: dict = {
                            'maxiter': 500, 
                            'gammal1': 0.049,
                            'gammal2': 0.049,
                            'tprimaldr': 2.0,
                            'rhoprimaldr': 0.1,
                            'tol': 10**-6
                          }
                        ) -> torch.Tensor:
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
    # Initialize z1 and z2
    z1 = b
    z2 = torch.stack((b, b, b))
    z1prev = z1.detach().clone()
    op = DeblurDenoiseOperators(kernel, b, i.get('tprimaldr'))

    exited_via_break = False
    for j in range(i.get('maxiter')):
        # Update x
        x = prox_box(z1, i.get('tprimaldr'))
        
        # Update y
        iso = None
        norm = None
        if problem == 'l1':
            # iso norm on y2, y3 parts
            iso = prox_iso(z2[[1,2],:,:], i.get('tprimaldr') * i.get('gammal1'))
            # l1 norm on y1-b part
            norm = b + prox_l1(z2[0,:,:] - b, i.get('tprimaldr'))
        else:
            # iso norm on y2, y3 parts
            iso = prox_iso(z2[[1,2],:,:], i.get('tprimaldr') * i.get('gammal2'))
            # l2 norm squared on y1-b part
            norm = b + prox_l2_squared(z2[0,:,:] - b, i.get('tprimaldr'))
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
        z1 = z1 + i.get('rhoprimaldr') * (u - x)
        z2 = z2 + i.get('rhoprimaldr') * (v - y)
        # real part only (imaginary part should be 0)
        #z1 = z1.real
        #z2 = z2.real

        if j == 0 or j == 15: # debug
        #    print(prox_box(z1, i.get('tprimaldr')))
            print(_l2_norm(z1, z1prev))
        #    plt.imshow(prox_box(z1, i.get('tprimaldr')).squeeze().numpy(), cmap='gray')
        #    plt.show()
        
        if _l2_norm(z1, z1prev) < i.get('tol'):
            exited_via_break = True
            break
    
    if not exited_via_break:
        print(f"Warning: maxiter reached ({i.get('maxiter')}), primal_dr did not converge")
        print(_l2_norm(z1, z1prev))
    
    return prox_box(z1, i.get('tprimaldr'))


def _l2_norm(x: torch.Tensor, y: torch.Tensor) -> float:
    '''
    Returns the l2 norm of the elements of x-y. That is, if we were to treat x and y as flattened
    vectors, it returns ||x-y||, using the l2 norm.
    '''
    return float(np.sqrt(torch.sum((x-y) * (x-y))))

# test
def primal_dr_splitting_test():
    import matplotlib.pyplot as plt
    from ..core.noise import create_motion_blur_kernel
    from ..core.convolution import circular_convolve2d
    from ..utils.conv_utils import read_image

    image = read_image("/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/other_images/manWithHat.tiff", shape=(500,500))

    motion_kernel = create_motion_blur_kernel(size=10, angle=45)
    motion_blurred = circular_convolve2d(image, motion_kernel)

    #plt.subplot(1, 2, 1)
    #plt.imshow(image.squeeze().numpy(), cmap="gray")
    #plt.title("Original Image")
    #plt.axis("off")

    #plt.subplot(1, 2, 2)
    #plt.imshow(motion_blurred.squeeze().numpy(), cmap="gray")
    #plt.title("Motion Blur")
    #plt.axis("off")

    #plt.tight_layout()
    #plt.show()

    res = primal_dr_splitting('l2', create_motion_blur_kernel(), 
                            motion_blurred.squeeze(),
                            {
                                'maxiter': 500, 
                                'gammal1': 0.049,
                                'gammal2': 0.049,
                                'tprimaldr': 1.5,
                                'rhoprimaldr': 0.05,
                                'tol': 10**-6
                            })

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
