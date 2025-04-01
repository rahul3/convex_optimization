"""
Implementation of Chambolle-Pock Method
"""

import torch
import numpy as np
from dev.python_code.multiplying_matrix import DeblurDenoiseOperators
from core.proximal_operators import prox_box, prox_iso, prox_l1, prox_l2_squared

def chambolle_pock_solver(problem: str, kernel: torch.Tensor, b: torch.Tensor,
                          i: dict = {
                              'maxiter': 500,
                              'gammal1': 0.049,
                              'gammal2': 0.049,
                              'tchambollepock': 2.0,
                              'schambollepock': 2.0,
                              'tol': 10**-6
                          }
                        ) -> torch.Tensor:
    """
    Chambolle-Pock Method

    problem: "l1" or "l2", indicating which version of the problem we're solving
    kernel: the kernel k used to do the convolution
    b: the blurry image as a matrix
    i: a dictionary with additional parameters. For primal_dr_splitting, it uses
        the following parameters (as keys in the dictionary):
        * maxiter (max allowed number of iterations)
        * gammal1 (the gamma value used for the iso norm in the l1 version of the problem)
        * gammal2 (the gamma value used for the iso norm in the l2 version of the problem)
        * tchambollepock (the value of t used in the algorithm)
        * schambollepock (the value of s used in the algorithm)
        * tol: stop when consecutive iterations are less than this distance from each other

    Returns the resulting deblurred image as a tensor.
    """
    # Initialize x, y, z
    x = b
    y = torch.stack((b,b,b))
    z = b
    xprev = x.detach().clone()
    op = DeblurDenoiseOperators(kernel, b, i.get('tchambollepock'), i.get('schambollepock'))
    t = i.get('tchambollepock')
    s = i.get('schambollepock')

    exited_via_break = False
    for j in range(i.get('maxiter')):
        # Update y
        iso = None
        norm = None
        arg = y + s * torch.stack((op.apply_K(z), op.apply_D1(z), op.apply_D2(z)))
        arg = arg.real
        if problem == 'l1':
            # iso norm on second, third parts of arg
            iso = prox_iso(arg[[1,2],:,:], s * i.get('gammal1'))
            # l1 norm on arg[1] - b
            norm = b + prox_l1(arg[0,:,:] - b, s)
        else:
            # iso norm on second, third parts of arg
            iso = prox_iso(arg[[1,2],:,:], s * i.get('gammal2'))
            # l1 norm on arg[1] - b
            norm = b + prox_l2_squared(arg[0,:,:] - b, s)
        y = y - torch.stack((norm, iso[0,:,:], iso[1,:,:]))

        # Update x
        xprev = x.detach().clone()
        arg = x - t * (op.apply_KTrans(y[0,:,:]) + 
                       op.apply_D1Trans(y[1,:,:]) + 
                       op.apply_D2Trans(y[2,:,:]))
        arg = arg.real
        x = prox_box(arg, t)

        # Update z
        z = 2 * x - xprev

        if j == 0 or j == 15: # debug
            print(_l2_norm(x, xprev))
        
        if _l2_norm(x, xprev) < i.get('tol'):
            exited_via_break = True
            break
    
    if not exited_via_break:
        print(f"Warning: maxiter reached ({i.get('maxiter')}), primal_dr did not converge")
        print(_l2_norm(x, xprev))

    return x

def _l2_norm(x: torch.Tensor, y: torch.Tensor) -> float:
    '''
    Returns the l2 norm of the elements of x-y. That is, if we were to treat x and y as flattened
    vectors, it returns ||x-y||, using the l2 norm.
    '''
    return float(np.sqrt(torch.sum((x-y) * (x-y))))

# test
import matplotlib.pyplot as plt
from core.noise import create_motion_blur_kernel
from core.convolution import circular_convolve2d
from utils.conv_utils import read_image

image = read_image("/home/lilian/phd_other/convex_optimization/img3.jpg", shape=(500,500))

motion_kernel = create_motion_blur_kernel(size=10, angle=45)
motion_blurred = circular_convolve2d(image, motion_kernel)

res = chambolle_pock_solver('l1', create_motion_blur_kernel(), 
                            motion_blurred.squeeze(),
                            {
                                'maxiter': 500,
                                'gammal1': 0.02, # 0.1 for l2 pic2
                                'gammal2': 0.025,
                                'tchambollepock': 60.0, # 60.0 for l1 pic3, 1.5 for l2 pic2
                                'schambollepock': 1.0,  # 1.0 for l1, 1.0 for l2 pic2
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

