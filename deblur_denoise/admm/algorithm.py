"""
Implementation of ADMM (Dual Douglas-Rachford) Algorithm
"""
import torch
import numpy as np
from scipy import ndimage

from ..core.convolution import circular_convolve2d
from ..core.noise import add_gaussian_noise, create_motion_blur_kernel, gaussian_filter
from ..core.proximal_operators import prox_l1, prox_box, prox_iso
from ..utils.conv_utils import read_image, display_images, display_complex_output
from deblur_denoise.op_math.python_code.multiplying_matrix import DeblurDenoiseOperators

def admm_solver(b: torch.Tensor,
                t: float=18, 
                rho: float=0.001, 
                gamma: float=0.5, 
                kernel: torch.Tensor=None,
                niters: int=1000,
                **kwargs):
    """
    ADMM (Dual Douglas-Rachford) Algorithm
    """
    # TODO: Implement the algorithm
    breakpoint()
    
    imh, imw = b.shape
    # Same dimension as image
    x = torch.clamp(torch.randn((imh, imw), dtype=torch.float32), min=0, max=1)
    u = torch.clamp(torch.randn((imh, imw), dtype=torch.float32), min=0, max=1)
    w = torch.clamp(torch.randn((imh, imw), dtype=torch.float32), min=0, max=1)

    y = torch.clamp(torch.randn((3, imh, imw), dtype=torch.float32), min=0, max=1)
    z = torch.clamp(torch.randn((3, imh, imw), dtype=torch.float32), min=0, max=1)


    print(f"x: {x.shape}, u: {u.shape}, w: {w.shape}, y: {y.shape}, z: {z.shape}")
    print(f"x.device: {x.device}, u.device: {u.device}, w.device: {w.device}, y.device: {y.device}, z.device: {z.device}")
    print(f"x.dtype: {x.dtype}, u.dtype: {u.dtype}, w.dtype: {w.dtype}, y.dtype: {y.dtype}, z.dtype: {z.dtype}")
    print(f"x.requires_grad: {x.requires_grad}, u.requires_grad: {u.requires_grad}, w.requires_grad: {w.requires_grad}, y.requires_grad: {y.requires_grad}, z.requires_grad: {z.requires_grad}")
    print(f"x.grad: {x.grad}, u.grad: {u.grad}, w.grad: {w.grad}, y.grad: {y.grad}, z.grad: {z.grad}")
    print(f"x.grad_fn: {x.grad_fn}, u.grad_fn: {u.grad_fn}, w.grad_fn: {w.grad_fn}, y.grad_fn: {y.grad_fn}, z.grad_fn: {z.grad_fn}")
    print(f"x.grad_fn: {x.grad_fn}, u.grad_fn: {u.grad_fn}, w.grad_fn: {w.grad_fn}, y.grad_fn: {y.grad_fn}, z.grad_fn: {z.grad_fn}")


    dd_ops = DeblurDenoiseOperators(kernel=kernel,
                                    blurred_image=b,
                                    tprimaldr=1,
                                    s=1)


    x_prev, x_next = x, x
    u_prev, u_next = u, u
    w_prev, w_next = w, w
    y_prev, y_next = y, y
    z_prev, z_next = z, z
    prev_sol, sol_next = torch.zeros((100, 100)), torch.zeros((100, 100))

    for i in range(1, niters+1):
        # A^{T}y^{k-1}
        # to compute A we need [K, D]^{T}
        K_T_y = dd_ops.apply_KTrans(y_prev[0]) # to get K^T y , we use the y[0]
        y_12 = torch.stack([y_prev[1], y_prev[2]], dim=0) # the part of y that interacts with D 
        y_12 = y_12.permute(1, 2, 0) # because of the way the apply_DTrans is implemented (convert from (2,100,100) to (100, 100, 2))
        D_T_y = dd_ops.apply_DTrans(y_12)
        A_T_y = K_T_y + D_T_y
        
        # A^{T}z^{k-1}
        K_T_z = dd_ops.apply_KTrans(z_prev[0])
        z_12 = torch.stack([z_prev[1], z_prev[2]], dim=0)
        z_12 = z_12.permute(1, 2, 0)
        D_T_z = dd_ops.apply_DTrans(z_12)
        A_T_z = K_T_z + D_T_z

        x_next = dd_ops.invert_matrix(u_prev + A_T_y - (1/t) * (w_prev + A_T_z))

        prox_input = torch.real(rho * x_next + (1 - rho) * u_prev + (1/t) * w_prev)
        u_next = prox_box(prox_input, lambda_val=(1/t))

        # y_k line 4
        K_prime = dd_ops.apply_K(x_next) # 100 x 100
        D_prime = dd_ops.apply_D(x_next)  # 100 x 100 x 2

        K_prime = K_prime.unsqueeze(-1) # 100 x 100 x 1

        A_x = torch.cat([K_prime, D_prime], dim=-1)
        # final term
        prox_param_ = torch.real(rho * A_x.permute(2, 0, 1) + (1 - rho) * y_prev + (1/t)*z_prev)
        prox_K = b + prox_l1(prox_param_[0] - b,lambda_val=(1/t))
        prox_D = prox_iso(prox_param_[1:], lambda_val=(1/t)*gamma)
        y_next = torch.cat([prox_K.unsqueeze(0), prox_D], dim=0)

        w_next = w_prev + t*(x_next - u_next)
        z_next = z_prev + t*(A_x.permute(2, 0, 1) - y_next)

        x_prev = x_next
        u_prev = u_next
        w_prev = w_next
        y_prev = y_next
        z_prev = z_next

        # Check for NaN values in all _next tensors
        if (torch.isnan(x_next).any() or 
            torch.isnan(u_next).any() or 
            torch.isnan(w_next).any() or 
            torch.isnan(y_next).any() or 
            torch.isnan(z_next).any()):
            print(f"NaN detected at iteration {i}")
            print(f"x_next has NaN: {torch.isnan(x_next).any()}")
            print(f"u_next has NaN: {torch.isnan(u_next).any()}")
            print(f"w_next has NaN: {torch.isnan(w_next).any()}")
            print(f"y_next has NaN: {torch.isnan(y_next).any()}")
            print(f"z_next has NaN: {torch.isnan(z_next).any()}")
            break

        # Checking if the solution is converging
        if i % 100 == 0:
            print(f"iteration {i} completed")
            K_T_y = dd_ops.apply_KTrans(y_next[0]) # to get K^T y , we use the y[0]
            y_12 = torch.stack([y_next[1], y_next[2]], dim=0) # the part of y that interacts with D 
            y_12 = y_12.permute(1, 2, 0) # because of the way the apply_DTrans is implemented (convert from (2,100,100) to (100, 100, 2))
            D_T_y = dd_ops.apply_DTrans(y_12)
            A_T_y = K_T_y + D_T_y

            K_T_z = dd_ops.apply_KTrans(z_next[0])
            z_12 = torch.stack([z_next[1], z_next[2]], dim=0)
            z_12 = z_12.permute(1, 2, 0)
            D_T_z = dd_ops.apply_DTrans(z_12)
            A_T_z = K_T_z + D_T_z # TODO Nans issu

            sol_param = u_next + A_T_y - (1/t)*(w_next + A_T_z)
            sol = torch.real(dd_ops.invert_matrix(sol_param))        
            # Store current solution for next comparison
            prev_sol = sol.clone()

            # Calculate and display the difference between current solution and previous solution
            if i > 0:  # Skip the first check since we don't have a previous solution to compare
                diff = torch.norm(sol - prev_sol) / torch.norm(prev_sol)
                print(f"Relative difference at iteration {i}: {diff:.6f}")
                
                # display_images(b, sol)
                tol = 1e-4
                if diff < tol:  
                    print(f"Converged at iteration {i} with relative difference {diff:.6f}")
                    break
            

    K_T_y = dd_ops.apply_KTrans(y_next[0]) # to get K^T y , we use the y[0]
    y_12 = torch.stack([y_next[1], y_next[2]], dim=0) # the part of y that interacts with D 
    y_12 = y_12.permute(1, 2, 0) # because of the way the apply_DTrans is implemented (convert from (2,100,100) to (100, 100, 2))
    D_T_y = dd_ops.apply_DTrans(y_12)
    A_T_y = K_T_y + D_T_y

    K_T_z = dd_ops.apply_KTrans(z_next[0])
    z_12 = torch.stack([z_next[1], z_next[2]], dim=0)
    z_12 = z_12.permute(1, 2, 0)
    D_T_z = dd_ops.apply_DTrans(z_12)
    A_T_z = K_T_z + D_T_z

    sol_param = u_next + A_T_y - (1/t)*(w_next + A_T_z)
    sol = torch.real(dd_ops.invert_matrix(sol_param))

    display_images(b, sol)
    

def admm_solver_test():
    """
    Test function for ADMM solver during development
    """
    # You can change these parameters for testing
    IMAGE_PATH = '/Users/rahulpadmanabhan/Code/ws3/convex_optimization/.develop/manWithHat.tiff'
    IMAGE_SHAPE = (100, 100)  # You can adjust this
    T = 18.0  # Step size
    RHO = 0.001  # Relaxation parameter
    GAMMA = 0.5  # Regularization parameter
    
    print("Loading image...")
    image = read_image(IMAGE_PATH, shape=IMAGE_SHAPE)
    print(f"Image shape: {image.shape}")
    
    print("\nRunning ADMM solver...")
    print(f"Parameters: t={T}, rho={RHO}, gamma={GAMMA}")
    
    # Create a simple blur kernel for testing
    kernel = torch.tensor([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]], dtype=torch.float32) / 16.0
    
    # Blur the image
    blurred = circular_convolve2d(image, kernel)
    
    # Run the solver
    result = admm_solver(b=blurred.squeeze(0).clone(), t=T, rho=RHO, gamma=GAMMA)
    
    # Display results
    display_images(blurred, result, 
                  title1="Blurred Image", 
                  title2="Deblurred Image")
    
    # Calculate metrics
    mse = torch.mean((result - image) ** 2)
    max_pixel = torch.max(image)
    psnr = 10 * torch.log10((max_pixel ** 2) / mse)
    
    print(f"\nMetrics:")
    print(f"MSE: {mse.item():.6f}")
    print(f"PSNR: {psnr.item():.2f} dB")

if __name__ == "__main__":
    admm_solver_test()


