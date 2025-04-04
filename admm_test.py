
import torch
import numpy as np
from scipy import ndimage

from deblur_denoise.core.convolution import circular_convolve2d
from deblur_denoise.core.noise import add_gaussian_noise
from deblur_denoise.core.proximal_operators import prox_l1, prox_box, prox_iso
from deblur_denoise.core.noise import create_motion_blur_kernel

from deblur_denoise.utils.conv_utils import read_image, display_images, display_complex_output

from deblur_denoise.op_math.python_code.multiplying_matrix import DeblurDenoiseOperators

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Best parameters: t=10, rho=0.0001, gamma=0.1

# IMAGE_PATH = '/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/other_images/grok_generated_image.jpg'
IMAGE_PATH = '/Users/rahulpadmanabhan/Code/ws3/convex_optimization/.develop/manWithHat.tiff'
# IMAGE_PATH = '/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/other_images/scientist.jpg'

image = read_image(IMAGE_PATH, shape=(100, 100))
image.shape


# Create different blur kernels
gaussian_kernel = torch.tensor([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], dtype=torch.float32)



def gaussian_filter(size, sigma):
    # size: filter size (e.g., [3, 3]), sigma: standard deviation
    x, y = np.meshgrid(np.linspace(-(size[0]//2), size[0]//2, size[0]),
                       np.linspace(-(size[1]//2), size[1]//2, size[1]))
    d = x**2 + y**2
    H = np.exp(-d / (2 * sigma**2))
    return H / H.sum()  # Normalize to sum to 1

# Example usage
kernel = gaussian_filter((3, 3), 5)
kernel = torch.from_numpy(kernel)
kernel = kernel.type(torch.float32)



# gaussian_blurred = circular_convolve2d(image, gaussian_kernel,)
gaussian_blurred = circular_convolve2d(image, kernel)
gaussian_blurred = gaussian_blurred.type(torch.float32)

motion_kernel = create_motion_blur_kernel(size=3, angle=45)
motion_blurred = circular_convolve2d(image, motion_kernel)


# display_images(image, gaussian_blurred)



b = gaussian_blurred.squeeze() # blurred image
b = motion_blurred.squeeze()
# Define parameter grid for grid search
t_values = np.linspace(0.01, 10, 10)  # step-size options
rho_values = np.linspace(0.00001, 0.01, 10)  # relaxation parameter options
gamma_values = np.linspace(0.1, 1.5, 10)  # gamma options

# Initialize variables to track best parameters and performance
best_t = None
best_rho = None
best_gamma = None
best_performance = float('inf')  # Lower is better for our objective function

# We'll set initial values to use until we find the best through grid search
t = 18  # default step-size
rho = 0.001  # default relaxation parameter
gamma = 0.5  # default gamma

# Grid search will be performed in the optimization loop
# The performance metric will be the final objective function value




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
                                tprimaldr=t,
                                s=t)


dd_ops = DeblurDenoiseOperators(kernel=kernel,
                            blurred_image=b,
                            tprimaldr=t,
                            s=t)

x_prev, x_next = x, x
u_prev, u_next = u, u
w_prev, w_next = w, w
y_prev, y_next = y, y
z_prev, z_next = z, z
prev_sol, sol_next = torch.zeros((100, 100)), torch.zeros((100, 100))

# Best parameters: t=3, rho=1e-05, gamma=0.1
# best performance config
# t_values = [3]
# rho_values = [0.01]
# gamma_values = [0.1]
# # New best parameters found: t=0.5, rho=1e-05, gamma=0.1
# t_values = [0.5]
# rho_values = [1e-05]
# gamma_values = [0.1]
# Best parameters: t=0.01, rho=0.0033400000000000005, gamma=0.1
t_values = [0.01]
rho_values = [0.0033400000000000005]
gamma_values = [0.1]

t_values = [18]
rho_values = [0.01]
gamma_values = [0.5]

for t in t_values:

    for rho in rho_values:
        for gamma in gamma_values:
            for i in range(1, 1000):
                # A^{T}y^{k-1}
                # to compute A we need [K, D]^{T}
                K_T_y = torch.real(dd_ops.apply_KTrans(y_prev[0])) # to get K^T y , we use the y[0]
                y_12 = torch.stack([y_prev[1], y_prev[2]], dim=0) # the part of y that interacts with D 
                y_12 = y_12.permute(1, 2, 0) # because of the way the apply_DTrans is implemented (convert from (2,100,100) to (100, 100, 2))
                D_T_y = torch.real(dd_ops.apply_DTrans(y_12))
                A_T_y = K_T_y + D_T_y
                
                # A^{T}z^{k-1}
                K_T_z = torch.real(dd_ops.apply_KTrans(z_prev[0]))
                z_12 = torch.stack([z_prev[1], z_prev[2]], dim=0)
                z_12 = z_12.permute(1, 2, 0)
                D_T_z = torch.real(dd_ops.apply_DTrans(z_12))
                A_T_z = K_T_z + D_T_z

                x_next = torch.real(dd_ops.invert_matrix(u_prev + A_T_y - (1/t) * (w_prev + A_T_z)))

                prox_input = rho * x_next + (1-rho)*u_prev + (1/t)*w_prev
                u_next = prox_box(prox_input, lambda_val=(1/t))

                # y_k line 4
                K_prime = torch.real(dd_ops.apply_K(x_next)) # 100 x 100
                D_prime = torch.real(dd_ops.apply_D(x_next) ) # 100 x 100 x 2

                K_prime = K_prime.unsqueeze(-1) # 100 x 100 x 1

                A_x = torch.cat([K_prime, D_prime], dim=-1)
                # final term
                prox_param_ = rho * A_x.permute(2, 1, 0) + (1 - rho) * y_prev + (1/t)*z_prev
                prox_K = b + prox_l1(prox_param_[0] - b,lambda_val=(1/t))
                prox_D = prox_iso(prox_param_[1:], lambda_val=(1/t)*gamma)
                y_next = torch.cat([prox_K.unsqueeze(0), prox_D], dim=0)

                w_next = w_prev + t*(x_next - u_next)
                z_next = z_prev + t*(A_x.permute(2, 1, 0) - y_next)

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

                if i%100 == 0:
                    print(f"iteration {i} completed")


                # Checking if the solution is converging
                if i > 1:
                    K_T_y = torch.real(dd_ops.apply_KTrans(y_next[0])) # to get K^T y , we use the y[0]
                    y_12 = torch.stack([y_next[1], y_next[2]], dim=0) # the part of y that interacts with D 
                    y_12 = y_12.permute(1, 2, 0) # because of the way the apply_DTrans is implemented (convert from (2,100,100) to (100, 100, 2))
                    D_T_y = torch.real(dd_ops.apply_DTrans(y_12))
                    A_T_y = K_T_y + D_T_y

                    K_T_z = torch.real(dd_ops.apply_KTrans(z_next[0]))
                    z_12 = torch.stack([z_next[1], z_next[2]], dim=0)
                    z_12 = z_12.permute(1, 2, 0)
                    D_T_z = torch.real(dd_ops.apply_DTrans(z_12))
                    A_T_z = K_T_z + D_T_z 

                    sol_param = u_next + A_T_y - (1/t)*(w_next + A_T_z)
                    sol = torch.real(dd_ops.invert_matrix(sol_param))        

                    # Calculate and display the difference between current solution and previous solution
                    if i > 1:  # Skip the first check since we don't have a previous solution to compare
                        diff = torch.norm(sol - prev_sol) / torch.norm(prev_sol)
                        if i%100 == 0:
                            print(f"Relative difference at iteration {i}: {diff:.6f}")
                        
                        # display_images(b, sol)
                        tol = 1e-4
                        if diff < tol:  
                            print(f"Converged at iteration {i} with relative difference {diff:.6f}")
                            break

                    # Store current solution for next comparison
                    prev_sol = sol.clone()
                
                    unblurred_image = read_image(IMAGE_PATH, shape=(100, 100))
                    unblurred_image = unblurred_image.type(torch.float32)
                    unblurred_image = unblurred_image.to(device)
                    unblurred_image = unblurred_image.squeeze()

                    if sol.shape == unblurred_image.shape:
                        # Print current parameter performance
                        # print(f"Parameters: t={t}, rho={rho}, gamma={gamma}")
                        # Calculate metrics for current parameters
                        mse = torch.mean((sol - unblurred_image) ** 2)
                        max_pixel = torch.max(unblurred_image)
                        psnr = 10 * torch.log10((max_pixel ** 2) / mse)
                        
                        # Update best parameters if current performance is better
                        current_performance = mse.item()  # Using MSE as our performance metric
                        if current_performance < best_performance:
                            best_performance = current_performance
                            best_t = t
                            best_rho = rho
                            best_gamma = gamma
                            print(f"New best parameters found: t={best_t}, rho={best_rho}, gamma={best_gamma}")
                            print(f"New best MSE: {best_performance:.6f}")
                    else:
                        print(f"Shape mismatch: solution shape {sol.shape} vs original image shape {unblurred_image.shape}")
            
print(f"Best parameters: t={best_t}, rho={best_rho}, gamma={best_gamma}")

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

image = read_image(IMAGE_PATH, shape=(100, 100))
image.shape
# Calculate the distance between the solution and the original image
if image.shape == sol.shape:
    # Compute the mean squared error (MSE)
    mse = torch.mean((sol - image) ** 2)
    # Compute the peak signal-to-noise ratio (PSNR)
    max_pixel = torch.max(image)
    psnr = 10 * torch.log10((max_pixel ** 2) / mse)
    # Compute the structural similarity index (SSIM) if available
    
    print(f"Mean Squared Error (MSE): {mse.item():.6f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr.item():.2f} dB")
    
    # Compute L1 distance (Mean Absolute Error)
    l1_distance = torch.mean(torch.abs(sol - image))
    print(f"L1 Distance (MAE): {l1_distance.item():.6f}")
    
    # Compute L2 distance (Root Mean Squared Error)
    l2_distance = torch.sqrt(mse)
    print(f"L2 Distance (RMSE): {l2_distance.item():.6f}")
else:
    print(f"Shape mismatch: solution shape {sol.shape} vs original image shape {image.shape}")

print(f"Best parameters: t={best_t}, rho={best_rho}, gamma={best_gamma}")

    
















