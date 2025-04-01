# %%
import os
import sys

import torch
import numpy as np
from scipy import ndimage
FILE_PATH = "/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/chambolle_pock.ipynb"
script_dir = os.path.dirname(os.path.abspath(""))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, os.pardir))
sys.path.append(os.path.join(script_dir, os.pardir, os.pardir))


# %%
from deblur_denoise.core.convolution import circular_convolve2d
from deblur_denoise.core.noise import add_gaussian_noise
from deblur_denoise.core.proximal_operators import prox_l1, prox_box, prox_iso
from deblur_denoise.core.noise import create_motion_blur_kernel

from deblur_denoise.utils.conv_utils import read_image, display_images, display_complex_output

from deblur_denoise.dev.python_code.multiplying_matrix import DeblurDenoiseOperators


# %%
IMAGE_PATH = '/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/other_images/grok_generated_image.jpg'
IMAGE_PATH = '/Users/rahulpadmanabhan/Code/ws3/convex_optimization/deblur_denoise/utils/other_images/manWithHat.tiff'


# %%
image = read_image(IMAGE_PATH, shape=(200, 200))
image.shape

# %%
import numpy as np
from scipy import ndimage

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

# %%
# gaussian_blurred = circular_convolve2d(image, gaussian_kernel,)
gaussian_blurred = circular_convolve2d(image, kernel)
gaussian_blurred = gaussian_blurred.type(torch.float32)

# %%
# display_images(image, gaussian_blurred)

# %%
b = gaussian_blurred.squeeze() # blurred image
t = 18 # step-size
s = 0.1 # step-size 2
gamma = 0.1

# %%
imh, imw = b.shape
# Same dimension as image
x = torch.rand((imh, imw), dtype=torch.float32)
z = torch.rand((imh, imw), dtype=torch.float32)

y = torch.rand((3, imh, imw), dtype=torch.float32)

# %%
x_prev, x_next = x, x
y_prev, y_next = y, y
z_prev, z_next = z, z

# %%
dd_ops = DeblurDenoiseOperators(kernel=kernel,
                                blurred_image=b,
                                tprimaldr=t,
                                s=s)

# %%
t_vals = np.linspace(3, 10, 3)
s_vals = np.linspace(3, 10, 3)
gamma_vals = np.linspace(0, 1, 3)


for i in range(100):
    Az_prev = torch.cat([dd_ops.apply_K(z_prev).unsqueeze(-1), dd_ops.apply_D(z_prev)], -1)
    g_prox_param = torch.real(y_prev + s*Az_prev.permute(2,0,1))
    prox_g_1 = g_prox_param[0] - s * b - s * prox_box((g_prox_param[0]/s) - b, lambda_val=1/s)
    prox_g_conj = g_prox_param[1:] - s*prox_iso((y[1:])/s, lambda_val=gamma/s)
    y_next = torch.cat([prox_g_1.unsqueeze(0), prox_g_conj], 0)
    prox_f_param_dtrans = dd_ops.apply_DTrans(y[1:].permute(1, 2, 0)) # hdim x wdim
    prox_f_param = x_prev - t * (dd_ops.apply_KTrans(y[0]) + prox_f_param_dtrans)
    x_next = prox_box(torch.real(prox_f_param), lambda_val=t)
    z_next = 2*x_next - x_prev


solution = x_next
    
    
# %%
display_images(b, solution)

# %%
# display_images(image, b)

# %%
y[1:].permute(2, 1, 0).shape

# %%



