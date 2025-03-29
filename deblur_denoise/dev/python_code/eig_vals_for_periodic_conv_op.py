import torch
import torch.nn.functional as F

def eig_vals_for_periodic_conv_op(filter_kernel, num_rows, num_cols):
    """
    Computes the eigenvalues of the 2D DFT of the convolution kernel.
    
    Parameters:
    -----------
    filter_kernel : torch.Tensor
        Correlation kernel (e.g. for Gaussian filter)
    num_rows : int
        Number of rows in the blurred image (b)
    num_cols : int
        Number of columns in the blurred image (b)
    
    Returns:
    --------
    eig_val_array : torch.Tensor
        num_rows x num_cols matrix containing the eigenvalues of the convolution kernel
    
    Notes:
    ------
    This is equivalent to the MATLAB's eigValsForPeriodicConvOp function.
    """
    # Construct the impulse: customary to put this in the upper left hand corner pixel
    a = torch.zeros((num_rows, num_cols), device=filter_kernel.device)
    a[0, 0] = 1
    
    # Simulate MATLAB's imfilter with 'circular' for periodic boundary conditions
    # First, pad the filter to match dimensions of the input
    if len(filter_kernel.shape) == 2:  
        # Add batch and channel dimensions for conv2d
        filter_kernel = filter_kernel.unsqueeze(0).unsqueeze(0)
    
    # Circular padding requires a custom implementation
    # We need to pad the input before convolution to ensure wrap-around behavior
    pad_h = filter_kernel.shape[2] // 2
    pad_w = filter_kernel.shape[3] // 2
    
    # Create padded input with circular padding
    a_padded = F.pad(a.unsqueeze(0).unsqueeze(0), 
                     (pad_w, pad_w, pad_h, pad_h), 
                     mode='circular')
    
    filter_kernel = filter_kernel.type(torch.float32)
    a_padded = a_padded.type(torch.float32)
    
    # Apply convolution
    ra_padded = F.conv2d(a_padded, filter_kernel, padding=0).squeeze()
    
    # Crop to original size
    ra = ra_padded[:num_rows, :num_cols]
    
    # Fourier transform of the impulse response
    ra_hat = torch.fft.fft2(ra)
    
    return ra_hat 