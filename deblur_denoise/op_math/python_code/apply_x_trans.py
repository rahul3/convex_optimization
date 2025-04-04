import torch
import torch.nn.functional as F

def apply_x_trans(w, x, kernel_size):
    """
    Compute X^T w where X is the convolution operation
    
    Parameters:
    -----------
    w : torch.Tensor
        Image tensor of size (kernel_size x kernel_size)
    x : torch.Tensor
        Image tensor of size (kernel_size x kernel_size)
    kernel_size : int
        Length of one side of the kernel
        
    Returns:
    --------
    xtw : torch.Tensor
        Result of X^T w operation
        
    Notes:
    ------
    (X^Tw)_(ij) = < w, conv2(x, impulse at the ij entry, 'same') >
    """
    im = torch.zeros((kernel_size, kernel_size), device=w.device)
    xtw = torch.zeros_like(im)
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            shift_im = torch.zeros_like(im)
            shift_im[i, j] = 1
            
            # Use padding='same' to match MATLAB's conv2(..., 'same')
            # Need to convert shift_im to have proper dimensions for conv2d
            shift_im = shift_im.unsqueeze(0).unsqueeze(0)
            x_reshaped = x.unsqueeze(0).unsqueeze(0)
            
            # 'same' padding in PyTorch requires explicit padding calculation
            padding = (shift_im.shape[2] // 2, shift_im.shape[3] // 2)
            conv_result = F.conv2d(x_reshaped, shift_im, padding=padding).squeeze()
            
            # Sum product of w and conv_result (equivalent to sum(sum(w.*conv_result)))
            xtw[i, j] = torch.sum(w * conv_result)
    
    return xtw 