import torch

def apply_periodic_conv_2d(x, eig_val_arr):
    """
    For a given "unblurred" image x and the eigenvalue array for
    the blurring kernel, computes the "blurred image" (e.g. Kx and Dx in the
    paper)
    
    Parameters:
    -----------
    x : torch.Tensor
        m x n tensor representing the image
    eig_val_arr : torch.Tensor
        m x n representing the eigenvalues of the 2D DFT
        of the convolution kernel
    
    Returns:
    --------
    out : torch.Tensor
        m x n the "blurred" image (i.e. Kx)
    
    Notes:
    ------
    Observe that K = Q^H eig_val_arr Q where Q is essentially the
    discrete fourier transform and Q^H is the inverse fourier transform. This
    performs Kx which reduces to ifft(eig_val_arr.*fft(x)).
    """
    return torch.fft.ifft2(eig_val_arr * torch.fft.fft2(x)) 