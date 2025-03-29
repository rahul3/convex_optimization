import torch
from .apply_periodic_conv_2d import apply_periodic_conv_2d
from .eig_vals_for_periodic_conv_op import eig_vals_for_periodic_conv_op

class DeblurDenoiseOperators:
    """
    Class for constructing matrices and operations required for image deblurring and denoising
    """
    def __init__(self, kernel, blurred_image, tprimaldr, s=None, device='cpu'):
        """
        Initialize the operators for deblurring and denoising
        
        Parameters:
        -----------
        kernel : torch.Tensor
            Convolution kernel for blurring
        blurred_image : torch.Tensor
            The blurred input image
        tprimaldr : float
            Step size parameter
        s : float, optional
            Additional step size parameter (if None, uses tprimaldr)
        """
        self.kernel = kernel # k x k
        self.b = blurred_image # 100 x 100
        self.t = tprimaldr  # scalar
        self.s = s if s is not None else tprimaldr # scalar
        
        # Get image dimensions
        self.num_rows, self.num_cols = blurred_image.shape # 100 x 100
        
        # Set device for all tensors
        self.device = device
        
        # Move input tensors to the specified device
        if self.kernel.device != self.device:
            self.kernel = self.kernel.to(self.device)
        
        if self.b.device != self.device:
            self.b = self.b.to(self.device)
        # Compute eigenvalue arrays
        self.setup_operators()
        
    def setup_operators(self):
        """Initialize all operators needed for deblurring and denoising"""
        # Compute eigenvalue arrays for K and D1, D2
        self.eig_arry_K = eig_vals_for_periodic_conv_op(self.kernel, self.num_rows, self.num_cols) # 100 x 100
        
        # Create finite difference kernels
        # d1_kernel = torch.tensor([[-1], [1]], device=self.b.device) # 2 x 1
        d1_kernel = torch.tensor([[-1, 1]], device=self.b.device) # 1 x 2
        d2_kernel = torch.tensor([[-1, 1]], device=self.b.device) # 1 x 2

        self.eig_arry_D1 = eig_vals_for_periodic_conv_op(d1_kernel, self.num_rows, self.num_cols) # 100 x 101
        self.eig_arry_D2 = eig_vals_for_periodic_conv_op(d2_kernel, self.num_rows, self.num_cols) # 100 x 101
        
        # Compute conjugate eigenvalue arrays
        self.eig_arry_KTrans = torch.conj(self.eig_arry_K)
        self.eig_arry_D1Trans = torch.conj(self.eig_arry_D1)
        self.eig_arry_D2Trans = torch.conj(self.eig_arry_D2)
        
        # Compute eigenvalues for matrix inversion
        self.eig_vals_mat = (
            torch.ones((self.num_rows, self.num_cols), device=self.b.device) + 
            self.t * self.t * self.eig_arry_KTrans * self.eig_arry_K + 
            self.t * self.t * self.eig_arry_D1Trans * self.eig_arry_D1 +
            self.t * self.s * self.eig_arry_D2Trans * self.eig_arry_D2
        )
    
    def apply_K(self, x):
        """Apply K operator"""
        return apply_periodic_conv_2d(x, self.eig_arry_K)
    
    def apply_D1(self, x):
        """Apply D1 operator (vertical gradient)"""
        return apply_periodic_conv_2d(x, self.eig_arry_D1)
    
    def apply_D2(self, x):
        """Apply D2 operator (horizontal gradient)"""
        return apply_periodic_conv_2d(x, self.eig_arry_D2)
    
    def apply_KTrans(self, x):
        """Apply K^T operator"""
        return apply_periodic_conv_2d(x, self.eig_arry_KTrans)
    
    def apply_D1Trans(self, x):
        """Apply D1^T operator"""
        return apply_periodic_conv_2d(x, self.eig_arry_D1Trans)
    
    def apply_D2Trans(self, x):
        """Apply D2^T operator"""
        return apply_periodic_conv_2d(x, self.eig_arry_D2Trans)
    
    def apply_D(self, x):
        """Apply D operator (combines D1 and D2)"""
        return torch.stack([self.apply_D1(x), self.apply_D2(x)], dim=2)
    
    def apply_DTrans(self, y):
        """Apply D^T operator"""
        return self.apply_D1Trans(y[:, :, 0]) + self.apply_D2Trans(y[:, :, 1])
    
    def apply_Mat(self, x):
        """Apply (I + K^TK + D^TD)x"""
        return x + self.apply_KTrans(self.apply_K(x)) + self.apply_DTrans(self.apply_D(x))
    
    def invert_matrix(self, x):
        """Apply (I + K^TK + D^TD)^(-1)x"""
        return torch.fft.ifft2(torch.fft.fft2(x) / self.eig_vals_mat) 