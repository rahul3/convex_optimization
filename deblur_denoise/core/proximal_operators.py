import torch
import numpy as np

def prox_l1(x: torch.Tensor | np.ndarray, lambda_val: float) -> torch.Tensor | np.ndarray:
    """
    Proximal operator of the L1 norm 
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - lambda_val, min=0)

def prox_l2(x: torch.Tensor | np.ndarray, lambda_val: float) -> torch.Tensor | np.ndarray:
    """
    Proximal operator of the L2 norm
    """
    return (1 - lambda_val / torch.norm(x)) * x if torch.norm(x) > lambda_val else torch.zeros_like(x) 

def prox_l2_squared(x: torch.Tensor | np.ndarray, lambda_val: float) -> torch.Tensor | np.ndarray:
    """
    Proximal operator of the squared L2 norm
    """
    return x / (2 * lambda_val + 1)

def prox_iso(x: torch.Tensor, lambda_val: float) -> torch.Tensor:
    """
    Proximal operator of the iso norm.

    x = [x_1, x_2] where x_i are N x N matrices
    """

    # Extract x_1 and x_2 from the input tensor
    x_1 = x[0, :, :]
    x_2 = x[1, :, :]
    
    # Compute the squared sum of x_1 and x_2 element-wise
    sum_squared = x_1**2 + x_2**2
    
    # Compute the element-wise square root of the sum of squares
    norm = torch.sqrt(sum_squared)
    
    # Initialize the outputs with the same shape as x_1 and x_2
    o_1 = torch.zeros_like(x_1)
    o_2 = torch.zeros_like(x_2)

    
    # Apply the conditions
    condition = norm > lambda_val
    o_1[condition] = (1 - lambda_val / norm[condition]) * x_1[condition]
    o_2[condition] = (1 - lambda_val / norm[condition]) * x_2[condition]
    
    # Return the result tensor
    return torch.stack((o_1, o_2))

def prox_box(x: torch.Tensor, lambda_val: float) -> torch.Tensor:
    """
    Proximal operator of the indicator function onto the box 0 <= x <= 1, 
    or in other words, projection onto 0 <= x <= 1.
    """
    return torch.minimum(torch.ones(x.size()), torch.maximum(torch.zeros(x.size()), x))

# tests
# t1
x = torch.tensor([[[0, 3], [1, 2]],[[2, 1], [1, 1/2]]])
l = 2

print(prox_iso(x, 2))

# test for prox_box
x = torch.tensor([[0.5,1.1], [-0.3, 0.2], [0.3,0.4]])
print(prox_box(x, 2))