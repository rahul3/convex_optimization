import torch

def prox_l1(x: torch.Tensor, lambda_val: float) -> torch.Tensor:
    """
    Proximal operator of the L1 norm 
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - lambda_val, min=0)