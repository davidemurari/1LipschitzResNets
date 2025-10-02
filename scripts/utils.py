import torch

def power_method(linear, u_init=None, k=1):
    if u_init is None or u_init.shape != linear.weight.shape[1:]:
        u_init = torch.randn(linear.weight.shape[1],device=linear.weight.device)
    u_init = u_init.to(linear.weight.device)
    u = u_init/u_init.norm()
    with torch.no_grad():
        for _ in range(k):
            u = u @ linear.weight.T @ linear.weight
            norm_ATAu = u.norm()
            u /= norm_ATAu

    # Spectral norm = √λ_max
    return norm_ATAu.sqrt().item(), u