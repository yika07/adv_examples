import torch


def stable_svd(x: torch.Tensor):
    # torch.svd may have convergence issues for GPU and CPU.
    # Taken from : https://github.com/pytorch/pytorch/issues/28293
    try:
        u, s, v = torch.svd(x)
    except:
        u, s, v = torch.svd(x + 1e-4 * x.mean() * torch.rand(x.shape, device=x.device))
    return u, s, v
