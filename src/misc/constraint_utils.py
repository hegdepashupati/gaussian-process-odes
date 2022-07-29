import torch
import torch.nn.functional as F


def softplus(x):
    lower = 1e-12
    return F.softplus(x) + lower


def invsoftplus(x):
    lower = 1e-12
    xs = torch.max(x - lower, torch.tensor(torch.finfo(x.dtype).eps).to(x))
    return xs + torch.log(-torch.expm1(-xs))
