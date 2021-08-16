import torch
from torch import distributions


def get_zdist(dist_name, dim, device=None):
    # Get distribution
    if dist_name == 'uniform':
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dist_name == 'gauss':
        mu = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
        zdist = distributions.Normal(mu, scale)
    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist
