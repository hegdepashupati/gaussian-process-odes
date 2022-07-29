from src.misc import transforms
from src.misc.settings import settings

import torch


class Param(torch.nn.Module):
    """
    A class to handle constrained --> unconstrained optimization using variable transformations.
    Similar to Parameter class in GPflow : https://github.com/GPflow/GPflow/blob/develop/gpflow/base.py


    """

    def __init__(self, value, transform=transforms.Identity(), name='var'):
        super(Param, self).__init__()
        self.transform = transform
        self.name = name
        value_ = self.transform.backward(value)
        self.optvar = torch.nn.Parameter(torch.tensor(data=value_,
                                                      dtype=settings.torch_float,
                                                      device=settings.device)).to(settings.device)

    def __call__(self):
        return self.transform.forward_tensor(self.optvar)

    def __repr__(self):
        return '{} parameter with {}'.format(self.name, self.transform.__str__())
