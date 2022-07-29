import torch
import numpy


class Settings:
    def __init__(self):
        pass

    @property
    def torch_int(self):
        return torch.int32

    @property
    def numpy_int(self):
        return numpy.int32

    @property
    def device(self):
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @property
    def torch_float(self):
        return torch.float32

    @property
    def numpy_float(self):
        return numpy.float32

    @property
    def jitter(self):
        return 1e-5


settings = Settings()
