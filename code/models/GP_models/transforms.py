import torch
import numpy as np

from abc import ABC, abstractmethod

class Transform(ABC):
    """
    Abstract base class for the input and output transforms. Implements the forward
    and reverse transform methods.
    """

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    @abstractmethod
    def reverse(self, X):
        raise NotImplementedError
    
class TensorTransform(Transform):
    """
    Transforms an input `numpy.ndarray` to `torch.tensor` and vice-versa
    """

    def __init__(self, target_type=torch.float64):
        self.target_type = target_type

    def forward(self, X):
        return torch.from_numpy(X).type(self.target_type)

    def reverse(self, X):
        return X.detach().cpu().numpy().astype(np.float64)