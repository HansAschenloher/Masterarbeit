import torch


class DirectCoding():
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def __call__(self, x):
        return torch.stack([x] * self.num_steps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_steps={self.num_steps})"
