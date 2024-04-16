from enum import Enum, auto

import clearml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST


class Dataset(Enum):
    MNIST = auto()
    FASHION_MNIST = auto()

def load_data(transform, batch_size, dataset: Dataset):
    if (dataset == Dataset.MNIST):
        return load_mnist(transform, batch_size)
    elif (dataset == Dataset.FASHION_MNIST):
        return load_fashion_mnist(transform, batch_size)
    else:
        raise ValueError("No valid dataset was provided")

def load_mnist(transform, batch_size):
    data_path = '/tmp/data/mnist'
    try:
        data = clearml.datasets.Dataset.get(dataset_name="MNIST", dataset_version="1.0.0")
        data_path = data.get_local_copy()
        data_train = MNIST(data_path, train=True, download=False, transform=transform)
        data_test = MNIST(data_path, train=False, download=False, transform=transform)
    except Exception:
        data_train = MNIST(data_path, train=True, download=True, transform=transform)
        data_test = MNIST(data_path, train=False, download=True, transform=transform)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader

def load_fashion_mnist(transform, batch_size):
    data_path = '/tmp/data/fashion_mnist'
    try:
        data = clearml.datasets.Dataset.get(dataset_name="FashionMNIST", dataset_version="1.0.0")
        data_path = data.get_local_copy()
        data_train = FashionMNIST(data_path, train=True, download=False, transform=transform)
        data_test = FashionMNIST(data_path, train=False, download=False, transform=transform)
    except Exception:
        data_train = FashionMNIST(data_path, train=True, download=True, transform=transform)
        data_test = FashionMNIST(data_path, train=False, download=True, transform=transform)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader

def loss(prediction, targets, **kwargs):
    mem_rec = prediction[1]
    num_steps = len(mem_rec)
    loss_val = torch.zeros(1, device=mem_rec[0].device)
    for step in range(num_steps):
        loss_val += nn.CrossEntropyLoss()(mem_rec[step], targets)

    return loss_val[0] / num_steps
