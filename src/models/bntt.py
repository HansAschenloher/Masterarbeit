from typing import cast, List, Union, Callable, Tuple, Optional, Any

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import utils

configurations = {

    'BNTT': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 'A'],
    ],

    'VGG7': [
        [64, 'M'],
        [128, 'M'],
        [256, 'M'],
        [512, 'M']
    ],

    'VGG9': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M']
    ],

    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, num_steps=25, beta=0.95,
                affine_flag=True) -> nn.Module:
    layers: List[nn.Module] = []
    in_channels = 3
    for layer in cfg:
        for value in layer:
            if value == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            if value == "A":
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                value = cast(int, value)
                conv2d = nn.Conv2d(in_channels, value, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               BatchNorm2dThroughTime(num_steps=num_steps, num_features=value, eps=1e-4, momentum=0.1,
                                                      affine=affine_flag),
                               snn.Leaky(beta=beta, init_hidden=True)]
                else:
                    layers += [conv2d, snn.Leaky(beta=beta, init_hidden=True)]
                in_channels = value
    return TemporalSequential(num_steps, *layers)


class TemporalLayer(nn.Module):
    def __init__(self, module, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.moduleList = nn.ModuleList([module for i in range(self.num_steps)])

    def __call__(self, x: torch.Tensor, t: int, **kwargs):
        return self.forward(x, t)

    def forward(self, x: torch.Tensor, t: int):
        return self.moduleList[t].forward(x)


class TemporalSequential(nn.Sequential):
    def __init__(self, num_steps, *args):
        super().__init__(*args)

    def forward(self, x: torch.Tensor, t: int):
        for module in self:
            if (isinstance(module, TemporalLayer)):
                x = module(x, t)
            else:
                x = module(x)
        return x


class BatchNorm2dThroughTime(TemporalLayer):
    def __init__(self,
                 num_steps,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device=None,
                 dtype=None):
        module = nn.BatchNorm2d(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            device,
            dtype
        )
        super().__init__(module, num_steps)


class BatchNorm1dThroughTime(TemporalLayer):
    def __init__(self,
                 num_steps,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device=None,
                 dtype=None):
        module = nn.BatchNorm1d(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            device,
            dtype
        )
        super().__init__(module, num_steps)


class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 10, num_steps=64):
        super().__init__()
        self.num_steps = num_steps
        self.beta = 0.9
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = TemporalSequential(num_steps,
                                             nn.Linear(256 * 4 * 4, 1024),
                                             BatchNorm1dThroughTime(self.num_steps, 1024, eps=1e-4, momentum=0.1,
                                                                    affine=True),
                                             snn.Leaky(beta=self.beta, init_hidden=True),
                                             nn.Linear(1024, num_classes),
                                             snn.Leaky(beta=self.beta, init_hidden=True, output=True)
                                             )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        utils.reset(self.features)
        utils.reset(self.classifier)
        mem_rec1 = [];
        spk_rec1 = [];
        for t in range(self.num_steps):
            spk_x = self.features(x[:, t], t)
            # spk_x = self.avgpool(spk_x)
            spk_x = torch.flatten(spk_x, 1)
            spk_x, mem_x = self.classifier(spk_x, t)
            mem_rec1.append(mem_x)
            spk_rec1.append(spk_x)

        return torch.stack(spk_rec1, dim=0), torch.stack(mem_rec1, dim=0)


def _vgg(cfg: str, batch_norm: bool, progress: bool, num_steps, num_cls, **kwargs: Any) -> VGG:
    model = VGG(make_layers(configurations[cfg], batch_norm=batch_norm, num_steps=num_steps), num_steps=num_steps, num_classes=num_cls,
                **kwargs)
    return model


def vgg7(num_steps=64) -> VGG:
    return _vgg("VGG7", False, False, num_steps=num_steps)


def vgg9(num_steps=64) -> VGG:
    return _vgg("VGG9", False, False, num_steps=num_steps)


def vgg11(num_steps=64, batch_norm=False, num_cls=100) -> VGG:
    return _vgg("VGG11", batch_norm, False, num_steps=num_steps, num_cls=num_cls)


def bntt(num_steps=25, num_cls=100) -> VGG:
    return _vgg('BNTT', True, False, num_steps, num_cls)
