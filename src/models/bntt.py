from typing import cast, List, Union, Tuple, Any

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch import utils

configurations = {
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
        [64, 'A'],
        [128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512]
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
                layers += [nn.MaxPool2d(kernel_size=2)]
            elif value == "A":
                layers += [nn.AvgPool2d(kernel_size=2)]
            else:
                value = cast(int, value)
                conv2d = nn.Conv2d(in_channels, value, kernel_size=3, padding=1, stride=1, bias=False)
                if batch_norm:
                    layers += [conv2d,
                               BatchNorm2dThroughTime(num_steps=num_steps, num_features=value, eps=1e-4, momentum=0.1,
                                                      affine=affine_flag),
                               snn.Leaky(beta=beta, init_hidden=True, spike_grad=Surrogate_BP_Function.apply),
                               nn.Dropout(0.2)
                               ]
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


class Surrogate_BP_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 10, num_steps=64, beta=0.95):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.beta = beta
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        classifier_hidden_layer = 4096
        p = 0.2
        self.classifier = TemporalSequential(num_steps,
                                             nn.Linear(512, classifier_hidden_layer, bias=False),
                                             BatchNorm1dThroughTime(self.num_steps, classifier_hidden_layer, eps=1e-4,
                                                                    momentum=0.1,
                                                                    affine=True),
                                             snn.Leaky(beta=self.beta, init_hidden=True,
                                                       spike_grad=Surrogate_BP_Function.apply),
                                             nn.Dropout(p=p),
                                             nn.Linear(classifier_hidden_layer, num_classes, bias=False),
                                             )
        # Init weights
        for m in self.modules():
            if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)):
                m.bias = None
            elif (isinstance(m, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        utils.reset(self.features)
        utils.reset(self.classifier)
        mem_fc2 = torch.zeros((x.shape[0], self.num_classes), device=torch.get_device(x))
        for t in range(self.num_steps):
            spk_x = self.features(x[:, t], t)
            spk_x = self.avgpool(spk_x)
            spk_x = torch.flatten(spk_x, 1)
            out = self.classifier(spk_x, t)
            mem_fc2 = mem_fc2 + out

        return mem_fc2 / self.num_steps


def vgg_with_bntt(cfg: str, batch_norm: bool, num_steps, num_cls, beta=0.95, **kwargs: Any) -> VGG:
    model = VGG(make_layers(configurations[cfg], batch_norm=batch_norm, num_steps=num_steps, beta=beta),
                num_steps=num_steps,
                num_classes=num_cls,
                beta=beta, **kwargs)
    return model
