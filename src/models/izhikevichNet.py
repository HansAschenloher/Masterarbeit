import sys

import snntorch as snn
import torch
import torch.nn as nn

sys.path.append('..')

class IzhikevichNet(nn.Module):
    def __init__(self, beta, num_steps, num_input, **kwargs):
        super().__init__()
        self.num_steps = num_steps
        cfg = snn.Izhikevich.cfgs["RS"]
        layer_config = [28 * 28, 100, 10]

        layers = []
        num_in = layer_config[0]
        for i, layer in enumerate(layer_config[1:]):
            if isinstance(layer, int):
                layers.append(nn.Linear(num_in, layer))
                if i == len(layer_config) - 2:
                    layers.append(
                        snn.Leaky(beta=beta, init_hidden=True, output=True, reset_mechanism="none")
                    )
                else:
                    layers.append(
                        snn.Izhikevich(*cfg[0],
                                       initial_u=cfg[1][1],
                                       initial_v=cfg[1][0],
                                       init_hidden=True,
                                       output=False,
                                       log_spikes=True,
                                       use_psp=(i > 0)))
                num_in = layer

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.net:
            if isinstance(layer, snn.Izhikevich) or isinstance(layer, snn.Leaky):
                layer.reset_mem()

        spk_rec = []
        mem_rec = []
        for step in range(self.num_steps):
            spk, mem = self.net(x[:, step])
            spk_rec.append(spk)
            mem_rec.append(mem)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
