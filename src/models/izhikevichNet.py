import snntorch as snn
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from neurons.izhikevichModel import Izhikevich


class IzhikevichNet(nn.Module):
    def __init__(self, num_steps, num_input, neuron_type, **kwargs):
        super().__init__()
        self.num_steps = num_steps
        num_inputs = num_input
        num_hidden = 100
        num_outputs = 10

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = Izhikevich(*Izhikevich.cfgs[neuron_type][0], threshold=0.030)

        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = Izhikevich(*Izhikevich.cfgs['RS'][0], threshold=0.030)

    def forward(self, x):
        # Initialize hidden states at t=0
        #mem1 = self.lif1.init_leaky()
        #mem2 = self.lif2.init_leaky()
        v1, u1 = -70, -15
        v2, u2 = -70, -15
        spk2_rec = []
        mem2_rec = []
        for step in range(self.num_steps):
            cur1 = self.fc1(x[:,step])
            spk1, v1, u1 = self.lif1(cur1, v1, u1)
            cur2 = self.fc2(spk1)
            spk2, v2, u2 = self.lif2(cur2, v2, u2)
            spk2_rec.append(spk2)
            mem2_rec.append(v2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
