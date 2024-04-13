import snntorch as snn
import torch
import torch.nn as nn
import sys

sys.path.append('..')
from neurons.izhikevich import Izhikevich as IZH
from neurons.lif import Leaky


class IzhikevichNet(nn.Module):
    def __init__(self, num_steps, num_input, neuron_type, alpha=0.95,use_psp=True, **kwargs):
        super().__init__()
        self.num_steps = num_steps
        self.resolution = 1
        num_inputs = num_input
        num_hidden = [100, 50]
        num_outputs = 10
        reset_mechanism = "subtract"
        self.neuron_type = neuron_type

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden[0])
        self.lif1 = IZH(*IZH.cfgs[neuron_type][0], threshold=30, time_resolution=1, graded_spikes_factor=1,
                        alpha=alpha, beta=beta, use_psp=use_psp)

        self.fc2 = nn.Linear(num_hidden[0], num_hidden[1])
        # self.lif2 = IZH(*Izhikevich.cfgs[neuron_type][0], threshold=30, time_resolution=1, graded_spikes_factor=1, alpha=0.95, beta=0.8)
        self.lif2 = Leaky(beta=0.95, reset_mechanism=reset_mechanism)

        self.fc3 = nn.Linear(num_hidden[1], num_outputs)
        self.lif3 = Leaky(beta=0.95, reset_mechanism=reset_mechanism)

    def forward(self, x):
        # Initialize hidden states at t=0
        v1, u1 = IZH.cfgs[self.neuron_type][1]
        v2, u2 = IZH.cfgs[self.neuron_type][1]
        v3, u3 = IZH.cfgs[self.neuron_type][1]
        spk3_rec = []
        mem2_rec = []

        spk1_rec = []
        spk2_rec = []

        I_pre_1, I_post_1 = 0, 0
        I_pre_2, I_post_2 = 0, 0

        for step in range(self.num_steps):
            cur1 = self.fc1(x[:, step])
            spk1, u1, v1, I_pre_1, I_post_1 = self.lif1(cur1, u1, v1, I_pre_1, I_post_1)
            spk1_rec.append(spk1)
            cur2 = self.fc2(spk1)
            spk2, v2 = self.lif2(cur2, v2)
            spk2_rec.append(spk2)
            mem2_rec.append(v2)
            # spk2, u2, v2, I_pre_2, I_post_2 = self.lif2(cur2, u2, v2, I_pre_2, I_post_2)
            # spk2_rec.append(spk2)

            # cur3 = self.fc3(spk2)
            # spk3, v3 = self.lif3(cur3, v3)
            # spk3_rec.append(spk3)
            # mem3_rec.append(v3)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0), [spk1_rec, spk2_rec]
