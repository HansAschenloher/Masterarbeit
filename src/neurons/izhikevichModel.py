import numpy as np
import snntorch as snn
import torch

from snntorch._neurons.neurons import _SpikeTorchConv


class Izhikevich(snn.Leaky):

    def __init__(
            self,
            a,
            b,
            c,
            d,
            threshold=0.03,
            spike_grad=None,
            surrogate_disable=False,
            init_hidden=False,
            inhibition=False,
            learn_beta=False,
            learn_threshold=False,
            reset_mechanism="subtract",
            state_quant=False,
            output=False,
            graded_spikes_factor=1.0,
            learn_graded_spikes_factor=False,
            reset_delay=True,
    ):
        super().__init__(
            1,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )

        self._init_gate_variables()
        self.a = torch.Tensor([a])
        self.b = torch.Tensor([b])
        self.c = torch.Tensor([c])
        self.d = torch.Tensor([d])

    cfgs = {
        'RS': [[0.02, 0.2, -65, 8], [-70, -14]],
        'IB': [[0.02, 0.2, -55, 4], [-70, -14]],
        'CH': [[0.02, 0.2, -50, 2], [-70, -14]],
        'LTS': [[0.02, 0.25, -65, 2], [-64.4, -16.1]],
        'TC': [[0.02, 0.25, -65, 0.05], [-64.4, -16.1]],
        'FS': [[0.1, 0.2, -65, 2], [-70, -14]],
        'RZ': [[0.1, 0.25, -65, 2], [-64.4, -16.1]]
    }

    def _init_gate_variables(self):
        a = torch.zeros(1)
        b = torch.zeros(1)
        c = torch.zeros(1)
        d = torch.zeros(1)
        u = torch.zeros(1)
        mem = torch.zeros(1)
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.register_buffer("c", c)
        self.register_buffer("d", d)
        self.register_buffer("u", u)
        self.register_buffer("mem", mem)

    def _base_state_function(self, input_, v, u, time_resolution):
        dv = 0.04 * v * v + 5 * v + 140 - u + input_
        du = self.a * (self.b * v - u)
        return v + dv/time_resolution, u + du/time_resolution

    def _base_sub(self, input_):
        v, u = self._base_state_function(input_)
        return v - self.reset * self.threshold, u + self.d

    def _base_zero(self, input_, v, u, time_resolution):
        self.mem = torch.Tensor((1 - self.reset) * v + self.reset * self.c)
        v, u = self._base_state_function(input_, self.mem, u, time_resolution=time_resolution)
        return v, u + self.d * self.reset

    def _base_int(self, input_):
        return self._base_state_function(input_)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)

    def forward(self, input_, v=False, u=False, time_resolution=1):

        if hasattr(v, "init_flag"):  # only triggered on first-pass
            v = _SpikeTorchConv(v, input_=input_)
        elif v is False and hasattr(
                v, "init_flag"
        ):  # init_hidden case
            v = _SpikeTorchConv(v, input_=input_)

        if hasattr(u, "init_flag"):  # only triggered on first-pass
            u = _SpikeTorchConv(u, input_=torch.Tensor(0))
        elif u is False and hasattr(
                u, "init_flag"
        ):  # init_hidden case
            u = _SpikeTorchConv(u, input_=torch.Tensor(0))

        if not self.init_hidden:
            self.reset = self.mem_reset(v)


            v, u = self._base_zero(input_, v, u, time_resolution= time_resolution)

            if self.state_quant:
                v = self.state_quant(v)

            if self.inhibition:
                spk = self.fire_inhibition(v.size(0), self.mem)  # batch_size
            else:
                spk = self.fire(v)

            return spk, v, u
