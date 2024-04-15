from snntorch._neurons.neurons import SpikingNeuron, _SpikeTensor
import torch.nn as nn
import torch


class Izhikevich(SpikingNeuron):
    cfgs = {
        'RS': [[0.02, 0.2, -65, 8], [-70, -14]],
        'IB': [[0.02, 0.2, -55, 4], [-70, -14]],
        'CH': [[0.02, 0.2, -50, 2], [-70, -14]],
        'LTS': [[0.02, 0.25, -65, 2], [-64.4, -16.1]],
        'TC': [[0.02, 0.25, -65, 0.05], [-64.4, -16.1]],
        'FS': [[0.1, 0.2, -65, 2], [-70, -14]],
        'RZ': [[0.1, 0.25, -65, 2], [-64.4, -16.1]]
    }

    def __init__(self,
                 a,
                 b,
                 c,
                 d,
                 initial_u=-14.0,
                 initial_v=-70.0,
                 num_neurons=1,
                 learn_abcd=False,
                 time_resolution=1,
                 threshold=30.0,
                 use_psp=True,
                 alpha=0.9,
                 beta=0.8,
                 spike_grad=None,
                 surrogate_disable=False,
                 init_hidden=False,
                 inhibition=False,
                 learn_threshold=False,
                 reset_mechanism="zero",
                 state_quant=False,
                 output=False,
                 graded_spikes_factor=1.0,
                 learn_graded_spikes_factor=False,
                 ):
        super().__init__(
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self._register_buffer(a, b, c, d, learn_abcd, initial_u, initial_v, num_neurons)
        self.register_buffer("time_resolution", torch.Tensor([time_resolution]))
        self.use_psp = use_psp

        self.register_buffer("alpha", torch.as_tensor(alpha))
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("syn_exc", torch.as_tensor(0))
        self.register_buffer("syn_inh", torch.as_tensor(0))

    def _register_buffer(self, a, b, c, d, learn_abcd, initial_u, initial_v, num_neurons):
        if not isinstance(a, torch.Tensor):
            a = torch.as_tensor(float(a))
        if not isinstance(b, torch.Tensor):
            b = torch.as_tensor(float(b))
        if not isinstance(c, torch.Tensor):
            c = torch.as_tensor(float(c))
        if not isinstance(d, torch.Tensor):
            d = torch.as_tensor(float(d))
        if learn_abcd:
            self.a = nn.Parameter(a)
            self.b = nn.Parameter(b)
            self.c = nn.Parameter(c)
            self.d = nn.Parameter(d)
        else:
            self.register_buffer("a", a)
            self.register_buffer("b", b)
            self.register_buffer("c", c)
            self.register_buffer("d", d)
        self.register_buffer("u", torch.as_tensor([initial_u] * num_neurons))
        self.register_buffer("v", torch.as_tensor([initial_v] * num_neurons))

    @staticmethod
    def init_mem():
        u, v = _SpikeTensor(init_flag=False), _SpikeTensor(init_flag=False)
        return u, v

    def forward(self, input, u=None, v=None, syn_exc=None, syn_inh=None):
        if (self.init_hidden and u != None):
            self.u = u
        if (self.init_hidden and v != None):
            self.v = v
        if (self.init_hidden and syn_exc != None and self.use_psp):
            self.syn_exc = syn_exc
        if (self.init_hidden and syn_inh != None and self.use_psp):
            self.syn_inh = syn_inh

        spk = 0
        for i in range(int(self.time_resolution)):
            self.reset = self.mem_reset(self.v)
            self.u, self.v, self.syn_exc, self.syn_inh = self.update_hidden(input, self.u, self.v, self.syn_exc,
                                                                            self.syn_inh)
            if spk == []:
                spk = self.fire(self.v)
            else:
                spk += self.fire(self.v)

        if (self.init_hidden):
            if (self.output):
                return spk, self.v
            return spk
        else:
            return spk, self.u, self.v, self.syn_exc, self.syn_inh

    def update_hidden(self, input_, u, v, syn_exc, syn_inh):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            raise NotImplementedError()
        elif self.reset_mechanism_val == 1:  # reset to zero
            u, v, syn_exc, syn_inh = self.update_state(input_, u, v, syn_exc, syn_inh)
            u += self.reset * self.d
            v -= self.reset * (v - self.c)

            return u, v, syn_exc, syn_inh
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            raise NotImplementedError()

    def update_state(self, input_, u, v, syn_exc, syn_inh):
        if (self.use_psp):
            syn_exc = self.alpha * syn_exc + input_
            syn_inh = self.beta * syn_inh - input_
            dv = 0.04 * v * v + 5 * v + 140 - u + syn_exc + syn_inh
        else:
            dv = 0.04 * v * v + 5 * v + 140 - u + input_
        du = self.a * (self.b * (v + dv - dv) - u)
        return u + du / self.time_resolution, v + dv / self.time_resolution, syn_exc, syn_inh

    def detach_hidden(cls):
        """Used to detach hidden states from the current graph.
        Intended for use in truncated backpropagation through
        time where hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Izhikevich):
                cls.instances[layer].syn_exc.detach_()
                cls.instances[layer].syn_inh.detach_()
                cls.instances[layer].u.detach_()
                cls.instances[layer].v.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance
        variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Izhikevich):
                cls.instances[layer].syn_exc = _SpikeTensor(init_flag=False)
                cls.instances[layer].syn_inh = _SpikeTensor(init_flag=False)
                cls.instances[layer].u = _SpikeTensor(init_flag=False)
                cls.instances[layer].v = _SpikeTensor(init_flag=False)
