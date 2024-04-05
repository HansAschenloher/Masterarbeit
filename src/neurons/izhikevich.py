from snntorch._neurons.neurons import SpikingNeuron, _SpikeTensor
import torch.nn as nn
import torch


class Izhikevich(SpikingNeuron):

    def __init__(self,
                 a,
                 b,
                 c,
                 d,
                 learn_abcd=False,
                 time_resolution=1,
                 threshold=30.0,
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

        self._register_buffer(a, b, c, d, learn_abcd)
        self.register_buffer("time_resolution", torch.Tensor([time_resolution]))
        self.I_pre = torch.as_tensor(0)
        self.I_post = torch.as_tensor(0)
        self.register_buffer("alpha", torch.as_tensor(alpha))
        self.register_buffer("beta", torch.as_tensor(beta))

    def _register_buffer(self, a, b, c, d, learn_abcd):
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

    @staticmethod
    def init_mem():
        u, v = _SpikeTensor(init_flag=False), _SpikeTensor(init_flag=False)
        return u, v

    def forward(self, input, u, v, I_pre, I_post):
        if (self.init_hidden):
            pass
        else:
            spk = 0
            for i in range(int(self.time_resolution)):
                self.reset = self.mem_reset(v)
                u, v, I_pre, I_post = self.update_hidden(input, u, v, I_pre, I_post)
                if spk == []:
                    spk = self.fire(v)
                else:
                    spk += self.fire(v)
            return spk, u, v, I_pre, I_post

    def update_hidden(self, input_, u, v, I_pre, I_post):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            raise NotImplementedError()
        elif self.reset_mechanism_val == 1:  # reset to zero
            u, v, I_pre, I_post = self.update_state(input_, u, v, I_pre, I_post)
            u += self.reset * self.d
            v -= self.reset * (v - self.c)

            return u, v, I_pre, I_post
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            raise NotImplementedError()

    def update_state(self, input_, u, v, I_pre, I_post):

        #self.I = self.alpha * self.I + input_
        #self.I_pre = self.alpha*self.I_pre.detach() + input_
        #self.I_post = (self.beta*self.I_post.detach() - input_)

        I_pre = self.alpha*I_pre + input_
        I_post = self.beta*I_post - input_

        dv = 0.04 * v * v + 5 * v + 140 - u + I_pre + I_post
        du = self.a * (self.b * (v + dv - dv) - u)
        return u + du / self.time_resolution, v + dv / self.time_resolution, I_pre, I_post

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Izhikevich):
                cls.instances[layer].I.detach_()
