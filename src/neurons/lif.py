from snntorch._neurons.neurons import SpikingNeuron, _SpikeTensor
import torch.nn as nn
import torch


class Leaky(SpikingNeuron):

    def __init__(self,
                 beta=0.95,
                 learn_beta=False,
                 threshold=1.0,
                 spike_grad=None,
                 surrogate_disable=False,
                 init_hidden=False,
                 inhibition=False,
                 learn_threshold=False,
                 reset_mechanism="subtract",
                 state_quant=False,
                 output=False,
                 graded_spikes_factor=0.35,
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

        self._register_beta(beta, learn_beta)

    def _register_beta(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)
        if learn_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)

    @staticmethod
    def init_leaky():
        mem = _SpikeTensor(init_flag=False)
        return mem

    def forward(self, input, mem):

        if (self.init_hidden):
            pass
        else:

            self.reset = self.mem_reset(mem)
            mem = self.update_mem(input, mem)
            spk = self.fire(mem)
            return spk, mem

    def update_mem(self, input_, mem):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            new_mem = self.update_state(input_, mem)
            new_mem -= self.reset * self.threshold
            return new_mem
        elif self.reset_mechanism_val == 1:  # reset to zero
            new_mem = self.update_state(input_, mem)
            new_mem -= self.reset * new_mem #set the mem to zero where reset flag is set
            return new_mem
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            raise NotImplementedError()

    def update_state(self, input_, mem):
        mem = self.beta.clamp(0, 1) * mem + input_
        return mem