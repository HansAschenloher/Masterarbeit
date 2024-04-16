from snntorch.spikegen import latency

class LatencyCoding():
    def __init__(self, num_steps, gain=0.2):
        self.num_steps = num_steps
        self.gain = gain

    def __call__(self, x):
        return latency(x, num_steps=self.num_steps, gain=self.gain, )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_steps={self.num_steps}, gain={self.gain})"

