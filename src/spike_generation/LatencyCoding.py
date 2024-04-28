from snntorch.spikegen import latency

class LatencyCoding():
    def __init__(self, num_steps, tau=5, threshold=0.01, linear=False):
        self.num_steps = num_steps
        self.tau = tau
        self.threshold = threshold
        self.linear = linear

    def __call__(self, x):
        return latency(x, num_steps=self.num_steps, tau=self.tau, threshold=self.threshold, linear=self.linear)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_steps={self.num_steps}, threshold={self.threshold}, tau={self.tau})"

