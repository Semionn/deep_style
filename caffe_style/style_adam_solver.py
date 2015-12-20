import numpy as np


class StyleAdamSolver:
    def __init__(self, learn_rate, beta1=0.7, beta2=0.999, lambd=1 - 1e-8,
                 eps=1e-8):
        self.learn_rate = learn_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambd = lambd
        self.eps = eps

    def init_state(self, param):
        m = np.zeros_like(param.grad_array)
        v = np.zeros_like(param.grad_array)
        t = np.zeros(1, dtype=int)
        return m, v, t

    def step(self, param, state):
        m, v, t = state
        grad = param.grad_array
        t += 1
        t = int(t)
        beta1_t = self.beta1 * self.lambd ** (t - 1)
        m *= beta1_t
        m += (1 - beta1_t) * grad
        v *= self.beta2
        v += (1 - self.beta2) * grad ** 2
        learn_rate = (self.learn_rate * (1 - self.beta2 ** t) ** 0.5 /
                      (1 - self.beta1 ** t))
        step = m / (np.sqrt(v) + self.eps)
        step *= -learn_rate
        param.step(step)
