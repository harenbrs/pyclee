from abc import ABC, abstractmethod

import numpy as np

from .types import Interval


class ForgettingMethod(ABC):
    @abstractmethod
    def __call__(self, interval: Interval):
        ...


class NoForgettingMethod(ForgettingMethod):
    def __call__(self, interval: Interval):
        return 1


class LinearForgettingMethod(ForgettingMethod):
    def __init__(self, t_0: Interval):
        self.t_0 = t_0
        self.m = 1/t_0
    
    def __call__(self, interval: Interval):
        if interval < self.t_0:
            return 1 - self.m*interval
        else:
            return 0


class TrapezoidalForgettingMethod(ForgettingMethod):
    def __init__(self, t_a: Interval, t_0: Interval):
        self.t_a = t_a
        self.t_0 = t_0
        self.m = 1/(t_0 - t_a)
    
    def __call__(self, interval: Interval):
        if interval <= self.t_a:
            return 1
        elif interval >= self.t_0:
            return 0
        else:
            # NOTE: differs from f_2 in paper's Table 1
            return 1 - self.m*(interval - self.t_a)


class ExponentialForgettingMethod(ForgettingMethod):
    def __init__(self, lmbda: float):
        self.lmbda = lmbda
    
    def __call__(self, interval: Interval):
        return np.exp(-self.lmbda*interval)


class SigmoidForgettingMethod(ForgettingMethod):
    def __init__(self, a: float, c: float):
        self.a = a
        self.c = c
    
    def __call__(self, interval: Interval):
        # NOTE: differs from f_6 in paper's Table 1
        return 1/(1 + np.exp(self.a*(interval - self.c)))
