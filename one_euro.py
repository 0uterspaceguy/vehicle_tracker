import math
import numpy as np

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self,
                t0: float, 
                x0: np.ndarray, 
                min_cutoff: float=1.0, 
                beta: float=0.5,
                d_cutoff: float=1.0):
        
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        # Previous values.

        self.x_prev = x0.astype(np.float32)
        self.dx_prev = np.zeros(self.x_prev.shape, dtype=np.float32)
        self.t_prev = float(t0)

    def __call__(self, 
                 t: float, 
                 x: np.ndarray):
        """Compute the filtered signal."""

        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e

        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        # cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * np.absolute(dx_hat)

        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.


        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat