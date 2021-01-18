import numpy as np
from sklearn.datasets import make_blobs


def make_orbiting_blobs(n_samples, n_steps=None, a=10, b=10, random_state=0, **kwargs):
    def center1(t):
        return (0, 0)
    
    def center2(t):
        return (a*np.cos(2*np.pi*t), b*np.sin(2*np.pi*t))
    
    def center3(t):
        t += 0.5
        return (a*np.cos(2*np.pi*t), b*np.sin(2*np.pi*t))
    
    if n_steps is None:
        n_steps = n_samples//3
    else:
        assert 3*n_steps <= n_samples
    
    samples_per_step = n_samples//n_steps
    
    samples = np.zeros((0, 2))
    y = np.zeros((0,))
    
    for i, t in enumerate(np.linspace(0, 0.5, n_steps)):
        step_samples, step_y = make_blobs(
            samples_per_step,
            centers=[center1(t), center2(t), center3(t)],
            random_state=random_state + i,
            **kwargs
        )
        samples = np.concatenate((samples, step_samples))
        y = np.concatenate((y, step_y))
    
    return samples, y
