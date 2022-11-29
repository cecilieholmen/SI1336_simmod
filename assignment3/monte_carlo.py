# %%
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit

# typing
from typing import Callable, Tuple
from numpy import ndarray

# %%
def metropolis(
    P: Callable, 
    f: Callable,
    seed: int = 1,
    x_0: float = 0,
    delta: float = 10,
    n_0: int = 100,
    n: int = 10_000,
) -> Tuple[float, float]:
    random_state = rnd.RandomState(seed=seed)
    n_steps = n + n_0
    x = np.zeros(n_steps)
    d = random_state.uniform(-delta, delta, size=n_steps)
    r = random_state.uniform(0, 1, size=n_steps)

    x[0] = x_0
    for i in range(n_steps-1):
        x_j = x[i] + d[i]
        w = P(x_j) / P(x[i])
        if w > r[i]:
            x[i+1] = x_j
        else:
            x[i+1] = x[i]
    
    mean = np.mean(f(x[n_0:n_0 + n]))
    std = np.std(f(x[n_0:n_0 + n]))
    standard_error = std / np.sqrt(n)
    return mean, standard_error

def f(x: ndarray) -> ndarray:
    return x

def P(x: float) -> float:
    return 0 if x < 0 else np.exp(-x)

# %% Plot values for different values of delta between 0.01 and 10
delta = np.logspace(-2, 1, 100)
means = np.zeros_like(delta)
std_errors = np.zeros_like(delta)

for i, d in tqdm(list(enumerate(delta))):
    means[i], std_errors[i] = metropolis(P, f, delta=d, n_0=10, n=100_000)

# %%
plt.title("Metropolis algorithm, n_0=10, n=100_000", fontsize=16)
plt.semilogx(delta, np.abs(1-means), 'x-', label='difference to the exact answer')
plt.semilogx(delta, std_errors, 'x-', label='standard error')
plt.xlabel("Delta", fontsize=14)
plt.ylabel("Error", fontsize=14)
plt.legend()
plt.grid()
plt.savefig("metropolis.png")
plt.show()

# %%
