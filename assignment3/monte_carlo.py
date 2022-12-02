# %%
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit

# typing
from typing import Tuple
from numpy import ndarray

# %%
@njit
def metropolis(
    seed: int = 1,
    x_0: float = 0,
    delta: float = 10,
    n_0: int = 100,
    n: int = 100_000,
) -> Tuple[float, float]:

    def f(x: ndarray) -> ndarray:
        return x

    def P(x: float) -> float:
        return 0 if x < 0 else np.exp(-x)
    
    rnd.seed(seed)
    n_steps = n + n_0
    x = np.zeros(n_steps)
    d = rnd.uniform(-delta, delta, size=n_steps)
    r = rnd.uniform(0, 1, size=n_steps)

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

# %% Plot values for different values of delta between 0.01 and 10
delta = np.logspace(-2, 1, 100)
means = np.zeros_like(delta)
std_errors = np.zeros_like(delta)

for i, d in tqdm(list(enumerate(delta))):
    means[i], std_errors[i] = metropolis(delta=d, n_0=10, n=100_000)

# %% Plot values for different values of n between 100 and 100_000
n = np.logspace(2, 5, 100, dtype=np.int32)
means_n = np.zeros_like(n, dtype=np.float64)
std_errors_n = np.zeros_like(n, dtype=np.float32)

for i, n_i in tqdm(list(enumerate(n))):
    means_n[i], std_errors_n[i] = metropolis(n_0=10, n=n_i)

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
plt.title("Metropolis algorithm, function of N", fontsize=16)
plt.semilogx(n, std_errors_n, 'x-', label='standard error')
plt.semilogx(n, np.abs(1-means_n), 'x-', label='difference to the exact answer')
plt.xlabel("N", fontsize=14)
plt.ylabel("Error", fontsize=14)
plt.legend()
plt.grid()
plt.savefig("metropolis_std_error.png")
plt.show()

# %%
