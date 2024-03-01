# %% [markdown]
# # Implémentation d'un _Markov Chain Monte Carlo_ (MCMC) avec l'algorithme de Metropolis.
# Nous allons suivre les étapes décrite dans l'excellent article d'introduction au MCMC de David W. Hogg et Daniel Foreman-Mackey, disponible [ici](https://ui.adsabs.harvard.edu/abs/2018ApJS..236...11H/abstract).
# La section 3 sera particulièrement utile pour ce cours.

# %%
from collections.abc import Callable
import numpy as np
import matplotlib.pyplot as plt
import tqdm

rng = np.random.default_rng()

# %% [markdown]
# ## Fonction de densité unidimensionnelle (Problèmes 2 et 3 de l'article)
# ### Densité Gaussienne
# Commençons par résoudre le problème no. 2 de l'article:
#
# - Fonction de densité $p(x)$ gaussienne à une dimension avec moyenne de 2 et variance de 2.
# - Distribution de proposition $q(x'|x)$ gaussienne pour $x'$ avec une moyenne $x$ et variance de 1.
# - Point initial $x = 0$.
# - Au moins $10^4$ itérations.

# %%
def gaussian_density(x, mean=2, var=2) -> float:
    return (np.sqrt(2 * np.pi * var)) ** -1 * np.exp(-0.5 * (x - mean) ** 2 / var)

# %%
def log_gaussian_density(x, mean=2, var=2) -> float:
    return -0.5 * (x - mean) ** 2 / var - 0.5 * np.log(2 * np.pi * var)

# %%
x_check = np.linspace(-4, 8, num=200)
plt.plot(x_check, gaussian_density(x_check), label="$\mathcal{N}$")
plt.plot(x_check, np.exp(log_gaussian_density(x_check)), "--", label="Exp(Log($\mathcal{N}$))")
plt.legend()
plt.show()

# %%
x0 = 0

n_steps = 10_000

x = x0

x_arr = []
for i in tqdm.tqdm(range(n_steps)):
    xp = rng.normal(loc=x, scale=1)
    r = rng.uniform(low=0.0, high=1.0)
    if (log_gaussian_density(xp) - log_gaussian_density(x)) > np.log(r):
        x = xp
    else:
        x = x
    x_arr += [x]

# %%
plt.plot(x_arr)
plt.show()

# %%
plt.hist(x_arr, bins=50, density=True, histtype="step", color="k")
plt.plot(x_check, gaussian_density(x_check), label="$\mathcal{N}$")
plt.show()

# %% [markdown]
# ### Distribution Uniforme

# %%
def log_uniform_density(x, low=3, high=7) -> float:
    assert low < high, "Lower bound should be lower than upper bound"
    if low < x < high:
        return - np.log(high - low)
    return -np.inf

# %%
def mcmc_metropolis(log_density: Callable, x0: float, n_steps: int) -> np.ndarray[float]:

    x_arr = np.empty(n_steps)

    x = x0
    for i in range(n_steps):
        xp = rng.normal(loc=x, scale=1.0)
        r = rng.uniform(low=0.0, high=1.0)
        if (log_density(xp) - log_density(x)) > np.log(r):
            x = xp
        x_arr[i] = x

    return x_arr

# %%
x_gaussian = mcmc_metropolis(log_gaussian_density, x0=0.0, n_steps=10_000)

# %%
plt.plot(x_gaussian)
plt.xlabel("Step")
plt.ylabel("x")
plt.show()

# %%
plt.hist(x_gaussian, bins=50, density=True, histtype="step", color="k")
plt.plot(x_check, gaussian_density(x_check), label="$\mathcal{N}$")
plt.legend()
plt.show()

# %%
x_uniform = mcmc_metropolis(log_uniform_density, x0=4.0, n_steps=100_000)

# %%
plt.plot(x_uniform)
plt.show()

# %%
plt.hist(x_uniform, bins=50, density=True, histtype="step", color="k")
plt.plot(x_check, np.exp([log_uniform_density(xi) for xi in x_check]), label="$\mathcal{U}(3, 7)$")
plt.legend()
plt.show()

# %% [markdown]
# ## Fonction de densité 2D
# Pour échantilloner un problème plus complexe, on peut généraliser le code ci-dessus à une distribution 2D

# %%
def mcmc_metropolis(log_density: Callable, x0: np.ndarray[float], n_steps: int, scale: np.ndarray[float] = None) -> np.ndarray[float]:

    x0 = np.asarray(x0)

    n_dim = len(x0)
    x_arr = np.empty((n_steps, n_dim))

    scale = scale or 1.0

    x = x0
    for i in range(n_steps):
        xp = rng.normal(loc=x, scale=scale, size=2)
        r = rng.uniform(low=0.0, high=1.0)
        if (log_density(xp) - log_density(x)) > np.log(r):
            x = xp
        x_arr[i] = x

    return x_arr

# %%
def log_gaussian_density_2d(x: np.ndarray[float]) -> float:
    x = np.asarray(x)
    ndim = 2
    mu = np.array([0.0, 0.0])
    cov = np.array([[2.0, 1.2], [1.2, 2.0]])
    assert len(x) == ndim, f"Wrong number of input dimensions. Got {len(x)}, expected {ndim}"
    return -0.5 * np.log(np.linalg.det(cov)) - 0.5 * ndim * (2 * np.pi) - 0.5 * (x - mu) @ np.linalg.inv(cov) @ (x - mu)

# %%
x_check = np.linspace(-5, 5, num=100)
y_check = x_check.copy()
# y_check = np.linspace(-10, 10, num=100)

gauss_grid = np.empty((len(x_check), len(y_check)))
for i, xc in enumerate(x_check):
    for j, yc in enumerate(y_check):
        gauss_grid[j, i] = log_gaussian_density_2d([xc, yc])

# %%
plt.pcolormesh(x_check, y_check, np.exp(gauss_grid))
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# %%
xy_arr = mcmc_metropolis(log_gaussian_density_2d, [0.0, 0.0], n_steps=10_000)

# %%
fig, axs = plt.subplots(nrows=xy_arr.shape[-1])
for i in range(xy_arr.shape[-1]):
    vals = xy_arr[:, i]
    axs[i].plot(vals)
axs[0].set_ylabel("X")
axs[1].set_ylabel("Y")
axs[-1].set_xlabel("Step")
plt.show()

# %%
plt.hist(xy_arr[:, 0], bins=50, density=True, histtype="step", color="k")
plt.xlabel("X")
plt.ylabel("Density")
plt.show()

# %%
plt.hist(xy_arr[:, 1], bins=50, density=True, histtype="step", color="k")
plt.xlabel("Y")
plt.ylabel("Density")
plt.show()

# %%
plt.scatter(*xy_arr.T, alpha=0.1)
plt.axis("square")
plt.show()

# %% [markdown]
# ## MCMC appliqué à l'analyse de données

# %%
from scipy.stats import norm


# %%
def model(param, x, xpos=10):

    A = param[0]
    B = param[1]

    m = B * np.ones_like(x) + A * norm.pdf((x - xpos) / 1.0)

    return m


# %%
data = np.loadtxt("../../devoir1/data_devoir1.csv", delimiter=",")
edata = np.sqrt(data)

# %%
l = 20
x = np.arange(l)

fig, ax = plt.subplots()
ax.step(x, data, where="mid")
ax.errorbar(x, data, yerr=edata, fmt="none", capsize=2)
ax.set_xlabel("Pixel")
ax.set_ylabel("Photons Collected")
plt.show()

# %%
def uniform_prior_scal(pval, minval, maxval):

    # NOTE: This simple implementation of prior is enough for scalars and loop
    # For arrays we need vectorize with mask.
    if (pval < minval) or (pval > maxval):
        return 0.0
    else:
        return 1.0 / (maxval - minval)


def uniform_prior(pval, minval, maxval):
    if np.isscalar(pval):
        return uniform_prior_scal(pval, minval, maxval)
    else:
        norm_const = 1.0 / (maxval - minval)
        return np.where(np.logical_and(pval >= minval, pval <= maxval), norm_const, 0.0)


def prior(param):

    A = param[0]
    B = param[1]

    pA = uniform_prior(A, 0, 1000)
    pB = uniform_prior(B, 0, 3000)

    return pA * pB


def likelihood(param, x, data, edata):

    mvals = model(param, x)

    # NOTE: The scipy function does not normalize by sigma.
    # Without it we get ~ -13 exponent instead of -46
    return np.prod(edata**-1 * norm.pdf((data - mvals) / edata), axis=0)
    # NOTE: Altenative implementation
    # return np.product((edata * np.sqrt(2 * np.pi)) ** -1 * np.exp(
    #     -0.5 * ((data - mvals) / edata) ** 2
    # ), axis=0)

def posterior(param, x, data, edata):

    return prior(param) * likelihood(param, x, data, edata)

# %%
def log_posterior(param):
    lp = np.log(prior(param))
    if not np.isfinite(lp):
        return - np.inf
    return lp + np.log(likelihood(param, x, data, edata))

# %%
A_test = 200
B_test = 1860
param_test = [A_test, B_test]

# %%
np.log(posterior(param_test, x, data, edata))

# %%
log_posterior(param_test)

# %%
AB_arr = mcmc_metropolis(log_posterior, param_test, n_steps=50_000)  #, scale=[100.0, 200.0])

# %%
xy_arr = AB_arr.copy()

# %%
fig, axs = plt.subplots(nrows=xy_arr.shape[-1])
for i in range(xy_arr.shape[-1]):
    vals = xy_arr[:, i]
    axs[i].plot(vals)
plt.show()

# %%
plt.hist(xy_arr[:, 0], bins=50, density=True, histtype="step", color="k")
plt.show()

# %%
plt.hist(xy_arr[:, 1], bins=50, density=True, histtype="step", color="k")
plt.show()

# %%
plt.scatter(*xy_arr.T)
plt.show()


# %%
