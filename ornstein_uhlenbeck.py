from typing import List, Tuple, Optional
import logging
import numpy as np
from nptyping import NDArray
import scipy.optimize as opt


class OrnsteinUhlenbeckEstimator:
    def __init__(self, data: Tuple[NDArray[float], NDArray[float]], **kwargs):
        """
        kwargs:
          n_it: number of iterations to take (default 1)
          init_mu: starting parameter for mu (default is mean of data)
          init_eta: starting parameter for eta (default is a very rough guess)
        """
        self.data = data
        n_iter = kwargs.pop("n_it", 1)
        if n_iter <= 0:
            logging.warning("Parameter estimates will not be accurate without at least one iteration.")
        for t, x in data:
            if t.size != x.shape[0]:
                raise RuntimeError("Time and signal data must have the same length.")
            if t.size < 1:
                raise RuntimeError("Not enough data points in a set.")
        self.ns = [ts.size - 1 for ts, _ in data]
        self.mu = kwargs.pop("init_mu", sum([np.sum(x) for (_, x) in data]) / sum(self.ns))
        self.eta = kwargs.pop("init_eta", np.average([eta_start(*tx) for tx in data], weights=self.ns),)
        if kwargs:
            logging.error("Unrecognized keyword arguments: {kwargs.keys()}")
            raise RuntimeError("Unrecognized keyword arguments")
        while n_iter > 0:
            self.iterate()
            n_iter -= 1
        self.variance = np.average([variance(t, x, self.eta, self.mu) for t, x in self.data], weights=self.ns )

    def iterate(self) -> None:
        """
        Do an iteration on eta then mu, using MLE for eta (at constant mu) then updating mu exactly.
        """

        def func(eta: float) -> float:
            return -2.0 * np.sum([likelihood(t, x, eta, self.mu) for t, x in self.data])

        def grad(eta: float) -> float:
            return np.array(np.sum([-2.0 * deta_likelihood(t, x, eta, self.mu) for t, x in self.data],))

        fit = opt.minimize(func, self.eta, jac=grad, bounds=[(0.0, 1.0)])
        if not fit.success:
            print("Error: fit was not successful.")
            raise RuntimeError(fit.message)
        self.eta = fit.x
        self.mu = mu_list(self.data, self.eta)

    def sigma_sq(self) -> float:
        """
        Return the sigma-squared parameter
        """
        return 2.0 * self.eta * self.variance

    def deviations(
        self, data: Optional[List[Tuple[NDArray[float], NDArray[float]]]] = None) -> NDArray[float]:
        """
        Returns the weighted deviations at each point.
        """
        the_data = data or self.data
        return np.concatenate([deviations(t, x, self.eta, self.mu) for t, x in the_data])


def eta_start(t: NDArray[float], x: NDArray[float]) -> float:
    """
    Returns a guesstimate of eta
    """
    dt = t[1:] - t[:-1]
    dx = x[1:] - x[:-1]
    xdx = (x[:-1] * dx).sum()
    x2dt = (np.square(x[:-1]) * dt).sum()
    eta = -xdx / x2dt
    assert eta > 0.0, "eta must be positive"
    return eta


def likelihood(t: NDArray[float], x: NDArray[float], eta: float, mu: float = 0.0) -> float:
    """
    Returns the likelihood where sigma^2
    """
    dt = t[1:] - t[:-1]
    nm1 = dt.size
    dev = variance(t, x, eta, mu)

    return (-0.5 * nm1 * (1.0 + np.log(2.0 * np.pi * dev))- 0.5 * np.log(-np.expm1(-2.0 * eta * dt)).sum())


def deta_likelihood(t: NDArray[float], x: NDArray[float], eta: float, mu: float = 0.0) -> float:
    """
    The gradient of the total likelihood with respect to eta.
    """
    dt = t[1:] - t[:-1]
    nm1 = dt.size
    dev = variance(t, x, eta, mu)
    deta_dev = deta_variance(t, x, eta, mu)
    exp_2etadt = np.exp(-2 * eta * dt)
    expm1_2etadt = np.expm1(-2 * eta * dt)
    return -0.5 * nm1 * deta_dev / dev + np.sum(dt * exp_2etadt / expm1_2etadt)


def opt_eta(t: NDArray[float], x: NDArray[float], mu: float = 0.0) -> float:
    """
    Return the eta parameter estimated by maximum likelihood.
    """
    eta_init = np.array([eta_start(t, x)])
    def func(eta: float) -> float:return -2.0 * likelihood(t, x, eta, mu)
    def grad(eta: float) -> float:return np.array([-2.0 * deta_likelihood(t, x, eta, mu)])
    fit = opt.minimize(func, eta_init, jac=grad, bounds=[(0.0, 1.0)])
    if not fit.success:
        print("Error: fit was not successful. {fit.message}")
    return fit.x


def deviations(t: NDArray[float], x: NDArray[float], eta: float, mu: float = 0.0) -> NDArray[float]:
    """
    Returns the appropriately-weighted unsigned deviations i.e. the difference from predicted divided by the standard deviation.
    """
    dt = t[1:] - t[:-1]
    xn = x[1:] - mu
    xp = x[:-1] - mu
    return xn - xp * np.exp(-eta * dt) / np.sqrt(-np.expm1(-2.0 * eta * dt))


def variance(t: NDArray[float], x: NDArray[float], eta: float, mu: float = 0.0) -> float:
    """
    Returns an analogue of the weighted average of squares.
    """
    dt = t[1:] - t[:-1]
    xn = x[1:] - mu
    xp = x[:-1] - mu
    return np.mean(np.square(xn - xp * np.exp(-eta * dt)) / (-np.expm1(-2.0 * eta * dt)))


def deta_variance(t: NDArray[float], x: NDArray[float], eta: float, mu: float = 0.0) -> float:
    """
    Returns the derivative of the `variance` function with respect to eta.
    """
    dt = t[1:] - t[:-1]
    xn = x[1:] - mu
    xp = x[:-1] - mu
    exp_etadt = np.exp(-eta * dt)
    expm1_2etadt = np.expm1(-2.0 * eta * dt)
    terms = (2.0 * dt * exp_etadt * (xn - xp * exp_etadt) * (xp - xn * exp_etadt)/ np.square(expm1_2etadt))
    return terms.mean()


def mu(t: NDArray[float], x: NDArray[float], eta: float) -> float:
    """
    Returns the appropriately weighted mu given eta.
    """
    num, den = _mu_one(t, x, eta)
    return num / den


def mu_list(data: List[Tuple[NDArray[float], NDArray[float]]], eta) -> float:
    """
    Returns the mu for a collection of samplings, properly weighting all together.
    """
    nums, dens = zip(*[_mu_one(t, x, eta) for t, x in data])
    return np.sum(nums) / np.sum(dens)


def _mu_one(t: NDArray[float], x: NDArray[float], eta: float) -> float:
    """
    Returns the numerator and denominator of an appropriately weighted mu given eta for a single t, x set
    """
    dt = t[1:] - t[:-1]
    xn, xp = x[1:], x[:-1]
    exp_etadt = np.exp(-eta * dt)
    expm1_etadt = np.expm1(-eta * dt)
    num = (xn - xp * exp_etadt) / (1.0 + exp_etadt)
    den = -expm1_etadt / (1.0 + exp_etadt)
    return num.sum(), den.sum()
