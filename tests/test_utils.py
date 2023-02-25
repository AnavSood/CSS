import numpy as np
from pycss.utils import *

def test_get_moments():
    X = np.array([[1, 1], [-1, -1], [0, 0]])
    mu_hat, Sigma_hat = get_moments(X)
    assert np.all(mu_hat == np.zeros(2))
    assert np.all(Sigma_hat == 2/3*np.ones(4).reshape((2, 2)))

def test_scale_cov():
    Sigma = np.ones((10, 10))
    v = np.arange(1, 11)
    assert np.all(scale_cov(Sigma, v) == np.outer(v, v)) 

def test_standardize_cov():
    rho = 1
    Sigma = 3*np.eye(10) + np.ones((10, 10))
    assert np.all(standardize_cov(Sigma) == 3/4*np.eye(10) + 1/4*np.ones(100).reshape((10, 10)))