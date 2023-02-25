from pycss.CSS import *
from pycss.utils import *
from pycss.subset_selection import *

def test_greedy_CSS_from_data():
    
    np.random.seed(0)
    n = 100
    p = 50
    k = 10
    X = np.random.normal(0, 1, (n, p))
    _, Sigma = get_moments(X)
    css = CSS()
    css.select_subset_from_data(X, k, method='greedy')
    S, Sigma_R, colinearity_errors = greedy_subset_selection(Sigma, 
                                                             k, 
                                                             css_objective)
    
    assert(css.n == n)
    assert(np.all(css.X == X))
    assert(css.p == p)
    assert(css.k == k)
    assert(np.all(css.Sigma == Sigma))
    assert(np.all(css.S == S))
    assert(np.all(css.Sigma_R == Sigma_R))
    assert(css.S_init is None)
    assert(css.converged is None)
    assert(np.all(np.array(colinearity_errors) ==  np.array(colinearity_errors)))


def test_greedy_CSS_from_cov():
    
    np.random.seed(0)
    n = 100
    p = 50
    k = 10
    X = np.random.normal(0, 1, (n, p))
    _, Sigma = get_moments(X)
    css = CSS()
    css.select_subset_from_cov(Sigma, k, method='greedy')
    S, Sigma_R, colinearity_errors = greedy_subset_selection(Sigma, 
                                                             k, 
                                                             css_objective)
    
    assert(css.n is None)
    assert(css.X is None)
    assert(css.p == p)
    assert(css.k == k)
    assert(np.all(css.Sigma == Sigma))
    assert(np.all(css.S == S))
    assert(np.all(css.Sigma_R == Sigma_R))
    assert(css.S_init is None)
    assert(css.converged is None)
    assert(np.all(np.array(colinearity_errors) ==  np.array(colinearity_errors)))

def test_swapping_CSS_from_data():
    
    np.random.seed(0)
    n = 100
    p = 50
    k = 10
    X = np.random.normal(0, 1, (n, p))
    _, Sigma = get_moments(X)
    css = CSS()
    css.select_subset_from_data(X, k, method='swap')
    S, Sigma_R, S_init, converged, colinearity_errors = swapping_subset_selection(Sigma, 
                                                                                  k, 
                                                                                  css_objective,
                                                                                  S_init=css.S_init)
    
    assert(css.n == n)
    assert(np.all(css.X == X))
    assert(css.p == p)
    assert(css.k == k)
    assert(np.all(css.Sigma == Sigma))
    assert(np.all(css.S == S))
    assert(np.all(css.Sigma_R == Sigma_R))
    assert(set(css.S_init)==set(S_init))
    assert(css.converged == converged)
    assert(np.all(np.array(colinearity_errors) ==  np.array(colinearity_errors)))

def test_swapping_CSS_from_cov():
    
    np.random.seed(0)
    n = 100
    p = 50
    k = 10
    X = np.random.normal(0, 1, (n, p))
    _, Sigma = get_moments(X)
    css = CSS()
    css.select_subset_from_cov(Sigma, k, method='swap')
    S, Sigma_R, S_init, converged, colinearity_errors = swapping_subset_selection(Sigma, 
                                                                                  k, 
                                                                                  css_objective,
                                                                                  S_init=css.S_init)
    
    assert(css.n is None)
    assert(css.X is None)
    assert(css.p == p)
    assert(css.k == k)
    assert(np.all(css.Sigma == Sigma))
    assert(np.all(css.S == S))
    assert(np.all(css.Sigma_R == Sigma_R))
    assert(set(css.S_init)==set(S_init))
    assert(css.converged == converged)
    assert(np.all(np.array(colinearity_errors) ==  np.array(colinearity_errors)))