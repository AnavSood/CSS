from pycss.PCSS import *
from pycss.utils import *
from pycss.subset_selection import *

def test_greedy_sph_PCSS_from_data():
    
    np.random.seed(0)
    n = 100
    p = 50
    k = 10
    X = np.random.normal(0, 1, (n, p))
    _, Sigma = get_moments(X)
    pcss = PCSS()
    pcss.compute_MLE_from_data(X, k, method='greedy', noise='sph')
    S, Sigma_R, colinearity_errors = greedy_subset_selection(Sigma, 
                                                             k, 
                                                             sph_pcss_objective)
    MLE, C_MLE_chol, C_MLE_inv = compute_MLE_from_selected_subset(Sigma, 
                                                                  S, 
                                                                  Sigma_R=Sigma_R,
                                                                  noise='sph', 
                                                                  mu_MLE=np.mean(X, axis=0))
    
    assert(pcss.n == n)
    assert(np.all(pcss.X == X))
    assert(pcss.p == p)
    assert(pcss.k == k)
    assert(np.all(pcss.Sigma == Sigma))
    assert(np.all(pcss.S == S))
    assert(np.all(pcss.Sigma_R == Sigma_R))
    assert(pcss.S_init is None)
    assert(pcss.converged is None)
    assert(pcss.MLE.keys() == MLE.keys())
    for key in MLE.keys():
        assert(np.all(pcss.MLE[key] == MLE[key]))
    assert(np.all(pcss.C_MLE_chol == C_MLE_chol))
    assert(np.all(pcss.C_MLE_inv == C_MLE_inv))
    assert(pcss.log_likelihood is not None)
    assert(np.all(np.array(pcss.colinearity_errors) == colinearity_errors))

def test_greedy_diag_PCSS_from_data():
    
    np.random.seed(0)
    n = 100
    p = 50
    k = 10
    X = np.random.normal(0, 1, (n, p))
    _, Sigma = get_moments(X)
    pcss = PCSS()
    pcss.compute_MLE_from_data(X, k, method='greedy', noise='diag')
    S, Sigma_R, colinearity_errors = greedy_subset_selection(Sigma, 
                                                             k, 
                                                             diag_pcss_objective)
    MLE, C_MLE_chol, C_MLE_inv = compute_MLE_from_selected_subset(Sigma, 
                                                                  S, 
                                                                  Sigma_R=Sigma_R,
                                                                  noise='diag', 
                                                                  mu_MLE=np.mean(X, axis=0))
    
    assert(pcss.n == n)
    assert(np.all(pcss.X == X))
    assert(pcss.p == p)
    assert(pcss.k == k)
    assert(np.all(pcss.Sigma == Sigma))
    assert(np.all(pcss.S == S))
    assert(np.all(pcss.Sigma_R == Sigma_R))
    assert(pcss.S_init is None)
    assert(pcss.converged is None)
    assert(pcss.MLE.keys() == MLE.keys())
    for key in MLE.keys():
        assert(np.all(pcss.MLE[key] == MLE[key]))
    assert(np.all(pcss.C_MLE_chol == C_MLE_chol))
    assert(np.all(pcss.C_MLE_inv == C_MLE_inv))
    assert(pcss.log_likelihood is not None)
    assert(np.all(np.array(pcss.colinearity_errors) == colinearity_errors))

def test_swap_sph_PCSS_from_data():
    
    np.random.seed(0)
    n = 100
    p = 50
    k = 10
    X = np.random.normal(0, 1, (n, p))
    _, Sigma = get_moments(X)
    pcss = PCSS()
    pcss.compute_MLE_from_data(X, k, method='swap', noise='sph')
    S, Sigma_R, S_init, converged, colinearity_errors = swapping_subset_selection(Sigma, 
                                                                                  k, 
                                                                                  sph_pcss_objective,
                                                                                  S_init=pcss.S_init)

    MLE, C_MLE_chol, C_MLE_inv = compute_MLE_from_selected_subset(Sigma, 
                                                                  S, 
                                                                  Sigma_R=Sigma_R,
                                                                  noise='sph', 
                                                                  mu_MLE=np.mean(X, axis=0))
    
    assert(pcss.n == n)
    assert(np.all(pcss.X == X))
    assert(pcss.p == p)
    assert(pcss.k == k)
    assert(np.all(pcss.Sigma == Sigma))
    assert(np.all(pcss.S == S))
    assert(np.all(pcss.Sigma_R == Sigma_R))
    assert(set(pcss.S_init)==set(S_init))
    assert(pcss.converged == converged)
    assert(pcss.MLE.keys() == MLE.keys())
    for key in MLE.keys():
        assert(np.all(pcss.MLE[key] == MLE[key]))
    assert(np.all(pcss.C_MLE_chol == C_MLE_chol))
    assert(np.all(pcss.C_MLE_inv == C_MLE_inv))
    assert(pcss.log_likelihood is not None)
    assert(np.all(np.array(pcss.colinearity_errors) == colinearity_errors))

def test_swap_diag_PCSS_from_data():
    
    np.random.seed(0)
    n = 100
    p = 50
    k = 10
    X = np.random.normal(0, 1, (n, p))
    _, Sigma = get_moments(X)
    pcss = PCSS()
    pcss.compute_MLE_from_data(X, k, method='swap', noise='diag')
    S, Sigma_R, S_init, converged, colinearity_errors = swapping_subset_selection(Sigma, 
                                                                                  k, 
                                                                                  diag_pcss_objective,
                                                                                  S_init=pcss.S_init)

    MLE, C_MLE_chol, C_MLE_inv = compute_MLE_from_selected_subset(Sigma, 
                                                                  S, 
                                                                  Sigma_R=Sigma_R,
                                                                  noise='diag', 
                                                                  mu_MLE=np.mean(X, axis=0))
    
    assert(pcss.n == n)
    assert(np.all(pcss.X == X))
    assert(pcss.p == p)
    assert(pcss.k == k)
    assert(np.all(pcss.Sigma == Sigma))
    assert(np.all(pcss.S == S))
    assert(np.all(pcss.Sigma_R == Sigma_R))
    assert(set(pcss.S_init)==set(S_init))
    assert(pcss.converged == converged)
    assert(pcss.MLE.keys() == MLE.keys())
    for key in MLE.keys():
        assert(np.all(pcss.MLE[key] == MLE[key]))
    assert(np.all(pcss.C_MLE_chol == C_MLE_chol))
    assert(np.all(pcss.C_MLE_inv == C_MLE_inv))
    assert(pcss.log_likelihood is not None)
    assert(np.all(np.array(pcss.colinearity_errors) == colinearity_errors))