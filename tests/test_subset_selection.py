import numpy as np
from scipy import stats 
from pycss.subset_selection import *
from pycss.utils import *

#####################################################################
#### MORE TESTING FOR THE ACTUAL SUBSET SELECTION FUNCTIONS #########
#### CAN BE FOUND IN THE NOTEBOOK testing_subset_selection.ipynb ####
#####################################################################

def get_equicorrelated_matrix(p, rho):
    return rho * np.ones((p, p)) + (1 - rho) * np.eye(p)

def test_random_argmin():
    x = np.array([1, 4, 0, -1, -1, 4])
    assert (random_argmin(x) in [3, 4])

def test_complement():
    n = 6 
    idxs = np.array([2, 4])
    assert(np.all(complement(n, np.array([])) ==  np.arange(n)))
    assert(np.all(complement(n, idxs) == np.array([0, 1, 3, 5]))) 

def test_perm_in_place():
    Sigma = np.diag(np.arange(1, 5))
    Sigma[0, 2] = 1
    idx_order = np.array([3, 0, 2, 1])
    perm_in_place(Sigma, np.array([0, 1, 2]), np.array([2, 1, 0]), idx_order=idx_order)
    assert(np.all(idx_order == np.array([2, 0, 3, 1])))
    assert(np.all(np.diag(Sigma) == np.array([3, 2, 1, 4])))
    assert(Sigma[2, 0] == 1)

def test_perm_in_place_3d():
    Sigma = np.diag(np.arange(1, 5))
    Sigma[0, 2] = 1
    idx_order = np.array([3, 0, 2, 1])
    Sigmas = np.dstack([Sigma, Sigma]).transpose((2, 0, 1))
    perm_in_place(Sigmas, np.array([0, 1, 2]), np.array([2, 1, 0]), idx_order=idx_order)
    assert(np.all(idx_order == np.array([2, 0, 3, 1])))
    for i in range(2):
        assert(np.all(np.diag(Sigmas[i, :, :]) == np.array([3, 2, 1, 4])))
        assert(Sigmas[i, 2, 0] == 1)

def test_swap_in_place():
    Sigma = np.diag(np.arange(1, 5))
    idx_order = np.array([3, 0, 2, 1])
    swap_in_place(Sigma, np.array([0, 1]), np.array([2, 1]), idx_order=idx_order)
    assert(np.all(idx_order == np.array([2, 0, 3, 1])))
    assert(np.all(np.diag(Sigma) == np.array([3, 2, 1, 4])))

def test_swap_in_3d_place():
    Sigma = np.diag(np.arange(1, 5))
    idx_order = np.array([3, 0, 2, 1])
    Sigmas = np.dstack([Sigma, Sigma]).transpose((2, 0, 1))
    swap_in_place(Sigmas, np.array([0, 1]), np.array([2, 1]), idx_order=idx_order)
    assert(np.all(idx_order == np.array([2, 0, 3, 1])))
    for i in range(2):
        assert(np.all(np.diag(Sigmas[i, :, :]) == np.array([3, 2, 1, 4])))

def test_regress_one_off_in_place():
    p = 10 
    rho = 1/2
    Sigma = get_equicorrelated_matrix(p, rho)
    copy = Sigma.copy()
    regress_one_off_in_place(Sigma, 0, tol=2)
    assert(np.all(Sigma == copy))
    regress_one_off_in_place(Sigma, 0, tol=TOL)
    assert(np.all(Sigma[0, :] == 0))
    assert(np.all(Sigma[:, 0] == 0))
    assert(np.all(copy[1:,1:] - Sigma[1:, 1:] == rho**2))

def test_regress_off_in_place():
    Sigma = np.diag(np.arange(1, 5))
    copy = Sigma.copy()
    regress_off_in_place(Sigma, np.array([]), tol=TOL)
    assert(np.all(Sigma == copy))
    regress_off_in_place(Sigma, np.array([0, 1, 2]), tol=TOL)
    assert(np.all(Sigma[:3, :3] == 0))
    assert(np.all(Sigma[3:, 3:] == copy[3:, 3:]))

def test_regress_one_off():
    p = 10 
    rho = 1/2
    Sigma = get_equicorrelated_matrix(p, rho)
    copy = Sigma.copy()
    Sigma_R = regress_one_off(Sigma, 0, tol=TOL)
    assert(np.all(Sigma == copy))
    assert(np.all(Sigma_R[0, :] == 0))
    assert(np.all(Sigma_R[:, 0] == 0))
    assert(np.all(Sigma[1:,1:] - Sigma_R[1:, 1:] == rho**2))

def test_regress_off():
    Sigma = np.diag(np.arange(1, 5))
    copy = Sigma.copy()
    Sigma_R = regress_off(Sigma, np.array([]), tol=TOL)
    assert(np.all(Sigma_R == copy))
    Sigma_R = regress_off(Sigma, np.array([0, 1, 2]), tol=TOL)
    assert(np.all(Sigma_R[:3, :3] == 0))
    assert(np.all(Sigma_R[3:, 3:] == copy[3:, 3:]))
    assert(np.all(Sigma == copy))

def test_update_cholesky_after_removing_first():
    # edge case
    p = 1
    rho = 0.5
    Sigma = get_equicorrelated_matrix(p, rho)
    L = np.sqrt(Sigma)
    assert(len(update_cholesky_after_removing_first(L)) == 0 )
    # real case
    p = 10
    Sigma = get_equicorrelated_matrix(p, rho)
    np.fill_diagonal(Sigma, np.arange(1, p + 1))
    L = np.linalg.cholesky(Sigma)
    L_ = update_cholesky_after_removing_first(L)
    assert(np.allclose(L_, np.linalg.cholesky(Sigma[1:, 1:])))

def test_update_cholesky_after_removing_last():
    # edge case
    p = 1
    rho = 0.5
    Sigma = get_equicorrelated_matrix(p, rho)
    L = np.sqrt(Sigma)
    assert(len(update_cholesky_after_removing_last(L)) == 0 )
    # real case
    p = 10
    Sigma = get_equicorrelated_matrix(p, rho)
    np.fill_diagonal(Sigma, np.arange(1, p + 1))
    L = np.linalg.cholesky(Sigma)
    L_ = update_cholesky_after_removing_last(L)
    assert(np.allclose(L_, np.linalg.cholesky(Sigma[:(p-1), :(p-1)])))

def test_update_cholesky_after_adding_last():
    # edge case
    L_ = update_cholesky_after_adding_last(np.array([[]]), np.array([4]))
    assert(len(L_.shape) == 2)
    assert(L_[0, 0] == 2)
    # real case
    p = 10
    rho = 0.5
    Sigma = get_equicorrelated_matrix(p, rho)
    np.fill_diagonal(Sigma, np.arange(1, p + 1))
    L = np.linalg.cholesky(Sigma)
    L_ = np.linalg.cholesky(Sigma[:(p-1), :(p-1)])
    assert(np.allclose(L, update_cholesky_after_adding_last(L_, Sigma[p-1, :])))

def test_solve_with_cholesky():
  
    p = 10
    rho = 0.5
    Sigma = get_equicorrelated_matrix(p, rho)
    np.fill_diagonal(Sigma, np.arange(1, p + 1))
    L = np.linalg.cholesky(Sigma)
    v = np.column_stack([np.arange(p), np.ones(p)])
    assert(np.allclose(np.linalg.inv(Sigma) @ v, solve_with_cholesky(L, v)))

def test_is_invertible():
    
    #edge case

    Sigma = np.array([[1, 5], [5, 25 + TOL]])
    flag, _ = is_invertible(Sigma, tol=TOL)
    assert(flag == False)
    
    p = 10
    rho = 0.5
    Sigma = get_equicorrelated_matrix(p, rho)
    np.fill_diagonal(Sigma, np.arange(1, p + 1))
    flag, Sigma_L = is_invertible(Sigma, tol=TOL)
    assert(flag == True)
    assert(np.allclose(Sigma_L, np.linalg.cholesky(Sigma)))

def test_css_score():
    p = 4
    rho = 1/2
    Sigma_R = np.zeros((p, p))
    Sigma_R[:3, :3] = get_equicorrelated_matrix(p-1, rho)
    np.fill_diagonal(Sigma_R, np.array([1, 2, 4, 0.5*TOL]))
    obj_vals = css_score(Sigma_R, tol=TOL)
    assert np.all(obj_vals == np.array([-(1**2 + 2*rho**2)/1,
                                        -(2**2 + 2*rho**2)/2, 
                                        -(4**2 + 2*rho**2)/4, 
                                        0]))


def test_greedy_css():
    Sigma = np.diag(np.array([1, 2, 3, 4, 5]))

    S, Sigma_R = greedy_css(Sigma,
                            k=5)
    assert(np.all(S  == np.array([4, 3, 2, 1, 0])))
    assert(np.all(Sigma_R == 0))

    S, Sigma_R = greedy_css(Sigma,
                            k=1,
                            exclude=np.array([4]))
    assert(np.all(S == 3))
    assert(np.all(Sigma_R == np.diag(np.array([1, 2, 3, 0, 5]))))

    S, Sigma_R = greedy_css(Sigma,
                            k=1,
                            include=np.array([0]))
    assert(np.all(S == 0))
    assert(np.all(Sigma_R == np.diag(np.array([0, 2, 3, 4, 5]))))
    
    S, Sigma_R = greedy_css(Sigma, 
                            cutoffs=6)
    assert(np.all(S == np.array([4, 3])))
    assert(np.all(Sigma_R == np.diag(np.array([1, 2, 3, 0, 0]))))

    S, Sigma_R = greedy_css(Sigma, 
                            cutoffs=6)
    
    Sigma[0, 0] = 3
    Sigma[1, 1] = 3
    Sigma[0, 1] = 3
    Sigma[1, 0] = 3
    S, Sigma_R = greedy_css(Sigma, 
                            cutoffs=4,
                            include=np.array([2]),
                            exclude=np.array([1]))
    assert(np.all(S == np.array([2, 0, 4])))
    assert(np.all(Sigma_R == np.diag(np.array([0, 0, 0, 4, 0]))))

    S, Sigma_R = greedy_css(Sigma, 
                            cutoffs=np.array([11, 6, 2, 1, 0]))
    assert(np.all(S == np.array([0, 4, 3, 2])) or np.all(S == np.array([1, 4, 3, 2])))
    assert(np.all(Sigma_R == np.diag(np.array([0, 0, 0, 0, 0]))))

def test_greedy_css_exclude_last():
    Sigma = np.eye(5)
    S, Sigma_R = greedy_css(Sigma,
                            cutoffs=0,
                            exclude=np.array([4]))
    assert(set(S) == set(np.arange(4)))
    correct_Sigma_R = np.zeros((5, 5))
    correct_Sigma_R[4, 4] = 1
    assert(np.all(Sigma_R == correct_Sigma_R))

    S, Sigma_R = greedy_css(Sigma,
                            k=4,
                            exclude=np.array([4]))
    assert(set(S) == set(np.arange(4)))
    assert(np.all(Sigma_R == correct_Sigma_R))
           


def test_swapping_css():

    p=10
    k=5
    Sigma = np.ones((p, p))
    S, Sigma_R, S_init, converged = swapping_css(Sigma,
                                                  k,
                                                  num_inits=10)
   
    assert(S is None)
    
    Sigma = np.eye(p)
    S, Sigma_R, S_init, converged = swapping_css(Sigma,
                                                 k,
                                                 S_init=np.array([0, 5, 6, 3, 2]))
    assert(converged)
    assert(set(S) == set(S_init))

    include=np.array([0, 5, 7])
    exclude=np.array([3, 2, 9])
    S, Sigma_R, S_init, converged =  swapping_css(Sigma,
                                                  k,
                                                  include=include,
                                                  exclude=exclude)
    assert(converged)
    assert(set(S) == set(S_init))
    assert(set(include).issubset(S))
    assert(len(set(exclude).intersection(S)) == 0)

    Sigma = np.diag(np.arange(1, p+1))
    S, Sigma_R, S_init, converged = swapping_css(Sigma,
                                                 k)
    assert(converged)
    assert(set(S) == set(np.arange(p-k, p)))

def test_swapping_order():
    eps=0.05
    p=6
    Sigma = np.diag(np.ones(p))
    Sigma[2, 0], Sigma[0, 2], Sigma[2, 1], Sigma[1, 2] = 1-eps, 1-eps, 1-eps, 1-eps
    Sigma[5, 3], Sigma[3, 5], Sigma[5, 4], Sigma[4, 5] = 1-eps, 1-eps, 1-eps, 1-eps
   
    S_init = np.array([1, 2])
    S, _, _, converged = swapping_css(Sigma, k=2, S_init=S_init)
    assert(np.all(S == S_init))
    assert(converged)

    S_init = np.array([4, 5])
    S, _, _, converged = swapping_css(Sigma, k=2, S_init=S_init)
    assert(np.all(S == S_init))
    assert(converged)

    S_init = np.array([1, 4])
    S, _, _, converged = swapping_css(Sigma, k=2, S_init=S_init)
    assert(set(S) == set(np.array([4, 5])))
    assert(converged)

    S_init = np.array([4, 1])
    S, _, _, converged = swapping_css(Sigma, k=2, S_init=S_init)
    assert(set(S) == set(np.array([1, 2])))
    assert(converged)


def test_swapping_and_greedy_agree():
    p = 100
    k = 1
    np.random.seed(0)
    A = np.random.normal(0, 1, (p, p))
    Sigma = A @ A.T
    S_swap, Sigma_R_swap, _, _ =  swapping_css(Sigma,
                                               k=1)
    
    S_greedy, Sigma_R_greedy, = greedy_css(Sigma, 
                                           k=1)
    assert(np.all(S_greedy == S_swap))
    assert(np.allclose(Sigma_R_swap, Sigma_R_greedy))

def test_ordozgoiti_example():
    eps = 0.01
    X = np.array([[1, 1, 1, 0],
                  [1, 1, 1 + eps, 0],
                  [1, 0, 0, 1 + eps],
                  [1, 0, 0, 1],
                  [0, 0, 0 , 1]])
    Sigma = X.T @ X
    
    S_greedy, _ = greedy_css(Sigma, 
                             k=1)
    
    assert(np.all(S_greedy == 0))

    S_swap, _, S_init, converged = swapping_css(Sigma,
                                        k=2,
                                        S_init=np.array([1, 3]))
    assert(converged)
    assert(set(S_swap) == set([1, 3]))
    assert(np.all(S_init == np.array([1, 3])))
                                                   
def test_swapping_beats_greedy():
    p = 100
    k = 50
    np.random.seed(0)
    A = np.random.normal(0, 1, (p, p))
    Sigma = A @ A.T

    S_greedy, Sigma_R_greedy = greedy_css(Sigma, 
                                          k=k)
    
    S_swap, Sigma_R_swap, S_init, _  = swapping_css(Sigma, 
                                                    k=k, 
                                                    S_init = S_greedy,
                                                    max_iter=1)

    assert(np.all(S_init == S_greedy))
    if (set(S_init) == set(S_swap)):
        assert(np.trace(Sigma_R_swap) <= np.trace(Sigma_R_greedy))
    else:
        assert(np.trace(Sigma_R_swap) < np.trace(Sigma_R_greedy))

def test_zero_iter_swapping():
    
    p = 100
    k = 20
    np.random.seed(0)
    A = np.random.normal(0, 1, (p, p))
    Sigma = A @ A.T
    
    S_swap, Sigma_R_swap, S_init, converged = swapping_css(Sigma, 
                                                           k=k, 
                                                           max_iter=0)

    regress_off_in_place(Sigma, S_init)
    assert(set(S_swap) == set(S_init))
    assert(np.allclose(Sigma_R_swap, Sigma))
    assert(not converged)

def test_exhaustive_css():
    
    p=20
    k=4
    Sigma = np.diag(np.arange(1, p + 1))
    S, Sigma_R = exhaustive_css(Sigma, k, show_progress=False)

    assert(set(S) == set(np.arange(p-k, p)))
    assert(np.all(Sigma_R == np.diag(np.concatenate([np.arange(1, p - k + 1), np.zeros(k)]))))

    Sigma = np.diag(np.arange(1, p + 1))
    S, Sigma_R = exhaustive_css(Sigma, k, include=np.array([0]), exclude=np.array([19, 17, 15]), show_progress=False)
    
    correct_S = set([0, 18, 16, 14])
    assert(set(S) == correct_S)
    correct_Sigma_R = Sigma.copy()
    for j in correct_S:
        correct_Sigma_R[j, j] = 0
    assert(np.all(Sigma_R == correct_Sigma_R))


def test_subset_factor_score():
    Sigma_R = np.ones((2, 2))
    scores, errors = subset_factor_score(Sigma_R, tol=TOL)
    assert(scores is None)
    assert(len(errors[0]) == 2)
    assert(len(errors[1]) == 2)

    p = 4
    rho = 1/2
    Sigma_R = get_equicorrelated_matrix(p, rho)
    obj_vals, errors = subset_factor_score(Sigma_R, tol=TOL)
    print(errors)
    assert(np.all(obj_vals == (p-1)*(np.log(1 - rho**2)) * np.ones(p)))
    assert(len(errors[0]) == 0)
    assert(len(errors[1]) == 0)

def test_greedy_factor_subset_selection():
    p = 5
    cutoffs = -np.inf* np.ones(p + 1)
    Sigma = np.ones((p, p))
    include = np.array([0, 1])
    S, reject = greedy_subset_factor_selection(Sigma, cutoffs=cutoffs, tol=TOL, include=include )
    assert(reject == False)
    assert(np.all(S == include))

    S, reject = greedy_subset_factor_selection(Sigma, cutoffs=cutoffs, tol=TOL)
    assert(reject == False)
    assert(len(S) == 1)

    rho = 1/2
    Sigma = np.zeros((p, p))
    Sigma[0, 0] = 1
    Sigma[0, 1:] = rho
    Sigma[1:, 0] = rho 
    Sigma[1:, 1:] = np.diag(np.ones(p-1)) + Sigma[1:, 0].reshape((p-1 ,1)) @ Sigma[0, 1:].reshape((1 ,p-1))

    S, reject = greedy_subset_factor_selection(Sigma, cutoffs=np.zeros(p+1), tol=TOL)
    assert(reject == False)
    assert(np.all(S == np.array([0])))

def test_swapping_subset_factor_selection():
    p = 5
    k=3
    cutoffs = -np.inf* np.ones(p + 1)
    Sigma = np.ones((p, p))
    include = np.array([0, 1])
    S, reject = swapping_subset_factor_selection(Sigma, k=3, cutoff=-np.inf, tol=TOL, include=include)
    assert(reject == False)
    assert(len(S) == k)
    assert(set(include).issubset(S))

    S_init=np.array([0, 1, 2])
    S, reject = swapping_subset_factor_selection(Sigma, k=3, S_init=S_init, cutoff=-np.inf, tol=TOL)
    assert(reject == False)
    assert(np.all(S == S_init))

    rho = 1/2
    Sigma = np.zeros((p, p))

    Sigma[0, 0] = 1
    Sigma[1, 1] = 1
    Sigma[:2, 2:] = rho
    Sigma[2:, :2] = rho 
    Sigma[2:, 2:] = np.diag(np.ones(p-2)) + Sigma[2:, :2] @ Sigma[:2, 2:]

    S, reject = swapping_subset_factor_selection(Sigma, k=2, S_init=np.array([2, 0]), cutoff=0, tol=TOL)
    assert(reject == False)
    assert(set(S) == set(np.array([0, 1])))