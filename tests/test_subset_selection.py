import numpy as np
from scipy import stats 
from pycss.subset_selection import *
from pycss.utils import *

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
    regress_one_off_in_place(Sigma, 0, 2)
    assert(np.all(Sigma == copy))
    regress_one_off_in_place(Sigma, 0, 1e-10)
    assert(np.all(Sigma[0, :] == 0))
    assert(np.all(Sigma[:, 0] == 0))
    assert(np.all(copy[1:,1:] - Sigma[1:, 1:] == rho**2))

def test_regress_off_in_place():
    Sigma = np.diag(np.arange(1, 5))
    copy = Sigma.copy()
    regress_off_in_place(Sigma, np.array([]), 1e-10)
    assert(np.all(Sigma == copy))
    regress_off_in_place(Sigma, np.array([0, 1, 2]), 1e-10)
    assert(np.all(Sigma[:3, :3] == 0))
    assert(np.all(Sigma[3:, 3:] == copy[3:, 3:]))

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
    tol = 1e-10
    Sigma = np.array([[1, 5], [5, 25 + 3*tol]])
    flag, _ = is_invertible(Sigma, tol)
    assert(flag == False)
    
    p = 10
    rho = 0.5
    Sigma = get_equicorrelated_matrix(p, rho)
    np.fill_diagonal(Sigma, np.arange(1, p + 1))
    flag, Sigma_L = is_invertible(Sigma, tol)
    assert(flag == True)
    assert(np.allclose(Sigma_L, np.linalg.cholesky(Sigma)))


def test_css_objective():
    p = 4
    rho = 1/2
    Sigma_R = np.zeros((p, p))
    Sigma_R[:3, :3] = get_equicorrelated_matrix(p-1, rho)
    np.fill_diagonal(Sigma_R, np.array([1, 2, 4, 0]))
    obj_vals, _ = css_objective(Sigma_R, flag_colinearity=False, tol=1e-10)
    assert np.all(obj_vals == np.array([-(1**2 + 2*rho**2)/1,
                                        -(2**2 + 2*rho**2)/2, 
                                        -(4**2 + 2*rho**2)/4, 
                                        0]))

def test_pcss_objective():
    p = 3
    rho = 1/2
    Sigma_R = get_equicorrelated_matrix(p, rho)
    np.fill_diagonal(Sigma_R, np.array([1, 2, 4]))
    obj_vals, _ = pcss_objective(Sigma_R, 
                                 noise='sph',
                                 flag_colinearity=True, 
                                 tol=1e-10)
    assert(np.all(obj_vals == np.array([np.log(1) + 2*np.log(np.sum([2 - rho**2, 4 - rho**2])) ,
                                        np.log(2) + 2*np.log(np.sum([1 - rho**2/2, 4 - rho**2/2])),
                                        np.log(4) + 2*np.log(np.sum([1 - rho**2/4, 2 - rho**2/4]))])))
    obj_vals, _ = pcss_objective(Sigma_R, 
                                 noise='diag',
                                 flag_colinearity=True, 
                                 tol=1e-10)
    assert(np.all(obj_vals == np.array([np.log(1) + np.sum(np.log([2 - rho**2, 4 - rho**2])) ,
                                        np.log(2) + np.sum(np.log([1 - rho**2/2, 4 - rho**2/2])),
                                        np.log(4) + np.sum(np.log([1 - rho**2/4, 2 - rho**2/4]))])))

def test_populate_colinearity_errors():
    errors = populate_colinearity_errors(np.array([0, 1]), 
                                         additions=np.array([2, 2, 2, 5, 6]),
                                         responses=np.array([4, 8, 9, 10, 11]))
    assert(len(errors) == 3)

    errors = populate_colinearity_errors(np.array([0, 1]))
    assert(len(errors) == 1)

    errors = populate_colinearity_errors(np.array([0, 1]), responses=np.array([4, 8, 9, 10, 11]))
    assert(len(errors) == 1)

def test_greedy_subset_selection():
    p = 15
    Sigma = np.diag(np.arange(1, p+1))
    S, Sigma_R, errors = greedy_subset_selection(Sigma, 
                                                 k=10, 
                                                 objective=css_objective, 
                                                 tol=1e-10,
                                                 flag_colinearity=False)
    assert(np.all(S == np.arange(14, 4, -1)))
    assert(np.all(Sigma_R[5:, 5:] == 0))
    assert(np.all(Sigma_R[:5, :5] == Sigma[:5, :5]))
    assert(len(errors) == 0)

    Sigma[14, :] = 0
    Sigma[:, 14] = 0

    S, Sigma_R, errors = greedy_subset_selection(Sigma, 
                                                 k=15, 
                                                 objective=css_objective, 
                                                 tol=1e-10,
                                                 flag_colinearity=False)
    assert(S[-1] == -1)



def test_swapping_subset_selection():
    p=10
    k=3
    Sigma = np.ones((p, p))
    S, Sigma_R, S_init, converged, errors = swapping_subset_selection(Sigma,
                                                                      k=k,
                                                                      objective=css_objective,
                                                                      tol=1e-10,
                                                                      flag_colinearity=False)
    assert(converged == False)
    assert(len(errors) == 1)
    
    Sigma = np.eye(p)
    S, Sigma_R, S_init, converged, errors = swapping_subset_selection(Sigma,
                                                                      k=k,
                                                                      objective=css_objective,
                                                                      tol=1e-10,
                                                                      flag_colinearity=False)
    assert(converged)
    assert(set(S) == set(S_init))

    Sigma = np.diag(np.arange(1, p+1))
    S, Sigma_R, S_init, converged, errors = swapping_subset_selection(Sigma,
                                                                      k=k,
                                                                      objective=css_objective,
                                                                      tol=1e-10,
                                                                      flag_colinearity=False)
    assert(converged)
    assert(set(S) == set(np.arange(p-k, p)))

def test_swapping_and_greedy_agree():
    p = 100
    k = 1
    np.random.seed(0)
    A = np.random.normal(0, 1, (p, p))
    Sigma = A @ A.T
    S_swap, Sigma_R_swap, _, _, _ = swapping_subset_selection(Sigma,
                                                              k=k,
                                                              objective=css_objective,
                                                              tol=1e-10,
                                                              flag_colinearity=False)
    S_greedy, Sigma_R_greedy, _ = greedy_subset_selection(Sigma, 
                                                          k=k, 
                                                          objective=css_objective, 
                                                          tol=1e-10,
                                                          flag_colinearity=False)
    assert(np.all(S_greedy == S_swap))
    assert(np.allclose(Sigma_R_swap, Sigma_R_greedy))

    S_swap, Sigma_R_swap, _, _, _ = swapping_subset_selection(Sigma,
                                                              k=k,
                                                              objective=sph_pcss_objective,
                                                              tol=1e-10,
                                                              flag_colinearity=True)
    S_greedy, Sigma_R_greedy, _ = greedy_subset_selection(Sigma, 
                                                          k=k, 
                                                          objective=sph_pcss_objective,
                                                          tol=1e-10,
                                                          flag_colinearity=True)
    assert(np.all(S_greedy == S_swap))
    assert(np.allclose(Sigma_R_swap, Sigma_R_greedy))

    S_swap, Sigma_R_swap, _, _, _ = swapping_subset_selection(Sigma,
                                                              k=k,
                                                              objective=diag_pcss_objective,
                                                              tol=1e-10,
                                                              flag_colinearity=True)
    S_greedy, Sigma_R_greedy, _ = greedy_subset_selection(Sigma, 
                                                          k=k, 
                                                          objective=diag_pcss_objective,
                                                          tol=1e-10,
                                                          flag_colinearity=True)
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
    
    S_greedy, Sigma_R_greedy, _ = greedy_subset_selection(Sigma, 
                                                          k=1, 
                                                          objective=css_objective,
                                                          tol=1e-10,
                                                          flag_colinearity=False)
    assert(np.all(S_greedy == 0))

    S_swap, Sigma_R_swap, _, converged, _ = swapping_subset_selection(Sigma,
                                                                      k=2,
                                                                      objective=css_objective,
                                                                      tol=1e-10,
                                                                      flag_colinearity=False,
                                                                      S_init=np.array([1, 3]))
    assert(converged)
    assert(set(S_swap) == set([1, 3]))
                                                    
def test_swapping_beats_greedy():
    p = 100
    k = 50
    np.random.seed(0)
    A = np.random.normal(0, 1, (p, p))
    Sigma = A @ A.T

    S_greedy, Sigma_R_greedy, _ = greedy_subset_selection(Sigma, 
                                                          k=k, 
                                                          objective=css_objective, 
                                                          tol=1e-10,
                                                          flag_colinearity=False)
    
    S_swap, Sigma_R_swap, S_init, _, _ = swapping_subset_selection(Sigma, 
                                                                   k=k, 
                                                                   objective=css_objective, 
                                                                   tol=1e-10,
                                                                   flag_colinearity=False,
                                                                   S_init = S_greedy,
                                                                   max_iter=1)

    assert(np.all(S_init == S_greedy))
    assert(np.round(np.mean(np.diag(Sigma_R_swap)), 5) <= np.round(np.mean(np.diag(Sigma_R_greedy)), 5))

def test_zero_iter_swapping():
    p = 100
    k = 20
    np.random.seed(0)
    A = np.random.normal(0, 1, (p, p))
    Sigma = A @ A.T
    
    S_swap, Sigma_R_swap, S_init, _, _ = swapping_subset_selection(Sigma, 
                                                                   k=k, 
                                                                   objective=css_objective, 
                                                                   tol=1e-10,
                                                                   flag_colinearity=False,
                                                                   max_iter=0)

    regress_off_in_place(Sigma, S_init)
    assert(set(S_swap) == set(S_init))
    assert(np.allclose(Sigma_R_swap, Sigma))

def test_compute_MLE_from_selected_subset():
    
    p = 15
    k = 5
    rho = 0.25
    C = get_equicorrelated_matrix(k, rho)
    np.fill_diagonal(C, np.arange(1, len(C) + 1))
    np.random.seed(0)
    W = np.random.choice(np.array([-1, 1]),  (p - k, k))
    D = np.random.chisquare(5, p-k)
    sigma_sq = np.random.chisquare(5)
    S = np.arange(k)

    Sigma = np.zeros((p, p))
    Sigma[:k, :k] = C.copy()
    Sigma[k:, :k] = W @ C
    Sigma[:k, k:] = C @ W.T
    Sigma[k:, k:] = W @  C @ W.T + np.diag(sigma_sq*np.ones(p-k))

    MLE, _, _ = compute_MLE_from_selected_subset(Sigma, S, noise='sph')
    
    assert(MLE['mu_MLE'] is None)
    assert(np.all(MLE['C_MLE'] == C))
    assert(np.allclose(MLE['W_MLE'], W))
    assert(np.allclose(MLE['sigma_sq_MLE'], sigma_sq))
    assert(np.all(S == MLE['S_MLE']))
    
    Sigma[k:, k:] = W @  C @ W.T + np.diag(D)

    MLE, _, _ = compute_MLE_from_selected_subset(Sigma, S, noise='diag')
    assert(MLE['mu_MLE'] is None)
    assert(np.all(MLE['C_MLE'] == C))
    assert(np.allclose(MLE['W_MLE'], W))
    assert(np.allclose(MLE['D_MLE'], D))
    assert(np.all(S == MLE['S_MLE']))

def test_noise_from_MLE():

    sph_dict = {'sigma_sq_MLE' : 0}
    assert('sph' == noise_from_MLE(sph_dict))

    diag_dict = {'D_MLE' : 0}
    assert('diag' == noise_from_MLE(diag_dict))

def test_compute_log_det_Sigma_MLE():
    
    p = 15
    k = 5
    rho = 0.25
    C = get_equicorrelated_matrix(k, rho)
    np.fill_diagonal(C, np.arange(1, len(C) + 1))
    np.random.seed(0)
    W = np.random.choice(np.array([-1, 1]),  (p - k, k))
    D = np.random.chisquare(5, p-k)
    sigma_sq = np.random.chisquare(5)
    S = np.arange(k)

    Sigma = np.zeros((p, p))
    Sigma[:k, :k] = C.copy()
    Sigma[k:, :k] = W @ C
    Sigma[:k, k:] = C @ W.T
    Sigma[k:, k:] = W @  C @ W.T + np.diag(sigma_sq*np.ones(p-k))

    MLE, C_MLE_chol, _ = compute_MLE_from_selected_subset(Sigma, S, noise='sph')
    
    assert(np.allclose(np.log(np.linalg.det(Sigma)), compute_log_det_Sigma_MLE(MLE, C_MLE_chol=C_MLE_chol)))
    
    Sigma[k:, k:] = W @  C @ W.T + np.diag(D)

    MLE, C_MLE_chol, _ = compute_MLE_from_selected_subset(Sigma, S, noise='diag')

    assert(np.allclose(np.log(np.linalg.det(Sigma)), compute_log_det_Sigma_MLE(MLE, C_MLE_chol=C_MLE_chol)))

def test_compute_log_det_Sigma_MLE():
    
    p = 15
    k = 5
    rho = 0.25
    C = get_equicorrelated_matrix(k, rho)
    np.fill_diagonal(C, np.arange(1, len(C) + 1))
    np.random.seed(0)
    W = np.random.choice(np.array([-1, 1]),  (p - k, k))
    D = np.random.chisquare(5, p-k)
    sigma_sq = np.random.chisquare(5)
    S = np.arange(k)

    Sigma = np.zeros((p, p))
    Sigma[:k, :k] = C.copy()
    Sigma[k:, :k] = W @ C
    Sigma[:k, k:] = C @ W.T
    Sigma[k:, k:] = W @  C @ W.T + np.diag(sigma_sq*np.ones(p-k))
    
    Sigma_inv = np.linalg.inv(Sigma)
    MLE, _, C_MLE_inv = compute_MLE_from_selected_subset(Sigma, S, noise='sph')

    top_left_block, bottom_left_block, bottom_right_block = compute_Sigma_MLE_inv(MLE, C_MLE_inv=C_MLE_inv)
    
    assert(np.allclose(top_left_block, Sigma_inv[:k, :k]))
    assert(np.allclose(bottom_left_block, Sigma_inv[k:, :k]))
    assert(np.allclose(np.diag(bottom_right_block), Sigma_inv[k:, k:]))
    
    Sigma[k:, k:] = W @  C @ W.T + np.diag(D)
    Sigma_inv = np.linalg.inv(Sigma)
    MLE, _, C_MLE_inv = compute_MLE_from_selected_subset(Sigma, S, noise='diag')

    top_left_block, bottom_left_block, bottom_right_block = compute_Sigma_MLE_inv(MLE, C_MLE_inv=C_MLE_inv)

    assert(np.allclose(top_left_block, Sigma_inv[:k, :k]))
    assert(np.allclose(bottom_left_block, Sigma_inv[k:, :k]))
    assert(np.allclose(np.diag(bottom_right_block), Sigma_inv[k:, k:]))

def test_compute_log_det_Sigma_MLE():
    
    p = 15
    k = 5
    rho = 0.25
    C = get_equicorrelated_matrix(k, rho)
    np.fill_diagonal(C, np.arange(1, len(C) + 1))
    np.random.seed(0)
    W = np.random.choice(np.array([-1, 1]),  (p - k, k))
    D = np.random.chisquare(5, p-k)
    sigma_sq = np.random.chisquare(5)
    S = np.arange(k)

    Sigma = np.zeros((p, p))
    Sigma[:k, :k] = C.copy()
    Sigma[k:, :k] = W @ C
    Sigma[:k, k:] = C @ W.T
    Sigma[k:, k:] = W @  C @ W.T + np.diag(sigma_sq*np.ones(p-k))
    
    Sigma_chol = np.linalg.cholesky(Sigma)
    MLE, C_MLE_chol, _ = compute_MLE_from_selected_subset(Sigma, S, noise='sph')

    top_left_block, bottom_left_block, bottom_right_block = compute_Sigma_MLE_chol(MLE, C_MLE_chol=C_MLE_chol)
    
    assert(np.allclose(top_left_block, Sigma_chol[:k, :k]))
    assert(np.allclose(bottom_left_block, Sigma_chol[k:, :k]))
    assert(np.allclose(np.diag(bottom_right_block), Sigma_chol[k:, k:]))
    
    Sigma[k:, k:] = W @  C @ W.T + np.diag(D)
    Sigma_chol = np.linalg.cholesky(Sigma)
    MLE, C_MLE_chol, _  = compute_MLE_from_selected_subset(Sigma, S, noise='diag')

    top_left_block, bottom_left_block, bottom_right_block = compute_Sigma_MLE_chol(MLE, C_MLE_chol=C_MLE_chol)

    assert(np.allclose(top_left_block, Sigma_chol[:k, :k]))
    assert(np.allclose(bottom_left_block, Sigma_chol[k:, :k]))
    assert(np.allclose(np.diag(bottom_right_block), Sigma_chol[k:, k:]))

def test_compute_imputed_moments():
    
    k = 5
    p = 100
    rho = 0.5
    C = get_equicorrelated_matrix(k, rho)
    np.fill_diagonal(C, np.arange(1, len(C) + 1))
    np.random.seed(0)
    W = np.random.normal(0, 1, (p - k, k))
    D = np.random.chisquare(1, p-k)

    Sigma = np.zeros((p, p))
    Sigma[:k, :k] = C.copy()
    Sigma[k:, :k] = W @ C
    Sigma[:k, k:] = C @ W.T
    Sigma[k:, k:] = W @  C @ W.T + np.diag(D)

    MLE = {'mu_MLE': np.arange(1, p+1),
           'S_MLE': np.arange(k),
           'C_MLE': C,
           'W_MLE': W,
           'D_MLE': D} 
    X = np.random.multivariate_normal(np.ones(p), Sigma, size=(2,))
    missing_idxs = [np.array([11, 13]), np.array([1, 3, 10, 12])]
    for i in range(len(missing_idxs)):
        X[i, missing_idxs[i]] = np.nan

    def replace_submatrix(mat, ind1, ind2, mat_replace):
        for i, index in enumerate(ind1):
            mat[index, ind2] = mat_replace[i, :]
        return mat

    def naive_impute_moments(X, Sigma_MLE, mu_MLE):
        n, p = X.shape
        m = np.zeros((n, p))
        Omega = np.zeros((n, p, p))
        
        for i in range(len(X)):
            m_i = np.zeros(p)
            Omega_i = np.zeros((p, p))
            x = X[i, :]
            missing = np.where(np.isnan(x))[0]
            not_missing = complement(p, missing)
            m_i[not_missing] = x[not_missing] 
            Sigma_MLE_not_missing_inv = np.linalg.inv(Sigma_MLE[not_missing, :][:, not_missing])
            m_i[missing] = Sigma_MLE[missing, :][:, not_missing] @ Sigma_MLE_not_missing_inv  @ (x[not_missing] - mu_MLE[not_missing]) + mu_MLE[missing]

            Omega_i = replace_submatrix(Omega_i, not_missing, missing, np.outer(x[not_missing], m_i[missing]) )
            Omega_i = replace_submatrix(Omega_i, missing, not_missing,  np.outer(m_i[missing], x[not_missing]))
            Omega_i = replace_submatrix(Omega_i, missing, missing, Sigma_MLE[missing,:][:, missing] - Sigma_MLE[missing, :][:, not_missing] @ Sigma_MLE_not_missing_inv  @ Sigma_MLE[not_missing,:][:, missing] + np.outer(m_i[missing], m_i[missing]))
            Omega_i = replace_submatrix(Omega_i, not_missing, not_missing, np.outer(x[not_missing], x[not_missing] ))

            m[i, :] = m_i
            Omega[i, :] = Omega_i
        
        return m, Omega
    
    m_naive, Omega_naive = naive_impute_moments(X, Sigma, MLE["mu_MLE"])
    m, Omega = compute_imputed_moments(X, MLE)
    assert(np.allclose(m[0, :], m_naive[0, :]))
    assert(np.allclose(Omega[0, :, :], Omega_naive[0, :, :]))
    assert(np.allclose(m[1, :], m_naive[1, :]))
    assert(np.allclose(Omega[1, :, :], Omega_naive[1, :, :]))
    
def test_compute_in_sample_mean_log_likelihood():
    n = 50
    p = 10
    np.random.seed(0)
    X = np.random.multivariate_normal(np.zeros(p), np.eye(p), (n, ))
    mu_hat, Sigma_hat = get_moments(X)
    log_det = np.log(np.linalg.det(Sigma_hat))
    assert(np.allclose(compute_in_sample_mean_log_likelihood(p, log_det), 
                       np.mean(stats.multivariate_normal(mean=mu_hat, cov=Sigma_hat).logpdf(X))))
