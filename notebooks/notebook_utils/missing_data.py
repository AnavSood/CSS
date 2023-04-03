import numpy as np
import cvxpy as cp 
from notebook_utils.utils import *
from pycss.CSS import *
from pycss.PCSS import *

def block_OMP_with_missing_data(X, k):
    
    n, p = X.shape
    w = np.ones((n, p))
    w[np.where(np.isnan(X))] = 0
    Y_tilde = np.nan_to_num(X, nan=0)
    vec_Y_tilde = [Y_tilde[:, i] for i in range(p)]
    y_tilde = vec_Y_tilde.copy()
        
    A = [ w[:, j][:, None] * Y_tilde for j in range(p) ]
    I = []
    
    for i in range(k):
        D = np.vstack([A[j].T @ y_tilde[j] for j in range(p)])
        s = np.argmax(np.sum(np.square(D), axis=1))
        I.append(s)
        
        A_I = [ w[:, j][:, None] * Y_tilde[:, np.array(I)] for j in range(p) ]
        A_I_pinv = [ np.linalg.pinv(w[:, j][:, None] * Y_tilde[:, np.array(I)]) for j in range(p)]
        theta = [A_I_pinv[j] @ vec_Y_tilde[j] for j in range(p)]
        temp = [A_I[j] @ theta[j] for j in range(p)]
        y_tilde = [vec_Y_tilde[j] - temp[j] for j in range(p)]
    
    #A_I_pinv = [ np.linalg.pinv(w[:, j][:, None] * Y_tilde[:, I]) for j in range(p)]
    W_hat = [A_I_pinv[j] @ vec_Y_tilde[j] for j in complement(p, I)]
    W_hat = np.vstack(W_hat)

    return np.array(I), W_hat 

def group_lasso_with_missing_data(X, k, solver='CLARABEL', tol=1e-10):

    X = np.nan_to_num(X, nan=0)
    temp = 2 * (X.T @ X)
    np.fill_diagonal(temp, 0)
    lamb_max = np.max(np.linalg.norm(temp, axis=0))
    curr_lamb = lamb_max/2
    upper_bound = lamb_max
    lower_bound = 0 

    n, p = X.shape
    W = cp.Variable((p, p))
    lamb = cp.Parameter(nonneg=True)
    objective = cp.Minimize(cp.sum(cp.sum_squares(X  - X @ W)) + lamb * cp.sum(cp.norm(W, axis=1)))
    constraints = [cp.diag(W) == 0]
    prob = cp.Problem(objective, constraints)
    
    while True:
        lamb.value = curr_lamb
        result = prob.solve(solver=solver)
        norms = np.sum(np.square(W.value), axis=1)/p
        where_selected = np.where(norms > tol)[0]
        num_selected = len(where_selected)
        if num_selected == k:
            W_hat = W.value[where_selected, :][:, complement(p, where_selected)].T
            return where_selected, W_hat
        if num_selected < k:
            upper_bound = curr_lamb
            curr_lamb = (lower_bound + curr_lamb)/2
        if num_selected > k:
            lower_bound = curr_lamb 
            curr_lamb = (upper_bound + curr_lamb)/2

def mean_imputation_css_with_missing_data(X, k, num_inits=1):

    n, p = X.shape
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X) 
    _, Sigma_hat = get_moments(X)
    best_S = np.nan
    best_obj = np.inf
    best_converged = False
    css = CSS()
    for i in range(num_inits):
        css.select_subset_from_data(X, k, method='swap')
        potential_obj = np.trace(css.Sigma_R)
        if potential_obj < best_obj:
            best_obj = potential_obj
            best_S = css.S 
            best_converged = css.converged
    best_S_comp = complement(p, best_S)
    W_hat = get_W(Sigma_hat, best_S)
    return best_S, W_hat, best_converged

def pcss_with_missing_data(X, k, num_inits=1):

    best_MLE = np.nan
    best_log_likelihood = -np.inf
    best_converged = False
    pcss = PCSS()
    for i in range(num_inits):
        pcss.compute_MLE_from_partially_observed_data(X, k)
        potential_log_likelihood = pcss.log_likelihood
        if potential_log_likelihood > best_log_likelihood:
            best_obj = potential_log_likelihood
            best_MLE = pcss.MLE
            best_converged = pcss.converged

    return best_MLE['S_MLE'], best_MLE['W_MLE'], best_converged, best_MLE['mu_MLE']

def get_masked_covariance(X):
    n, p = X.shape
    mu = np.nanmean(X, axis=0)
    Sigma_hat = np.zeros((p, p))
    where_nan = []
    for i in range(p):
        where_nan.append(np.where(1 - np.isnan(X[:, i]))[0])
    for i in range(p):
        for j in range(i, p):
            overlap = np.array(list(set(where_nan[i]).intersection(set(where_nan[j]))))
            cov = np.cov(X[overlap, i], X[overlap, j])[0][1]
            Sigma_hat[i, j] = cov
            Sigma_hat[j, i] = cov
    
    eig_vals, eig_vectors = np.linalg.eig(Sigma_hat)
    eig_vals = np.where(eig_vals < 0 , 0, eig_vals)
    Sigma_hat = eig_vectors @ (eig_vals[:, None] * eig_vectors.T)
    
    return Sigma_hat 

def masked_css_with_missing_data(X, k, num_inits=1, method='swap'):
    Sigma_hat = get_masked_covariance(X)
    p = Sigma_hat.shape[0]
    best_S = np.nan
    best_obj = np.inf
    best_converged = False
    css = CSS()
    for i in range(num_inits):
        css.select_subset_from_cov(Sigma_hat, k, method=method)
        potential_obj = np.trace(css.Sigma_R)
        if potential_obj < best_obj:
            best_obj = potential_obj
            best_S = css.S 
            best_converged = css.converged
    best_S_comp = complement(p, best_S)
    W_hat = get_W(Sigma_hat, best_S)
    return best_S, W_hat, best_converged

def obj_css(Sigma, idxs, resid=None):
    p = len(Sigma)
    k = len(idxs)
    if resid is None:
        resid = regress_off(Sigma, idxs)
    return np.trace(resid)/(p-k)

def get_W(Sigma, S):
    p = Sigma.shape[0]
    S_comp = complement(p, S)
    return Sigma[S_comp, :][:, S] @ np.linalg.inv(Sigma[S, :][:, S])

def analyze_missing_data(X_missing, X, k, methods=['mean', 'em', 'block', 'lasso'], solver='CLARABEL', num_inits=1):
    
    n, p = X.shape
    _, Sigma_hat = get_moments(X)
    second_moment = 1/n * X.T @ X
    css_results = {}
    W_results = {}
    num_missing = np.sum(np.isnan(X_missing))

    if 'mean' in methods:
        S, W_hat, converged = mean_imputation_css_with_missing_data(X_missing, k, num_inits=num_inits)
        if not converged:
            print('mean imputation did not converge.')
        css_results['mean'] = obj_css(Sigma_hat, S)
        W_results['mean'] = np.linalg.norm(W_hat - get_W(Sigma_hat, S)) 
        #X_filled = np.where(np.isnan(X_missing), np.nanmean(X_missing, axis=0), X_missing) 
        #imputation_results['mean'] = np.sum(np.square(X - X_filled))/num_missing 
    
    if 'em' in methods:   
        S, W_hat, converged = pcss_with_missing_data(X_missing, k, num_inits=num_inits)
        css_results['em'] = obj_css(Sigma_hat, S)
        W_results['em'] = np.linalg.norm(W_hat - get_W(Sigma_hat, S)) 
        #m, _ = compute_imputed_moments(X_missing, MLE)
        #imputation_results['em'] = np.sum(np.square(X - m))/num_missing 
    
    if 'mask' in methods:
        S, W_hat, converged = masked_css_with_missing_data(X_missing, k, num_inits=num_inits)
        css_results['mask'] = obj_css(Sigma_hat, S)
        W_results['mask'] = np.linalg.norm(W_hat - get_W(Sigma_hat, S))

    if 'block' in methods:
        S, W_hat = block_OMP_with_missing_data(X_missing, k)
        css_results['block'] = obj_css(Sigma_hat, S)
        W_results['block'] = np.linalg.norm(W_hat - get_W(second_moment, S))

    if 'lasso' in methods:
        S, W_hat = group_lasso_with_missing_data(X, k, solver=solver)
        css_results['lasso'] = obj_css(Sigma_hat, S)
        W_results['lasso'] = np.linalg.norm(W_hat - get_W(second_moment, S))
    
    return css_results, W_results