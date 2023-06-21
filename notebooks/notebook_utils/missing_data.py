import numpy as np
import cvxpy as cp 
from pycss.CSS import *

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

    return np.array(I)

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
            return where_selected
        if num_selected < k:
            upper_bound = curr_lamb
            curr_lamb = (lower_bound + curr_lamb)/2
        if num_selected > k:
            lower_bound = curr_lamb 
            curr_lamb = (upper_bound + curr_lamb)/2

def get_projected_covariance_from_missing_data(X):
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

def covariance_css_with_missing_data(X, k, method='swap', num_inits=1):
    Sigma_hat = get_projected_covariance_from_missing_data(X)
    p = Sigma_hat.shape[0]
    css = CSS()
    css.select_subset_from_cov(Sigma_hat, k=k, method=method, num_inits=num_inits)
    return css.S, css.converged

def analyze_missing_data(X_missing, X, k, methods=['cov', 'block', 'lasso'], solver='CLARABEL', method='swap', num_inits=1):
    
    n, p = X.shape
    X_missing_c = X_missing - np.nanmean(X_missing, axis=0)
    _, Sigma_hat = get_moments(X)
    css_results = {}
  
    if 'cov' in methods:
        S, converged = covariance_css_with_missing_data(X_missing, k, num_inits=num_inits)
        css_results['cov'] = np.trace(regress_off(Sigma_hat, S))

    if 'block' in methods:
        S = block_OMP_with_missing_data(X_missing_c, k)
        css_results['block'] = np.trace(regress_off(Sigma_hat, S))

    if 'lasso' in methods:
        S = group_lasso_with_missing_data(X_missing_c, k, solver=solver)
        css_results['lasso'] =  np.trace(regress_off(Sigma_hat, S))
    
    return css_results