import numpy as np 
from pycss.subset_selection import *
from pycss.utils import * 

def sample_null_dist(n, p, k, B=int(1e5), seed=0):
    if seed is not None:
        np.random.seed(0)

    num_adjusted_samples = n - k - 1
    num_features = p-k
    full_dfs = np.array([num_adjusted_samples - i + 1 for i in range(2, num_features + 1)])
    full_chi_sqs = np.random.chisquare(df=full_dfs, size=(B, len(full_dfs)))

    null_dfs = np.arange(1, num_features)
    null_chi_sqs = np.random.chisquare(df=null_dfs, size=(B, len(null_dfs)))
    return n*(np.sum( np.log(null_chi_sqs/full_chi_sqs + 1), axis=1))

def Q(qs, n, p, k, B=int(1e5), seed=0):
    return np.quantile(sample_null_dist(n, p, k, B=B, seed=seed), qs)

def select_subset(X, 
                  alpha, 
                  include=np.array([]), 
                  exclude=np.array([]), 
                  quantile_dict={}, 
                  B=int(1e5),
                  max_iter=100,
                  num_inits=1,
                  exhaustive_cutoff=0,
                  show_progress=True,
                  tol=TOL):
    n, p = X.shape
    _, Sigma_hat = get_moments(X)
    Sigma_hat = standardize_cov(Sigma_hat)
    
    crit_vals = np.array([Q(1-alpha, n, p, i, B=B) if (1 - alpha, n, p , i) not in quantile_dict.keys() else quantile_dict[( 1 - alpha, n, p , i)] for i in range(p + 1)])
    cutoffs = crit_vals/n  + np.linalg.slogdet(Sigma_hat)[1]

    S, reject = greedy_subset_factor_selection(Sigma_hat,
                                               cutoffs,
                                               include=include,
                                               exclude=exclude,
                                               tol=tol)
    
    
    if reject:
        warnings.warn("We can still reject the model with this S, but nothing more can be added.")
        return S
    if len(S) <= 1:
        return S 


    k = len(S)
    while not reject:
        num_options = math.comb(p - len(include) - len(exclude), k - len(include))
        k = k-1
        
        if num_options <= exhaustive_cutoff:
            S, reject = exhaustive_subset_factor_selection(Sigma_hat,
                                                           k,
                                                           cutoffs[k],
                                                           include=include,
                                                           exclude=exclude,
                                                           show_progress=show_progress,
                                                           tol=TOL)
        else:
            S, reject = swapping_subset_factor_selection(Sigma_hat,
                                                         k,
                                                         cutoffs[k],
                                                         max_iter=max_iter,
                                                         num_inits=num_inits,
                                                         include=include,
                                                         exclude=exclude,
                                                         tol=TOL)
        if reject:
            if num_options <= exhaustive_cutoff:
                S, reject = exhaustive_subset_factor_selection(Sigma_hat,
                                                           k+1,
                                                           cutoffs[k+1],
                                                           include=include,
                                                           exclude=exclude,
                                                           show_progress=show_progress,
                                                           find_minimizer=True, 
                                                           tol=TOL)
            else:
                S, reject = swapping_subset_factor_selection(Sigma_hat,
                                                             k+1,
                                                             cutoffs[k+1],
                                                             max_iter=max_iter,
                                                             num_inits=num_inits,
                                                             find_minimizer=True,
                                                             include=include,
                                                             exclude=exclude,
                                                             tol=TOL)
            return S 