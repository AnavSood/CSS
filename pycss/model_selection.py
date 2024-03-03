import numpy as np 
from pycss.subset_selection import *
from pycss.utils import * 

def sample_null_dist(n, p, k, B=int(1e4), seed=0):

    """
    Takes `B` samples from the null distribution correponding to 
    `n` observed samples, `p` variables, and a size `k` subset. 
    """

    if seed is not None:
        np.random.seed(0)

    num_adjusted_samples = n - k - 1
    num_features = p-k
    diag_dfs = np.array([num_adjusted_samples - i + 1 for i in range(2, num_features + 1)])
    diag_chi_sqs = np.random.chisquare(df=diag_dfs, size=(B, len(diag_dfs)))

    off_diag_dfs = np.arange(1, num_features)
    off_diag_chi_sqs = np.random.chisquare(df=off_diag_dfs, size=(B, len(off_diag_dfs)))

    if seed is not None:
        np.random.seed()
    
    return n*(np.sum( np.log(off_diag_chi_sqs/diag_chi_sqs + 1), axis=1))

def Q(qs, n, p, k, B=int(1e4), seed=0):
    return np.quantile(sample_null_dist(n, p, k, B=B, seed=seed), qs)

def select_subset(X, 
                  alpha, 
                  method='swap',
                  include=np.array([]), 
                  exclude=np.array([]), 
                  quantile_dict={}, 
                  B=int(1e4),
                  max_iter=100,
                  num_inits=1,
                  exhaustive_cutoff=-1,
                  show_progress=True,
                  tol=TOL):
    
    """
    Given a `(n, p)`-shaped data matrix `X` determines the smallest subset size for which we fail
    to reject that a subset factor model is sufficient, and then selects a subset of that size.

    Parameters
	----------
	X : np.array
	    A `(n, p)`-shaped data matrix.
    alpha : float
        The error control target.
    method : str, default=`swap`
        The method by which to search for a minimizing subset during the procedure. Options are
        `swap` and `greedy`.
    include : np.array[int], default=np.array([])
        A list of variables that must be included. 
    exclude: np.array[int], default=np.array([])
        A list of variables that must not be included.
    quantile_dict : dict[(float, int, int, int)), float], default=None
        A dictionary that maps a tuple of (quantile, `n`, `p`, `k`) to the appropriate quantile of the null
        distribution. Used to determin the ciritcal values for the tests. If not passed in will be computed 
        internally (which is recommended). 
    B : int, default=10000
        Number of samples of the null distribution to take when computing the relevant quantiles to determine
        the critical values. Irrelevant if the relevant quantile is in `quantile_dict`
    max_iter : int, default=100
        Only relevant if method is `swap`. Maximum number of iterations for the swapping algorithm to achieve 
        convergence.
    num_inits : int, default=1
        Only relevant if method is `swap`. Number of random initializations to try.
    exhaustive_cutoff : int, default=-1
        Only relevant if method is `swap`. If the total number of subsets to search over is less than this value, 
        then an exhaustive search is conducted rather than search via the swapping algorithm.
    show_progress : bool, default=True
        Only relevant if method is `swap`. If `True`, informs the user of the number of subsets being searched over
        and shows a progress bar in the case of an exhaustive search. 
    tol : float, default=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
	
    Returns 
	-------
	S : np.array
        The selected subset. 
    Sigma_R : np.array
        The '(p, p)'-shaped residual covariance corresponding to `S`. 
    S_init : np.array[int]
        The initialization the resulted in the selected subset. 
    converged : bool
        Whether or not the algorithm achieved convergence. 
	"""

    n, p = X.shape
    _, Sigma_hat = get_moments(X)
    Sigma_hat = standardize_cov(Sigma_hat)

    S = select_subset_from_cov(Sigma_hat,
                               n,
                               alpha, 
                               method=method,
                               include=include, 
                               exclude=exclude, 
                               quantile_dict=quantile_dict, 
                               B=B,
                               max_iter=max_iter,
                               num_inits=num_inits,
                               exhaustive_cutoff=exhaustive_cutoff,
                               show_progress=show_progress,
                               tol=tol)
    return S

def select_subset_from_cov(Sigma_hat, 
                           n, 
                           alpha,
                           method='swap',
                           include=np.array([]), 
                           exclude=np.array([]), 
                           quantile_dict={}, 
                           B=int(1e4),
                           max_iter=100,
                           num_inits=1,
                           exhaustive_cutoff=-1,
                           show_progress=True,
                           tol=TOL):
    """
    Given a `(p, p)`-shaped sample covariance `Sigma_hat` and sample size `n` from which it was computed,
    determines finds the smallest subset size for which we fail to reject that a subset factor model is sufficient, 
    and then selects a subset of that size. See `select_subset` for a description of the inputs. 
    """
    
    p = Sigma_hat.shape[0]
    
    crit_vals = np.array([Q(1-alpha, n, p, i, B=B) if (1 - alpha, n, p , i) not in quantile_dict.keys() else quantile_dict[( 1 - alpha, n, p , i)] for i in range(p + 1)])
    cutoffs = crit_vals/n  + np.linalg.slogdet(Sigma_hat)[1]

    if method == 'greedy':
        S, reject = greedy_subset_factor_selection(Sigma_hat,
                                                   cutoffs,
                                                   include=include,
                                                   exclude=exclude,
                                                   tol=tol)
    
    
        if reject:
            warnings.warn("We can still reject the model with this S, but nothing more can be added.")
    
        return S

    if method == 'swap':

        if len(include) >= p + 1 - len(exclude):
            raise ValueError("Include and exclude are not compatiable with this problem.")
        

        for k in range(len(include), p + 1 - len(exclude)):

            if math.comb(p - len(exclude), k - len(include)) <= exhaustive_cutoff:
                S, reject = exhaustive_subset_factor_selection(Sigma_hat, 
                                                               k, 
                                                               cutoffs[k], 
                                                               include=include,
                                                               exclude=exclude,
                                                               show_progress=show_progress,
                                                               tol=tol)

            else:
                S, reject = swapping_subset_factor_selection(Sigma_hat,
                                                             k,
                                                             cutoffs[k],
                                                             max_iter=max_iter,
                                                             num_inits=num_inits,
                                                             include=include,
                                                             exclude=exclude,
                                                             tol=tol)
            if not reject:
             return S
        
        warnings.warn("We can still reject the model with this S, but nothing more can be added.")
        return S
    
############ FOR REVIEWS ONLY ##################

def comp_df(p, k, model='pcss'):
    if model == 'pcss':
        return p + k * (k+1)/2 + (p-k)*k + 1
    if model == 'sf':
        return p + k * (k+1)/2 + (p-k)*k + p-k
    
def comp_L(Sigma, n, S, penalty='AIC', model='pcss'):
    p = Sigma.shape[0]
    Sigma_R = regress_off(Sigma, S)
    (_, logdet) = np.linalg.slogdet(Sigma[:, S][S, :])
    diag = np.diag(Sigma_R)[complement(p, S)]
    if model == 'pcss':
        L = 1/2 * logdet + p/2 * (1 + np.log(2*np.pi)) + (p-k)/2 * np.log(np.sum(diag)/(p-k)) 
    if model == 'sf':
        L = 1/2 * logdet + p/2 * (1 + np.log(2*np.pi)) + 1/2 * np.sum(np.log(diag)) 
        
    return -1 * n * L 

def comp_adj_L(Sigma, n, S, penalty='AIC', model='pcss'):
    S = np.array(S)
    k = len(S)
    p = Sigma.shape[0]
    d = comp_df(p, k, model)
    L = comp_L(Sigma, n, S, penalty=penalty, model=model) 
    if 'AIC':
        return 2*d - 2*L
    if 'BIC':
        return d*np.log(n) - 2*L 
    

def forward_backward(Sigma,
                     n,
                     penalty='AIC',
                     model='pcss'):
    p = Sigma.shape[0]
    best_obj = np.inf 
    forward = True
    backward = True
    S = []
    
    while forward or backward:
        while forward:
            if len(S) == p:
                forward=False
                break
            best_i = None
            for i in range(p):
                if i not in S:
                    obj = comp_adj_L(Sigma, 
                                     n, 
                                     S + [i], 
                                     penalty=penalty, 
                                     model=model)
                if obj < best_obj:
                    best_obj = obj
                    best_i = i
                    
            if best_i is None:
                forward = False
            else:
                S = S + [best_i]
                backward = True      
        while backward:
            if len(S) == 1:
                backward = False
                break
            best_i = None
            for i in S:
                S_copy = S.copy()
                S_copy.remove(i)
                obj = comp_adj_L(Sigma, 
                                     n, 
                                     S_copy, 
                                     penalty=penalty, 
                                     model=model)
                if obj < best_obj:
                    best_obj = obj
                    best_i = i
                    
            if best_i is None:
                backward = False
            else:
                S.remove(best_i)
                forward = True
            
    return S
