import numpy as np
import scipy
from pycss.PCSS import * 
from pycss.subset_selection import compute_log_det_Sigma_MLE


def sample_LRT_stat_under_null(n, p, k, B=int(1e5), noise='sph', seed=0):
    
    """
    Samples the likelihood ratio test statistic for comparing the fully unrestricted 
    multivariate Gaussian model to the restricted Gaussian PCSS model under then null.
  
    Parameters
    ----------
   	n : int
        The sample size. 
    p : int
        The number of variables. 
    k : int
        The size of the subset under the null.
    B : int, default=100000
        The number of samples to take
    noise : str, default='sph'
        Whether the null is a spherical or diagonal Gaussian PCSS model. Must be 
        `'sph'` or `'diag'`. 
    seed : int, default=0
        The random seed. If `None`, then no random seed is used. 

    Returns 
	-------
    np.array
        Samples from the specified null distribution.
	"""

    if seed is not None:
        np.random.seed(0)
    
    num_adjusted_samples = n - k - 1 
    num_features = p-k
    full_dfs = np.array([num_adjusted_samples - i + 1 for i in range(1, num_features + 1)])
    full_chi_sqs = np.random.chisquare(df=full_dfs, size=(B, len(full_dfs)))
    if noise == 'sph':
        null_chi_sq = np.random.chisquare(df=int(num_features*(num_features-1)/2), size=(B, ) )
        return n*(num_features * np.log((np.sum(full_chi_sqs, axis=1) + null_chi_sq)/num_features) - np.sum(np.log(full_chi_sqs), axis=1))
    if noise =='diag':
        null_dfs = np.arange(1, num_features)
        null_chi_sqs = np.random.chisquare(df=null_dfs, size=(B, len(null_dfs)))
        null_chi_sqs = np.hstack([np.zeros(B).reshape((B, 1)), null_chi_sqs])
        return n*(np.sum( np.log(null_chi_sqs/full_chi_sqs + 1), axis=1))

def Q(alphas, n, p, k, B=int(1e5), noise='sph', seed=0):
    
    """
    Returns the quantiles of the pivotal null distribution of the likelihood ratio test statistic
    for comparing the fully unrestricted  multivariate Gaussian model to the restricted Gaussian PCSS model.
    Does so by sampling a large number of times from the distribution and returning the empirical quantiles.
  
    Parameters
    ----------
   	alphas : float, np.array
       The quantile or quantiles to return.
    n : int
        The sample size. 
    p : int
        The number of variables. 
    k : int
        The size of the subset under the null.
    B : int, default=100000
        The number of samples to take
    noise : str, default='sph'
        Whether the null is a spherical or diagonal Gaussian PCSS model. Must be 
        `'sph'` or `'diag'`. 
    seed : int, default=0
        The random seed. If `None`, then no random seed is used. 

    Returns 
	-------
    np.array
        Quantiles of the specified null distribution, in accordance with `alphas`. 
	"""
    return np.quantile(sample_LRT_stat_under_null(n, p, k, B=B, noise=noise, seed=seed), alphas)

def cov_df(p):
    return int(p*(p+1)/2)

def model_df(p, k, noise):
    if noise == 'sph':
        return cov_df(k) + (p-k)*k + 1 if k < p else cov_df(p)
    if noise == 'diag':
        return cov_df(k) + (p-k)*(k+ 1)

def sieves_gaussian_LRT(Sigma_hat, n, alpha, noise='sph', method='swap', num_inits=1, quantiles={}, B=int(1e5), seed=0, asymptotic=False):
    
    """
    Select the subset size via a sequential sieves hypothesis testing proecedure. 
  
    Parameters
    ----------
   	Sigma_hat : np.array
       Sample covariance of the observed data
    n : int
        The sample size. 
    alpha : int
        The desired level (Type I error control) of the procedure.
    noise : str, default='sph'
        Whether the null is a spherical or diagonal Gaussian PCSS model. Must be 
        `'sph'` or `'diag'`. 
    method : str, default=`'swap'`
        The method by which to select the subset. Must be `'greedy'` or `'swap'`. 
    num_inits : int, default=1
        The number of initializations to try if method is `'swap'`. Only relevant if
        method is `'swap'`. 
    quantiles : Dict[(int, int, int, float, string), float], default = {}
        A dictionary which has tabulated precomputed quantiles of the null distribution. The key should
        be of the form (n, p, k, beta, noise), where, for, the null distribution, n is the number of 
        samples, p is the number of variables, k is the size of the subset, beta is the quantile, and
        noise is either 'greedy' or 'swap' depending on whether the null is a spherical or diagonal 
        Gaussian PCSS model. If a needed quantile is not provided, it will be computed internally and
        stored. 
    B : int, default=100000
        The number of samples to take from the pivotal null distribution to find quantiles which 
        are not provided. 
    seed : int, default=0
        The random seed to use when computing quantiles which are not provided. If `None` then no 
        random seed is used.  

    Returns 
	-------
    np.array
        Quantiles of the specified null distribution, in accordance with `alphas`. 
	"""

    k = 1
    p = Sigma_hat.shape[0]
    full_log_det = np.log(np.linalg.det(Sigma_hat))
    pcss = PCSS()

    while k < p - 2:
        restricted_log_det = np.inf
        S = None
        if method == 'greedy':
            num_inits = 1
        for _ in range(num_inits):
            pcss.compute_MLE_from_cov(Sigma_hat, k, method=method, noise=noise)
            potential_restricted_log_det = compute_log_det_Sigma_MLE(pcss.MLE, pcss.C_MLE_chol)
            if potential_restricted_log_det < restricted_log_det:
                restricted_log_det = potential_restricted_log_det
                S = pcss.S

        if not asymptotic and (n, p, k, 1 - alpha, noise) not in quantiles:
            quantiles[(n, p, k, 1 - alpha, noise)] = Q(1 - alpha, n, p, k, B=B, noise=noise, seed=seed)
        
        T = n*(restricted_log_det - full_log_det)
        if not asymptotic:
            if  T <= quantiles[(n, p, k, 1 - alpha, noise)]:
                return S
            else:
                k = k + 1
        else:
            df = cov_df(p) - model_df(p, k, noise)
            if T <= scipy.stats.chi2.ppf(1 - alpha, df):
                return S
            else:
                k = k + 1

    
    return np.arange(p)