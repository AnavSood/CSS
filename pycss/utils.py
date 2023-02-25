import numpy as np

def get_moments(X):
    
    """
	Given a possibly uncentered data matrix `X`, returns the 
    sample mean and (biased) sample covariance 

	Parameters
	----------
	X : np.array
	    A `(n, p)`-shaped data matrix.
	
    Returns 
	-------
	mu_hat : np.array
        The sample mean.
    Sigma_hat : np.array
        The (biased) sample variance. 
	"""
    
    n, p = X.shape
    mu_hat = np.mean(X, axis=0)
    X_c = X - mu_hat
    Sigma_hat = 1/n * X_c.T @ X_c
    return mu_hat, Sigma_hat

def scale_cov(Sigma, s):

    """
	Given a covariance Sigma returns the covariance one gets after
    scaling the `i`-th variable by `s[i]`.

	Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix.
	s : np.array
        A `(p,)`-shaped array of scaling coefficients 

    Returns 
	-------
	np.array
       A `(p, p)`-shaped covariance matrix which corresponds to the 
       original covariance Sigma after each variable is scaled by the 
       entries in `s`.
	"""

    return s[None, :] * Sigma * s[:, None]

def standardize_cov(Sigma):
    
    """
	Given a covariance Sigma returns the covariance one gets after
    sclaing the `i`-th to have unit variance.

	Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix.

    Returns 
	-------
	np.array
       A `(p, p)`-shaped covariance matrix which corresponds to the 
       original covariance Sigma after each variable is scaled to have
       unit variance.
	"""
    s = np.diag(Sigma)**(-1/2)
    return scale_cov(Sigma, s)

