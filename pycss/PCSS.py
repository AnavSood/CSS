from pycss.subset_selection import *
from pycss.utils import * 

class PCSS():

    """
    Class for performing probabilistic column subset selection under 
    the Gaussian PCSS models. 
    """
  
    def __init__(self):
        """
        Initialize `self.flag_colinearity` to `False`. 
        """
        self.flag_colinearity = False

    def _check_inputs(self, p, k, method, noise):
        
        """
        Helper function which ensures that inputed values are 
        in the allowable range or set. 

        Parameters
        ----------
        p : int
            Dimension of covariance matrix. 
	    k : int
            Size of subset to select. 
        method : str
            Method by which to select the subset. Must be `'greedy'` or `'swap'`.
        noise : str
            Noise structure of the mode. Must be `'sph'` or `'diag'`  
        """

        if not isinstance(p, (int, np.integer)):
            raise ValueError("p must be an integer.")
        if p <= 0:
            raise ValueError("p must be >= 0.")
        if not isinstance(k, (int, np.integer)):
            raise ValueError("k must be an integer.")
        if k <= 0 or k > p:
            raise ValueError("k must be > 0 and <= p.")
        if method not in {'greedy', 'swap'}:
            raise ValueError("Requested method not supported.")
        if noise not in ['sph', 'diag']:
            raise ValueError("Noise must be 'sph' or 'diag'.")


    def compute_MLE_from_data(self, 
                              X, 
                              k, 
                              noise='sph', 
                              method='greedy', 
                              tol=TOL, 
                              max_iter=100, 
                              S_init=None):
        
        """
        Given sample-level data, computes the MLE. The following class variables 
        will be populated:

        self.n : int
            Number of samples. 
        self.X : np.array
            The `(n, p)`-shaped sample-level data matrix used for selection.
        self.p : int
            Number of variables selected from.
        self.k : int
            Number of variables selected. 
        self.Sigma : np.array
            The `(p, p)`-shaped covariance matrix used for selection. 
        self.S : np.array
            The selected subset. 
        self.Sigma_R : int
            The `(p, p)`-shaped covariance matrix residual covariance from
            regressing out the selected variables. 
        self.S_init : np.array
            The `(k,)`-shaped subset used as the intialization for the swapping algorithm.
            If `method` is `'greedy'`, this will be `None`. 
        self.converged : bool
            Boolean value of whether or not the swapping algorithm converged.
            If `method` is `'greedy'`, this will be `None`. 
        self.MLE : Dict[str, np.array]
            A dictionary containing the maximum likelihood estimates. The keys for the spherical Gaussian
            PCSS model are `'mu_MLE'`, `'S_MLE'`, `'C_MLE'`, `'W_MLE'` and `'sigma_sq_MLE'`. The keys for the diagonal
            Gaussian PCSS model are the same, but `'sigma_sq_MLE'` is replaced by `'D_MLE'`.
        self.C_MLE_chol : np.array
            The '(k, k)'-shaped cholesky decomposition of C_MLE. 
        self.C_MLE_inv : np.array
            The '(k, k)'-shaped inverse of C_MLE. 
        self.log_likelihhod : float
            The maximized log likelihood, divided by the sample size `n`.
        self.colinearity_errors : list[ValueError]
            List of any errors with colinearity experienced during subset search. If this list 
            is non-empty then `self.S`, `self.Sigma_R`, `self.MLE`, `self.C_MLE_chol`, `self.C_MLE_inv`
            should all be `None`. Also if this list is non-empty and `method` is `'swap'` then 
            `converged` should be `False`.

        Parameters
        ----------
        X : np.array
            The `(n, p)`-shaped sample-level data matrix used for selection.
        k : int 
            The size of the subset to select. 
        noise : str, default=`'sph'`
            If `'sph``, finds MLE under spherical Gaussian PCSS model. If `'diag'` finds MLE 
            under diagonal Gaussian PCSS model. 
        method : str, default=`'greedy'`
            The method for selecting the subset. Should either be `'greedy'` to use greedy subset search
            or `'swap'` to use the iterative swapping, gradient descent like subset search. 
        tol : float, default=`TOL`
            Tolerance at which point we consider a variable to have zero variance.
        max_iter : int, default=`100`
            Maximum number of iterations to run the swapping algorithm. If algorithm has not 
            converged within `max_iter` iterations, `converged` will be `False. Irrelevant if 
            `method` is `'greedy'`. 
        S_init : np.array, default=`None`
            Intial subset to start the swapping algorithm with. If not included, an initial subset is 
            selected uniformly randomly. Irrelevant if `method` is `'greedy'`. 
        """

        method = method.lower()
        noise = noise.lower()
        n, p = X.shape
        self._check_inputs(p, k, method, noise)

        mu_MLE, Sigma = get_moments(X)

        self.compute_MLE_from_cov(Sigma, 
                                  k, 
                                  noise=noise,
                                  method=method, 
                                  tol=tol,
                                  max_iter=max_iter,
                                  S_init=S_init,
                                  mu_MLE=mu_MLE,
                                  from_data=True)
    
        self.n = n
        self.X = X
  

    def compute_MLE_from_cov(self, 
                             Sigma, 
                             k, 
                             noise='sph', 
                             method='greedy', 
                             tol=TOL, 
                             max_iter=100, 
                             S_init=None,
                             mu_MLE=None,
                             from_data=False):
        
        """
        Given the sample covariance, computes the MLE. The following class variables 
        will be populated:

        self.n : int
            Number of samples. Will be `None` if function is called directly by user.
        self.X : np.array
            The `(n, p)`-shaped sample-level data matrix used for selection.
            Will be `None` if function is called directly by user.
        self.p : int
            Number of variables selected from.
        self.k : int
            Number of variables selected. 
        self.Sigma : np.array
            The `(p, p)`-shaped covariance matrix used for selection. 
        self.S : np.array
            The selected subset. 
        self.Sigma_R : int
            The `(p, p)`-shaped covariance matrix residual covariance from
            regressing out the selected variables. 
        self.S_init : np.array
            The `(k,)`-shaped subset used as the intialization for the swapping algorithm.
            If `method` is `'greedy'`, this will be `None`. 
        self.converged : bool
            Boolean value of whether or not the swapping algorithm converged.
            If `method` is `'greedy'`, this will be `None`. 
        self.MLE : Dict[str, np.array]
            A dictionary containing the maximum likelihood estimates. The keys for the spherical Gaussian
            PCSS model are `'mu_MLE'`, `'S_MLE'`, `'C_MLE'`, `'W_MLE'` and `'sigma_sq_MLE'`. The keys for the diagonal
            Gaussian PCSS model are the same, but `'sigma_sq_MLE'` is replaced by `'D_MLE'`. 
        self.C_MLE_chol : np.array
            The '(k, k)'-shaped cholesky decomposition of C_MLE. 
        self.C_MLE_inv : np.array
            The '(k, k)'-shaped inverse of C_MLE. 
        self.log_likelihhod : float
            The maximized log-likelihood, divided by the sample size `n`.
        self.colinearity_errors : list[ValueError]
            List of any errors with colinearity experienced during subset search. If this list 
            is non-empty then `self.S`, `self.Sigma_R`, `self.MLE`, `self.C_MLE_chol`, `self.C_MLE_inv`
            should all be `None`. Also if this list is non-empty and `method` is `'swap'` then 
            `converged` should be `False`.

        Parameters
        ----------
        Sigma : np.array
            The `(p, p)`-shaped covariance matrix used for selection.
        k : int 
            The size of the subset to select. 
        noise : str, default=`'sph'`
            If `'sph``, finds MLE under spherical Gaussian PCSS model. If `'diag'` finds MLE 
            under diagonal Gaussian PCSS model.
        method : str, default=`'greedy'`
            The method for selecting the subset. Should either be `'greedy'` to use greedy subset search
            or `'swap'` to use the iterative swapping, gradient descent like subset search. 
        tol : float, default=`TOL`
            Tolerance at which point we consider a variable to have zero variance.
        max_iter : int, default=`100`
            Maximum number of iterations to run the swapping algorithm. If algorithm has not 
            converged within `max_iter` iterations, `converged` will be `False. Irrelevant if 
            `method` is `'greedy'`. 
        S_init : np.array, default=`None`
            Intial subset to start the swapping algorithm with. If not included, an initial subset is 
            selected uniformly randomly. Irrelevant if `method` is `'greedy'`.
        mu_MLE : None, default=`None`
            The MLE for mu. If known, user should pass in the sample mean.
        from_data : bool, default=`False`
            Whether the function was called directly by the user (in which case it should be `False`) or
            if it was called internally. 
        """

        p = Sigma.shape[0]

        if not from_data:
            if Sigma.shape[0] != Sigma.shape[1]:
                raise ValueError("Sigma must be a square matrix.")
            method = method.lower()
            noise = noise.lower()
            self._check_inputs(p, k, method, noise)
            self.n = None
            self.X = None
    
        if noise == 'sph':
            objective = sph_pcss_objective
        if noise == 'diag':
            objective = diag_pcss_objective 
  
        if method == 'greedy':
            S, Sigma_R, colinearity_errors = greedy_subset_selection(Sigma, 
                                                                     k, 
                                                                     objective, 
                                                                     tol=tol, 
                                                                     flag_colinearity=self.flag_colinearity)
            self.S_init = None
            self.converged = None 

        if method == 'swap':
            S, Sigma_R, S_init, converged, colinearity_errors = swapping_subset_selection(Sigma, 
                                                                                          k,
                                                                                          objective,
                                                                                          max_iter=max_iter,
                                                                                          S_init=S_init,
                                                                                          tol=tol,
                                                                                          flag_colinearity=self.flag_colinearity)
            self.S_init = S_init 
            self.converged = converged 
    
        self.Sigma = Sigma
        self.p = Sigma.shape[0]
        self.k = k
        self.S = S
        self.Sigma_R = Sigma_R

        if len(colinearity_errors) > 0:
            self.colinearity_errors = colinearity_errors
            self.MLE, self.C_MLE_chol, self.C_MLE_inv, self.log_likelihood = None, None, None, None
            raise ValueError("Issues with colinearity.")
            
        self.colinearity_errors = []
        MLE, C_MLE_chol, C_MLE_inv = compute_MLE_from_selected_subset(Sigma, 
                                                                      S, 
                                                                      Sigma_R=Sigma_R,
                                                                      noise=noise, 
                                                                      mu_MLE=mu_MLE)
        self.log_likelihood = compute_in_sample_mean_log_likelihood(p, compute_log_det_Sigma_MLE(MLE, C_MLE_chol=C_MLE_chol))
        self.MLE = MLE
        self.C_MLE_chol = C_MLE_chol
        self.C_MLE_inv = C_MLE_inv

  
