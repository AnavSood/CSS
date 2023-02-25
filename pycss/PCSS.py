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
        self.MLE_init : Dict[str, np.array]
            Will be `None` in this case. Only relevant for finding the MLE in presence of missing data. 
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
        self.MLE_init : Dict[str, np.array]
            Will be `None` in this case. Only relevant for finding the MLE in presence of missing data. 
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
        self.MLE_init = None

        if len(colinearity_errors) > 0:
            self.colinearity_errors = colinearity_errors
            self.MLE, self.C_MLE_chol, self.C_MLE_inv = None, None, None
            raise ValueError("Issues with colinearity.")
            
        self.colinearity_errors = []
        self.MLE, self.C_MLE_chol, self.C_MLE_inv = compute_MLE_from_selected_subset(Sigma, 
                                                                                     S, 
                                                                                     Sigma_R=Sigma_R,
                                                                                     noise=noise, 
                                                                                     mu_MLE=mu_MLE)

    def compute_MLE_from_partially_observed_data(self, 
                                                 X, 
                                                 k, 
                                                 noise='sph', 
                                                 tol=TOL, 
                                                 max_iter=100,
                                                 max_em_iter=100,
                                                 tau=1e-5, 
                                                 MLE_init=None):
        
        """
        Given the sample covariance, computes the MLE. The following class variables 
        will be populated:

        self.n : int
            Number of samples.
        self.X : np.array
            The `(n, p)`-shaped sample-level data matrix used for selection. Missing will be filled in with
            `np.nan`. 
        self.p : int
            Number of variables selected from.
        self.k : int
            Number of variables selected. 
        self.Sigma : np.array
            Will be 'None' 
        self.S : np.array
            The selected subset. 
        self.Sigma_R : int
            Will be 'None'.
        self.S_init : np.array
            Will be 'None'.
        self.converged : bool
            Boolean value of whether or not the EM algorithm converged. 
        self.MLE : Dict[str, np.array]
            A dictionary containing the maximum likelihood estimates. The keys for the spherical Gaussian
            PCSS model are `'mu_MLE'`, `'S_MLE'`, `'C_MLE'`, `'W_MLE'` and `'sigma_sq_MLE'`. The keys for the diagonal
            Gaussian PCSS model are the same, but `'sigma_sq_MLE'` is replaced by `'D_MLE'`. 
        self.C_MLE_chol : np.array
            The '(k, k)'-shaped cholesky decomposition of C_MLE. 
        self.C_MLE_inv : np.array
            The '(k, k)'-shaped inverse of C_MLE. 
        self.MLE_init : Dict[str, np.array]
            A dictionary containing the intialization for the EM algorithm. The keys are the same as in 
            `self.MLE`. 
        self.colinearity_errors : list[ValueError]
            List of any errors with colinearity experienced during subset search. If this list 
            is non-empty then `self.S`, `self.Sigma_R`, `self.MLE`, `self.C_MLE_chol`, `self.C_MLE_inv`
            should all be `None`. Also if this list is non-empty and `method` is `'swap'` then 
            `converged` should be `False`.

        Parameters
        ----------
        X : np.array
            The `(n, p)`-shaped data matrix with missing values encoded as `np.nan`.
        k : int 
            The size of the subset to select. 
        noise : str, default=`'sph'`
            If `'sph``, finds MLE under spherical Gaussian PCSS model. If `'diag'` finds MLE 
            under diagonal Gaussian PCSS model. 
        tol : float, default=`TOL`
            Tolerance at which point we consider a variable to have zero variance.
        max_iter : int, default=`100`
            Maximum number of iterations to run the swapping algorithm. If algorithm has not 
            converged within `max_iter` iterations, `converged` will be `False. Irrelevant if 
            `method` is `'greedy'`. 
        max_em_iter : int, default=`100`
            Maximum number of iterations to run the EM algorithm. 
        tau : float, default=1e-5
            The threshold to decide whether the EM algorithm is converged. If the change in observed
            likelihood is < tau, then we say the algorithm has converged. 
        MLE_init : Dict[str, np.array], default='None'
            Dictionary containing the initialization for the EM algorithm. If `MLE_init` is `None`
            or any of the values in `MLE_init` are `None`. then the initialization will be the MLE 
            for a randomly selected subset after the missing values have been mean-imputed. If MLE
            has the key 'S_init' with a corresponding entry which is not 'None', then 'S_init' will 
            be used as the initial subset instead of a randomly selected subset. 
        """
        
        n, p = X.shape
        self.n = n
        self.X = X
        self.p = p
        self.k = k
        self.S_init = None
        self.Sigma = None
        self.Sigma_R = None

        self._check_inputs(p, k, 'swap', noise)
        
        if noise == 'sph':
            MLE_keys = set(['C_MLE', 'W_MLE', 'S_MLE', 'mu_MLE', 'sigma_sq_MLE'])
        if noise == 'diag':
            MLE_keys = set(['C_MLE', 'W_MLE', 'S_MLE', 'mu_MLE', 'D_MLE'])

        # get initialization for the EM algorithm 
        if MLE_init is None or None in MLE_init.values() or set(MLE_init.keys()) != MLE_keys:

            X_init = np.where(np.isnan(X), np.nanmean(X, axis=0), X)  
            mu_init, Sigma_init = get_moments(X_init)
      
            if MLE_init is None:
                MLE_init = {}
                MLE_init['S_MLE'] = np.random.choice(np.arange(p), k, replace=False)
            elif len(MLE_init['S_MLE']) != k:
                raise ValueError("Initial subset must be of length k.")

            MLE_init, C_MLE_chol, C_MLE_inv = compute_MLE_from_selected_subset(Sigma_init, 
                                                                               MLE_init['S_MLE'], 
                                                                               noise=noise,
                                                                               mu_MLE=mu_init)
        self.MLE_init = MLE_init
        MLE = MLE_init.copy()
      
        iter = 0
        break_flag = False
        converged = False
        prev_log_likelihood = -np.inf
        curr_log_likelihood = compute_in_sample_mean_log_likelihood(compute_log_det_Sigma_MLE(MLE, C_MLE_chol), p)

        if noise == 'sph':
            objective = sph_pcss_objective
        if noise == 'diag':
            objective = diag_pcss_objective 
    
        while iter < max_em_iter and (not break_flag):

            # Perfom the E-step 
            m, Omega = compute_imputed_moments(X, MLE)
            m = np.mean(m, axis=0)
            Omega = np.mean(Omega, axis=0)
            Psi = Omega - np.outer(m, m)

            # Perform the M-step
            S, Sigma_R, _, _, colinearity_errors = swapping_subset_selection(Psi, 
                                                                             k,
                                                                             objective,
                                                                             max_iter=max_iter,
                                                                             S_init=MLE['S_MLE'],
                                                                             tol=tol,
                                                                             flag_colinearity=self.flag_colinearity)
            self.S = S
            
            if len(colinearity_errors) > 0:
                self.colinearity_errors = colinearity_errors
                self.MLE, self.C_MLE_chol, self.C_MLE_inv = None, None, None
                raise ValueError("Issues with colinearity.")
      
            MLE, C_MLE_chol, C_MLE_inv = compute_MLE_from_selected_subset(Psi, 
                                                                          S, 
                                                                          Sigma_R=Sigma_R,
                                                                          noise=noise, 
                                                                          mu_MLE=m)
      
            prev_log_likelihood = curr_log_likelihood
            curr_log_likelihood = compute_in_sample_mean_log_likelihood(compute_log_det_Sigma_MLE(MLE, C_MLE_chol), p)
            # Terminate if increase in observed likelihood is small enough
            if curr_log_likelihood - prev_log_likelihood < tau:
                converged = True
                break_flag = True

        self.colinearity_errors = []
        self.converged = converged
        self.MLE = MLE
        self.C_MLE_chol = C_MLE_chol
        self.C_MLE_inv = C_MLE_inv

  
