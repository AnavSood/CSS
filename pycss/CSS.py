from pycss.subset_selection import *
from pycss.utils import * 

class CSS():

    """
    Class for performing column subset selection.
    """
  
    def __init__(self):
        """
        Initialize `self.flag_colinearity` to `False`. 
        """
        self.flag_colinearity = False


    def _check_inputs(self, p, k, method):
        
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

    def select_subset_from_data(self, 
                                X, 
                                k, 
                                standardize=False, 
                                center=True,  
                                method='greedy',
                                tol=TOL,
                                max_iter=100,
                                S_init=None):
        
        """
        Given sample-level data, selects a subset. The following class variables 
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
        self.Sigma_R : np.array
            The `(p, p)`-shaped covariance matrix residual covariance from
            regressing out the selected variables. 
        self.S_init : np.array
            The `(k,)`-shaped subset used as the intialization for the swapping algorithm.
            If `method` is `'greedy'`, this will be `None`. 
        self.converged : bool
            Boolean value of whether or not the swapping algorithm converged.
            If `method` is `'greedy'`, this will be `None`. 
        self.colinearity_errors : list[ValueError]
            List of any errors with colinearity experienced during subset search. If this list 
            is non-empty then `self.S` and `self.Sigma_R` should be `None`. Also if this list
            is non-empty and `method` is `'swap'` then `converged` should be `False`.

        Parameters
        ----------
        X : np.array
            The `(n, p)`-shaped sample-level data matrix used for selection.
        k : int 
            The size of the subset to select. 
        standardize :  bool, default=`False`
            Whether or not to standardize the sample covariance so that the variables have unit variance. 
        center : bool, default='True' 
            Whether or not to center the data matrix `X` prior to computing its sample covariance.
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
    
        method=method.lower()
        n, p = X.shape
        self._check_inputs(p, k, method)
    
        if center:
            X -= np.mean(X, axis = 0)
    
        self.select_subset_from_cov(1/n * X.T @ X , 
                                    k, 
                                    standardize=standardize, 
                                    method=method, 
                                    tol=tol,
                                    max_iter=max_iter,
                                    S_init=S_init,
                                    from_data=True)
    
        self.n = n
        self.X = X



    def select_subset_from_cov(self, 
                               Sigma, 
                               k, 
                               standardize=False, 
                               method='greedy',
                               tol=TOL,
                               max_iter=100,
                               S_init=None,
                               from_data=False):

        """
        Given a covariance matrix, selects a subset. The following class variables 
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
        self.Sigma_R : np.array
            The `(p, p)`-shaped covariance matrix residual covariance from
            regressing out the selected variables. 
        self.S_init : np.array
            The `(k,)`-shaped subset used as the intialization for the swapping algorithm.
            If `method` is `'greedy'`, this will be `None`. 
        self.converged : bool
            Boolean value of whether or not the swapping algorithm converged.
            If `method` is `'greedy`, this will be `None`. 
        self.colinearity_errors : list[ValueError]
            List of any errors with colinearity experienced during subset search. If this list 
            is non-empty then `self.S` and `self.Sigma_R` should be `None`. Also if this list
            is non-empty and `method` is `'swap'` then `converged` should be `False`.

        Parameters
        ----------
        Sigma : np.array
            The `(p, p)`-shaped covariance matrix used for selection.
        k : int 
            The size of the subset to select. 
        standardize :  bool, default=`False`
            Whether or not to standardize the sample covariance so that the variables have unit variance. 
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
        from_data : bool, default=`False`
            Whether the function was called directly by the user (in which case it should be `False`) or
            if it was called internally. 
        """
        
        p = Sigma.shape[0]
        
        if not from_data:
            if Sigma.shape[0] != Sigma.shape[1]:
                raise ValueError("Sigma must be a square matrix.")
            method=method.lower()
            self._check_inputs(p, k, method)
            self.n = None
            self.X = None

        if standardize:
            Sigma = standardize_cov(Sigma)
    
        if method == 'greedy':
            S, Sigma_R, colinearity_errors = greedy_subset_selection(Sigma, 
                                                                     k, 
                                                                     css_objective, 
                                                                     tol=tol, 
                                                                     flag_colinearity=self.flag_colinearity)
            self.S_init = None
            self.converged = None

        if method == 'swap':
            S, Sigma_R, S_init, converged, colinearity_errors = swapping_subset_selection(Sigma, 
                                                                                          k,
                                                                                          css_objective,
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
            raise ValueError("Issues with colinearity.")
        
        self.colinearity_errors = []

    
    