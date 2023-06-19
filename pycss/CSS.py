from pycss.subset_selection import *
from pycss.utils import * 

class CSS():

    """
    Class for performing column subset selection.
    """
  
    def __init__(self):
        pass

    def _check_inputs(self, method, objective):
        
        """
        Ensures `'method'` and `'objective'` inputs are supported.   

        Parameters
        ----------
        method : str
            Method by which to select the subset. Must be `'greedy'`, `'swap'`, or `'exhaustive''. 
        objective : str
            Objective that cutoff correponds to. Must be  `'css'`,  or `'rsq''. 
        """

        if method not in {'greedy', 'swap', 'exhaustive'}:
            raise ValueError("Requested method not supported.")
        if method == 'greedy' and objective not in {'css', 'rsq'}:
            raise ValueError("Requested objective not supported.")

    def select_subset_from_data(self, 
                                X, 
                                k=None, 
                                method='greedy',
                                include=np.array([]),
                                exclude=np.array([]), 
                                cutoff=None,
                                objective='css',
                                center=True, 
                                standardize=True, 
                                num_inits=1,
                                max_iter=100,
                                S_init=None,
                                show_progress=True,
                                tol=TOL):
        
        """
        Given unit-level data, selects a subset. The following class variables 
        will be populated:

        self.p : int
            Number of variables selected from.
        self.Sigma : np.array
            The `(p, p)`-shaped covariance matrix used for selection. 
        self.S : np.array
            The selected subset. 
        self.Sigma_R : np.array
            The `(p, p)`-shaped covariance matrix residual covariance corresponding to `S`. 
        self.S_init : np.array[int]
            The `(k,)`-shaped subset used as the intialization for the swapping algorithm.
            If `method` is `'greedy'` or `'exhaustive'`, this will be `None`. 
        self.converged : bool
            Boolean value of whether or not the swapping algorithm converged.
            If `method` is `'greedy'` or `'exhaustive'``, this will be `None`. 
        self.include : np.array[int], default=np.array([])
            A list of variables that must be included. 
        self.exclude: np.array[int], default=np.array([])
            A list of variables that must not be included.

        Parameters
        ----------
        X : np.array
            The `(n, p)`-shaped unit-level data matrix used for selection.
        k : int, default None
            The size of the subset to select. Ignored when `method` is `'greedy'` and `cutoff` is not `None`. 
        method : str, default=`'greedy'`
            The method for selecting the subset. Options are `'greedy'`, `'swap'`, and `'exhaustive'`.
       include : np.array[int], default=np.array([])
            A list of variables that must be included. 
        exclude: np.array[int], default=np.array([])
            A list of variables that must not be included.
        cutoff: float
            Only relevant if 'method' is `'greedy'`, in which case variables are selected until a certain 
            cutoff based on `objective` is reached. 
        objective: str
            Only relevant if 'method' is `'greedy' and `cutoff` is not `None`. If `objective` is `'css'` then variables 
            are selected until css objective is <= `cutoff`.  If `objective` is `'rsq'` then variables are selected
            until the average R^2 for the remaining variables is >= `cutoff`. 
        center : bool, default='True' 
            Whether or not to center the data matrix `X` prior to computing its sample covariance.
        standardize :  bool, default=`False`
            Whether or not to standardize the sample covariance so that the variables have unit variance.
        num_inits : int, default=1
           Only relevant if `method` is `'swap'` and `S_init` is not `None`. Number of random initializations to try. 
        max_iter : int, default=100
            Only relevant if `method` is `'swap'`. Maximum number of iterations for the swapping algorithm to achieve
            convergence. If the algorithm does not achieve converge it returns the best subset till that point.
        S_init : np.array[int] 
            Only relevant if `method` is `'swap'`. Size `k` array of variables that serves as the initialization 
            for the swapping algorithm. If `None` then `num_inits` random initializations are tried.   
        show_progress: bool
            Only relevant if `method` is `'exhuastive'`. If `True`, informs the user of the number of subsets
            being searched over and shows a progress bar.
        tol : float, default=`TOL`
            Tolerance at which point we consider a variable to have zero variance. 
        """
        
        method = method.lower()
        objective = objective.lower()
        n, p = X.shape
        self._check_inputs(method, objective)
    
        if center:
            X -= np.mean(X, axis = 0)

        self.select_subset_from_cov(Sigma = 1/n * X.T @ X, 
                                    k=k, 
                                    method=method,
                                    include=include,
                                    exclude=exclude, 
                                    cutoff=cutoff,
                                    objective=objective,
                                    standardize=standardize, 
                                    num_inits=num_inits,
                                    max_iter=max_iter,
                                    S_init=S_init,
                                    show_progress=show_progress,
                                    tol=tol)

    def select_subset_from_cov(self, 
                               Sigma, 
                               k=None, 
                               method='greedy',
                               include=np.array([]),
                               exclude=np.array([]),
                               cutoff=None,
                               objective='css',
                               standardize=True, 
                               num_inits=1,
                               max_iter=100,
                               S_init=None,
                               show_progress=True,
                               tol=TOL):

        """
        Given a covariance, selects a subset. The following class variables 
        will be populated:

        self.p : int
            Number of variables selected from.
        self.Sigma : np.array
            The `(p, p)`-shaped covariance matrix used for selection. 
        self.S : np.array
            The selected subset. 
        self.Sigma_R : np.array
            The `(p, p)`-shaped covariance matrix residual covariance corresponding to `S`. 
        self.S_init : np.array[int]
            The `(k,)`-shaped subset used as the intialization for the swapping algorithm.
            If `method` is `'greedy'` or `'exhaustive'`, this will be `None`. 
        self.converged : bool
            Boolean value of whether or not the swapping algorithm converged.
            If `method` is `'greedy'` or `'exhaustive'``, this will be `None`. 
        self.include : np.array[int], default=np.array([])
            A list of variables that must be included. 
        self.exclude: np.array[int], default=np.array([])
            A list of variables that must not be included.

        Parameters
        ----------
        Sigma : np.array
            The `(p, p)`-shaped covariance matrix used for selection.
        k : int, default None
            The size of the subset to select. Ignored when `method` is `'greedy'` and `cutoff` is not `None`. 
        method : str, default=`'greedy'`
            The method for selecting the subset. Options are `'greedy'`, `'swap'`, and `'exhaustive'`.
        include : np.array[int], default=np.array([])
            A list of variables that must be included. 
        exclude: np.array[int], default=np.array([])
            A list of variables that must not be included.
        cutoff: float
            Only relevant if 'method' is `'greedy'`, in which case variables are selected until a certain 
            cutoff based on `objective` is reached. 
        objective: str
            Only relevant if 'method' is `'greedy' and `cutoff` is not `None`. If `objective` is `'css'` then variables 
            are selected until css objective is <= `cutoff`.  If `objective` is `'rsq'` then variables are selected
            until the average R^2 for the remaining variables is >= `cutoff`. 
        standardize :  bool, default=`False`
            Whether or not to standardize the sample covariance so that the variables have unit variance.
        num_inits : int, default=1
           Only relevant if `method` is `'swap'` and `S_init` is not `None`. Number of random initializations to try. 
        max_iter : int, default=100
            Only relevant if `method` is `'swap'`. Maximum number of iterations for the swapping algorithm to achieve
            convergence. If the algorithm does not achieve converge it returns the best subset till that point.
        S_init : np.array[int] 
            Only relevant if `method` is `'swap'`. Size `k` array of variables that serves as the initialization 
            for the swapping algorithm. If `None` then `num_inits` random initializations are tried.   
        show_progress: bool
            Only relevant if `method` is `'exhuastive'`. If `True`, informs the user of the number of subsets
            being searched over and shows a progress bar.
        tol : float, default=`TOL`
            Tolerance at which point we consider a variable to have zero variance. 
        """
        
        method = method.lower()
        objective = objective.lower()
        self._check_inputs(method, objective)

        p = Sigma.shape[0]
        
        if method == 'greedy' and objective == 'rsq':
            standardize = True 

        if standardize:
            Sigma = standardize_cov(Sigma)
    
        if method == 'greedy':
            
            if cutoff is not None:
                if objective == 'css':
                    cutoffs = cutoff
                if objective == 'rsq':
                    cutoffs = np.array([(p - i)*(1 - cutoff) for i in range(1, p + 1)]) 
            else:
                cutoffs=None

            S, Sigma_R = greedy_css(Sigma=Sigma,
                                    k=k,
                                    cutoffs=cutoffs,
                                    include=include,
                                    exclude=exclude,
                                    tol=TOL)
            self.S_init = None
            self.converged = None

        if method == 'swap':
            S, Sigma_R, S_init, converged  = swapping_css(Sigma=Sigma,
                                                          k=k,
                                                          num_inits=num_inits, 
                                                          max_iter=max_iter,
                                                          S_init=S_init,
                                                          include=include,
                                                          exclude=exclude,
                                                          tol=tol)
            self.S_init = S_init 
            self.converged = converged 
        
        if method == 'exhaustive':

            S, Sigma_R = exhaustive_css(Sigma=Sigma, 
                                        k=k, 
                                        include=include,
                                        exclude=exclude,
                                        show_progress=show_progress,
                                        tol=tol)
            
            self.S_init = None
            self.converged = None 


        self.Sigma = Sigma
        self.p = p
        self.include = include
        self.exclude = exclude 
        self.S = S
        self.Sigma_R = Sigma_R



    
    