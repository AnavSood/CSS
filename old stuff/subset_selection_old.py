import numpy as np
from choldate import cholupdate
from scipy.linalg import solve_triangular

TOL = 1e-10

def random_argmin(x):
    """
	Given an array `x` randomly returns index of one 
    of the minimum values.

	Parameters
	----------
	x : np.array
	    A input array.
	
    Returns 
	-------
	int
        The index of a minimimal value of `x`. 

	"""
    return np.random.choice(np.flatnonzero(x == x.min()))

def complement(n, idxs):

    """
	Returns all the integers in 0 to `n-1` not in `idxs`

	Parameters
	----------
	n : int
	    Upper bound (not inclusive) of the set of integers we consider.
    idxs : np.array
        Indices to take the complement of.  
	
    Returns 
	-------
	np.array
        All the integers in 0 to `n-1` not in `idxs`
	"""

    if len(idxs) == 0:
        return np.arange(n)
    return np.delete(np.arange(n), idxs)

def perm_in_place(Sigma, orig, perm, idx_order=None, row=True, col=True):
    
    """
	Given a square matrix `Sigma`, permutes rows and columns in `orig` according 
    to the permutation `perm` in place. 

	Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped matrix or a `(B, p, p)`-shaped batch of such matrices 
        which we permute the rows and columns of. 
    orig : np.array
        Indicies to apply permutation to.
    perm : np.array
        Permutaiton to apply. 
    idx_order : np.array, default=`None`.
        An ordering for the variables of Sigma, which, if not `None` will also be appropriately
        permuted in place. 
    row : bool, default=`True`
        Whether to permute the rows of Sigma.
    col : bool, default=`True`
        Whether to permute the cols of Sigma. 
	"""

    if len(Sigma.shape) == 2:
        if col:
            Sigma[:, orig] = Sigma[:, perm]
        if row:
            Sigma[orig, :] = Sigma[perm, :]
    if len(Sigma.shape) == 3:
        if col:
            Sigma[:, :, orig] = Sigma[:, :, perm]
        if row:
            Sigma[:, orig, :] = Sigma[:, perm, :]
    if idx_order is not None:
        idx_order[orig] = idx_order[perm]

def swap_in_place(Sigma, idxs1, idxs2, idx_order=None, disjoint=False, row=True, col=True):
    
    """
	Given a square matrix `Sigma`, swaps the locations of the rows and cols
    in `idxs1` with those in `idxs2` in place. If indices belong to both 
    `idxs1` and `idxs2` they don't move.

	Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped matrix or `(B, p, p)`-shaped batch of such matrices
        which we swap the rows and columns of.
    idxs1 : np.array
        Indices to swap with `idxs2`.
    idxs2 : np.array
        Indices to swap with `idxs1`
    idx_order : np.array, default=`None`.
        An ordering for the variables of Sigma, which, if not `None` will also be appropriately
        permuted in place. 
    dijoint : bool, default=`False`
        Whether `idxs1` and `idxs2` are disjoint.
    row : bool, default=`True`
        Whether to permute the rows of Sigma.
    col : bool, default=`True`
        Whether to permute the cols of Sigma. 
	"""
    
    if not disjoint:
        idxs1, idxs2 = set(idxs1), set(idxs2)
        idxs1, idxs2 = list(idxs1 - idxs2) , list(idxs2 - idxs1)
    if len(idxs1) == 0:
        return 
    perm_in_place(Sigma, np.concatenate([idxs1, idxs2]), np.concatenate([idxs2, idxs1]), idx_order=idx_order, row=row, col=col)

def regress_one_off_in_place(Sigma, j, tol=TOL):

    """
	Given covariance `Sigma` of some variables, computes the covariance 
    of said variables after regressing the `j`th one off of the others, in place.

	Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix.
    j : int
        Index of the variable to regress off.
    tol : float, default=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
	"""

    if Sigma[j, j] > tol:
        Sigma[:, :] = Sigma - np.outer(Sigma[:, j], Sigma[:, j])/Sigma[j, j]    
  
def regress_off_in_place(Sigma, S, tol=TOL):
    
    """
	Given covariance `Sigma` of some variables, computes the covariance 
    of said variables after regressing the variables in `S` off of 
    the others, in place.

	Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix.
    S : np.array
        Array of indices of variables to regress off.
    tol : float, default=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
	"""

    for j in S:
        regress_one_off_in_place(Sigma, j, tol)

def update_cholesky_after_removing_first(L):
    
    """
    Given the cholesky decomposition (lower triangular) `L` of a positive
    definite covariance matrix, cholesky of the same covariance matrix
    after removal of the first row and column. 
   
    Parameters
	----------
	L : np.array
	    A `(p, p)`-shaped Cholesky decomposition of positive definite covariance matrix. 
	
    Returns 
	-------
	np.array  
        A `(p-1, p-1)`-shaped Cholesky decomposition of the positive definite covariance matrix
        `L @ L.T` after removing its first row and column. 
	"""
    
    L_ = L[1:, 1:].copy()
    v = L[1:, 0].copy()
    L_ = L_.T
    cholupdate(L_, v)
    return L_.T

def update_cholesky_after_removing_last(L):
    """
    Given the cholesky decomposition (lower triangular) `L` of a positive
    definite covariance matrix, cholesky of the same covariance matrix
    after removal of the last row and column. 
   
    Parameters
	----------
	L : np.array
	    A `(p, p)`-shaped Cholesky decomposition of positive definite covariance matrix. 
	
    Returns 
	-------
	np.array  
        A `(p-1, p-1)`-shaped Cholesky decomposition of the positive definite covariance matrix
        `L @ L.T` after removing its last row and column. 
	"""
    
    shape = L.shape
    return L[:shape[0] - 1, :shape[1] - 1]

def update_cholesky_after_adding_last(L_, v):

    """
    Given the cholesky decomposition (lower triangular) `L` of a positive
    definite covariance matrix, cholesky of the same covariance matrix
    after adding `v` as the last row and column 
   
    Parameters
	----------
	L_ : np.array
	    A `(p, p)`-shaped Cholesky decomposition of positive definite covariance matrix. 
    v : np.array
        A `(p,)`-shaped array to add as the last row and column of the covariance matrix. 
	
    Returns 
	-------
	np.array  
        A `(p+1, p+1)`-shaped Cholesky decomposition of the positive definite covariance matrix
        `L @ L.T` after adding `v` as the last row and column. 
	"""
    
    p = len(v)
   
    if p == 1:
        return np.sqrt(np.array([v]))

    a = solve_triangular(L_, v[:(p-1)], lower=True) 
    d = np.sqrt(v[p-1] - np.inner(a, a))
    L = np.zeros((p, p))
    L[:(p-1), :(p-1)] = L_
    L[p-1, :(p-1)] = a
    L[p-1, p-1] = d

    return L

def solve_with_cholesky(L, M):
    
    """
    Given the cholesky decomposition (lower triangular) `L` of a positive
    definite covariance matrix Sigma, finds Sigma inverse times the matrix `M`
   
    Parameters
	----------
	L : np.array
	    A `(p, p)`-shaped Cholesky decomposition of positive definite covariance matrix. 
    M : np.array
        A `(p, )` or `(p, k)`-shaped array to multiply by the corresponding inverse covariance. 
	
    Returns 
	-------
	np.array  
        A `(p, )` or `(p, k)-shaped array which is the result of multipying Sigma inverse time `M`
	"""
    
    return solve_triangular(L.T, solve_triangular(L, M, lower=True), lower=False)

def is_invertible(Sigma, tol=TOL):

    """
    Given a covariance matrix `Sigma` checks that Sigma is invertible in a 
    very specific way. Particularly, it ensures that the residual variance of 
    each variable after regressing the others off is > `tol`.
   
    Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix. 
    tol : float, default=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
	
    Returns 
	-------
	bool 
        Whether or not `Sigma` is invertible by our criterion.
    Sigma_L : np.array
        Cholesky decomposition of Sigma if Sigma is invertible, otherwise None.
	"""
    
    p = Sigma.shape[0]

    try:
        Sigma_L = np.linalg.cholesky(Sigma)
    except:
        return False, None
    if p == 1 and Sigma[0, 0] > tol:
      return True, Sigma_L
    idxs = np.arange(p)
    perm = np.concatenate([np.arange(1, p), [0]])
    for i in range(p):
      Sigma_L_ = update_cholesky_after_removing_first(Sigma_L)
      v = Sigma[1:p, 0]
      if Sigma[0, 0] - v.T @ solve_with_cholesky(Sigma_L_, v) <= tol:
        return False, None
      perm_in_place(Sigma, idxs, perm) 
      Sigma_L = update_cholesky_after_adding_last(Sigma_L_, Sigma[p-1, :])
    return True, Sigma_L

def css_objective(Sigma_R, flag_colinearity=False, tol=TOL):
    
    """
    Given a current residual covariance matrix `Sigma_R` computes 
    CSS objective values for each variable in `Sigma_R`. The variable
    with the lowest objective value most reduces the CSS objective when
    added to the currently selected subset. 

    Parameters
	----------
	Sigma_R : np.array
	    Current residual covariance matrix.
    flag_colinearity : bool, default=`False`
        Whether or not to flag colinearity issues - not applicable for this objective. 
    tol : float, default=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
	
    Returns 
	-------
	np.array
        Objective values for each variable.
    (np.array, np.array)
        Indices where the residuals were below the specified tolerance - will alway be 
        empty for this objective. 
	"""
    diag = np.diag(Sigma_R)
    return -1 * np.divide(np.sum(np.square(Sigma_R), axis=1), diag, out=np.zeros_like(diag, dtype=float), where=(diag!=0)), (np.array([]), np.array([]))

def pcss_objective(Sigma_R, noise='sph', flag_colinearity=True, tol=TOL):

    """
    Given a current residual covariance matrix `Sigma_R` computes 
    Gaussian PCSS objective values for each variable in `Sigma_R`. The variable
    with the lowest objective value most increases the Gaussian PCSS likelihood 
    when added to the current subset. 

    Parameters
	----------
	Sigma_R : np.array
	    Current residual covariance matrix,
    noise : string
        Either `'sph'` or `'diag'` depending on if you want to maximize likelihood under the
        spherical or diagonal Gaussian PCSS model.
    flag_colinearity : bool, default=`True`
        Whether or not to flag colinearity issues.
    tol : float, defeault=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
	
    Returns 
	-------
	np.array
        Objective values for each variable.
    (np.array, np.array)
        Indices where the residuals were below the specified tolerance.
	"""

    diag = np.diag(Sigma_R)
    resids = diag - (1/diag)[:, None] * np.square(Sigma_R)
    np.fill_diagonal(resids, 1)
    if np.any(resids < tol):
      return None, np.where(resids < tol)
    
    if noise == 'sph':
      np.fill_diagonal(resids, 0)
      num_remaining = Sigma_R.shape[0] - 1
      objective_values = np.log(diag) + num_remaining * np.log(np.sum(resids, axis=1)) if num_remaining > 0 else np.log(diag) 
      return objective_values, (np.array([]), np.array([]))

    if noise == 'diag': 
      objective_values = np.log(diag) + np.sum(np.log(resids), axis=1)
      return objective_values, (np.array([]), np.array([]))

def sph_pcss_objective(Sigma_R, flag_colinearity=True, tol=TOL):
    '''
    Calls `pcss_objective` with `noise='sph'`.
    '''
    return pcss_objective(Sigma_R, noise='sph', flag_colinearity=flag_colinearity, tol=TOL)

def diag_pcss_objective(Sigma_R, flag_colinearity=True, tol=TOL):
    '''
    Calls `pcss_objective` with `noise='diag'`.
    '''
    return pcss_objective(Sigma_R, noise='diag', flag_colinearity=flag_colinearity, tol=TOL)
    
def populate_colinearity_errors(current_subset, additions=None, responses=None):
    
    """
    Given a current subset, additions, and responses, documents colinearity errors.

    Parameters
	----------
	current_subset : np.array
	    Currently selected subset. If `additions` and `responses` are `None` then currently selected 
        subset itself should be colinear. 
    additions : np.array, default='None'
        Variables which when added to the currently selected subset allow the selected subset to perfectly
        reconstruct the variable in `responses`
    responses : np.array, default='None'
        Variables which are perfectly predicted by the current subset 
        (plus the corresponding addition if `adittions` is not `None`).
	
    Returns 
	-------
	List[ValueError]
        List of ValueError objects detailing what the colinearity issues are. 
	"""

    if additions is None and responses is None:
        return [ValueError("The variables " + str(current_subset) + " are colinear.")]
    if additions is None:
        return [ValueError("The variables " + str(current_subset) + " perfectly predict " + str(responses))]
    else:
        errors = []
        for addition in set(additions):
            errors.append(ValueError("The variables " + str(np.concatenate([current_subset, np.array([addition])])) + 
                                     " perfectly predict " + str(responses[np.where(additions == addition)[0]])))
        return errors 

def greedy_subset_selection(Sigma, 
                            k,
                            objective,
                            tol=TOL,
                            flag_colinearity=False):
   
    """
    Given a covariance `Sigma`, a subset size `k`, and an objective function `objective`
    returns the greedily selected subset of variables `k` which minimize the objective. 

    Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix. 
    k : int
        Size of subset to search for.
    objective : Callable[np.array, bool, np.float]
        A python function which defines the objective. On each iteration the variable which 
        minimizes objective will be selected.
    tol : float, defeault=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
    flag_colinearity : bool, default=`True`
        Whether or not to flag colinearity issues and terminate upon their happening. 
	
    Returns 
	-------
	S : np.array
        The selected subset, in order it was selected.
    Sigma_R : np.array
        The `(p, p)`-shaped residual covariance matrix resulting from regressing the selected subset
        out of all the varibles (including themselves). 
    errors : List[ValueError] 
         List of ValueError objects detailing what the colinearity issues are. 
	"""
    
    S = -1 * np.ones(k).astype(int)
    Sigma_R = Sigma.copy()
    p = Sigma.shape[0]
    idx_order = np.arange(p)
    num_active=p

    for i in range(k):

      # subset to acvice variables
      Sigma_R_active = Sigma_R[:num_active, :num_active] 
      # compute objective values
      obj_vals, colinearity_error_idxs = objective(Sigma_R_active, flag_colinearity=flag_colinearity, tol=tol) 
      
      if len(colinearity_error_idxs[0]) > 0:
        return None, None, populate_colinearity_errors(S[:i], 
                                                       idx_order[colinearity_error_idxs[0]], 
                                                       idx_order[colinearity_error_idxs[1]])
      # select next variable
      j_star = random_argmin(obj_vals)
      S[i] = idx_order[j_star]

      # regress off selected variable
      regress_one_off_in_place(Sigma_R_active, j_star, tol=tol)
      
      # swap selected variable with last active position
      swap_in_place(Sigma_R, [j_star], [num_active - 1], idx_order=idx_order)
      # decrement number active 
      num_active -= 1
      
      # swap any variables with < tol variance to bottom and update num active
      if not flag_colinearity:
        zero_idxs = np.where(np.diag(Sigma_R_active)[:num_active] < tol)[0]
        num_zero_idxs = len(zero_idxs)
        idxs_to_swap = np.arange(num_active - num_zero_idxs, num_active)
        swap_in_place(Sigma_R, zero_idxs, idxs_to_swap, idx_order=idx_order)
        num_active -= num_zero_idxs
      
      # terminate early if all variables are explained
      if num_active == 0 and i != k - 1:
        break

    perm_in_place(Sigma_R, np.arange(p), np.argsort(idx_order))
    
    return S, Sigma_R, []

def swapping_subset_selection(Sigma, 
                              k,
                              objective,
                              max_iter=100,
                              S_init=None,
                              tol=TOL,
                              flag_colinearity=False):

    """
    Given a covariance `Sigma`, a subset size `k`, and an objective function `objective`
    returns the subset of variables `k` which minimize the objective selected by a gradient descent
    like iterative swapping algorithm. 

    Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix. 
    k : int
        Size of subset to search for.
    objective : Callable[np.array, bool, np.float]
        A python function which defines the objective. On each iteration the variable which 
        minimizes objective will be selected.
    max_iter : int, default=`100`
        Maximum number of iterations to run the swapping algorithm. If algorithm has not 
        converged within `max_iter` iterations, the algorithm will terminate and provide 
        results in its current state. In this case `converged` will be `False.
    S_init : np.array, default=`None`
        Intial subset to start the algorithm with. If not included, an initial subset is 
        selected uniformly randomly.  
    tol : float, default=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
    flag_colinearity : bool, default=`True`
        Whether or not to flag colinearity issues and terminate upon their happening. 
	
    Returns 
	-------
	S : np.array
        The selected subset, in order it was selected.
    Sigma_R : np.array
        The `(p, p)`-shaped residual covariance matrix resulting from regressing the selected subset
        out of all the varibles (including themselves). 
    S_init : np.array
        The inital subset that the algorithm starts with.
    converged : bool
        Whether the algorithm has converged. If `converged` is `False` and the `errors` list is non-empty 
        then `S` and `Sigma_R` must be `None`. 
    errors : List[ValueError]
         List of ValueError objects detailing what the colinearity issues are. 
	"""

    converged = False
    p = Sigma.shape[0]
    d = p-k
    idx_order = np.arange(p)
    
    if S_init is None:
        S_init = np.random.choice(idx_order, k, replace=False)
    elif len(S_init) != k:
        raise ValueError("Initial subset must be of length k.")
    
    Sigma_R = Sigma.copy()
    # these will always be the indices of the selected subset
    subset_idxs = np.arange(d, p)
    # swap initial variables to bottom of Sigma 
    swap_in_place(Sigma_R, subset_idxs, S_init, idx_order=idx_order)
    S = idx_order[d:].copy()
    Sigma_S = Sigma[:, S][S, :].copy()
    invertible, Sigma_S_L = is_invertible(Sigma_S) 
    
    if not invertible:
        return None, None, S_init, converged, populate_colinearity_errors(S)
    
    regress_off_in_place(Sigma_R, np.arange(d, p))
    zero_idxs = np.where(np.diag(Sigma_R)[:d] <= tol)[0]
    num_zero_idxs = len(zero_idxs)
    
    if flag_colinearity and num_zero_idxs > 0:
        return None, None, S_init, converged, populate_colinearity_errors(S, responses=idx_order[zero_idxs])

    # number of completed iterations
    N = 0
    # counter of how many consecutive times a selected variable was not swapped
    not_replaced = 0
    # permutation which shifts the last variable in the subset to the top of the subset
    subset_idxs_permuted = np.concatenate([subset_idxs[1:], np.array([subset_idxs[0]])])
    break_flag = False 

    while N < max_iter and (not break_flag):
        for i in range(k):
            S_0 = S[0]
            # Remove first variable from selected subset 
            T = S[1:]

            # Update cholesky after removing first variable from subset 
            Sigma_T_L = update_cholesky_after_removing_first(Sigma_S_L) 

            # Update residual covariance after removing first variable from subset
            v = Sigma[:, S_0] - Sigma[:, T] @ solve_with_cholesky(Sigma_T_L, Sigma[T, S_0]) if k > 1 else Sigma[:, S_0]
            reordered_v = v[idx_order]
            Sigma_R = Sigma_R + np.outer(reordered_v, reordered_v)/v[S_0]

            # Swap first variable from subset to to top of residual matrix 
            swap_in_place(Sigma_R, np.array([0]), np.array([d]), idx_order=idx_order)  
        
            # If not flag_colinearity, find indices of variables with zero variance
            if not flag_colinearity:
                zero_idxs = np.where(np.diag(Sigma_R)[:(d + 1)] <= tol)[0]
                num_zero_idxs = len(zero_idxs)
                # In residual matrix, swap variables with zero indices to right above currently selected subset (of size k-1)
                swap_in_place(Sigma_R, zero_idxs, np.arange(d + 1 - num_zero_idxs, d + 1), idx_order=idx_order)
            else:
                num_zero_idxs = 0
        
            # update num_active
            num_active = d + 1 - num_zero_idxs 

            # compute objectives and for active variables and find minimizers
            obj_vals, colinearity_error_idxs = objective(Sigma_R[:num_active, :num_active], flag_colinearity=flag_colinearity, tol=tol)

            if len(colinearity_error_idxs[0]) > 0:
                return None, None, S_init, converged, populate_colinearity_errors(S[:i], 
                                                                                  idx_order[colinearity_error_idxs[0]], 
                                                                                  idx_order[colinearity_error_idxs[1]])
        
            choices = np.flatnonzero(obj_vals == obj_vals.min())

            # if removed variable is a choice, select it, otherwise select a random choice
            if 0 in choices:
                not_replaced += 1
                j_star = 0 
            else:
                not_replaced = 0
                j_star = np.random.choice(choices)

            # Add new choice as the last variable in selected subset
            S_new = idx_order[j_star]
            S[:k-1] = S[1:]
            S[k-1] = S_new
            # Update cholesky after adding new choice as last variable in selected subset 
            Sigma_S_L = update_cholesky_after_adding_last(Sigma_T_L, Sigma[S_new, S])
            # In residual covariance, regress selected variable off the remaining
            #regress_one_off_in_place(Sigma_R[:(d+1), :(d+1)], j_star) #alternative option
            regress_one_off_in_place(Sigma_R[:num_active, :num_active], j_star)
            # In residual covariance swap new choice to top of selected subset and then permute selected subset
            # so the new choice is at the bottom, reflecting S
            swap_in_place(Sigma_R, np.array([j_star]), np.array([d]), idx_order=idx_order)
            perm_in_place(Sigma_R, subset_idxs,  subset_idxs_permuted, idx_order=idx_order)
        
            if not_replaced == k:
                converged=True
                break_flag=True
                break

        N += 1

    perm_in_place(Sigma_R, np.arange(p), np.argsort(idx_order))
    return S, Sigma_R, S_init, converged, []

def compute_MLE_from_selected_subset(Sigma, S, Sigma_R=None, noise='sph', mu_MLE=None):

    """
    Given a covariance `Sigma` and selected subset `S` computes and returns the maximum
    likelihood estimates under the Gaussian PCSS model. 

    Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix. 
    S : np.array
        The selected subset of size `k`. 
    Sigma_R : np.array, default=`None`
        The residual covariance after you regress the the selected variables in S
        out of all the variables.
    noise : str, default=`'sph'`
        Either 'sph' or 'diag' depending on if you want maximum likelihood estimates
        under the spherical or diagonal Gaussian PCSS model. 
    mu_MLE : np.array, default=`None`
        The sample mean of the observed data, if available. 
        
    Returns 
	-------
	MLE : Dict[str, np.array]
        A dictionary containing the maximum likelihood estimates. The keys for the spherical Gaussian
        PCSS model are `'mu_MLE'`, `'S_MLE'`, `'C_MLE'`, `'W_MLE'` and `'sigma_sq_MLE'`. The keys for the diagonal
        Gaussian PCSS model are the same, but `'sigma_sq_MLE'` is replaced by `'D_MLE'`.
    C_MLE_chol : np.array
        The `(k, k)`-shaped Cholesky decomposition of the covariance of the selected subset. 
    C_MLE_inv : np.array
       The `(k, k)`-shaped inverse of the covariance of the selected subset. 
	"""
    
    if Sigma_R is None:
        Sigma_R = Sigma.copy()
        regress_off_in_place(Sigma_R, S)

    p = Sigma.shape[0]
    k = len(S)

    S_ord = np.sort(S)
    S_ord_comp = complement(p, S_ord)

    C_MLE = Sigma[:, S_ord][S_ord, :]

    C_MLE_chol = np.linalg.cholesky(C_MLE) 
    C_MLE_inv = solve_with_cholesky(C_MLE_chol, np.eye(k))
    W_MLE = Sigma[S_ord_comp, :][:, S_ord] @ C_MLE_inv

    if noise == 'sph':
        sigma_sq_MLE = np.sum(np.diag(Sigma_R)[S_ord_comp]) /(p - k) 
    if noise == 'diag':
        D_MLE = np.diag(Sigma_R)[S_ord_comp]
    
    if noise == 'sph':
        MLE = {'mu_MLE': mu_MLE,
               'S_MLE': S_ord,
               'C_MLE': C_MLE,
               'W_MLE': W_MLE,
               'sigma_sq_MLE': sigma_sq_MLE}
    elif noise == 'diag':
        MLE = {'mu_MLE': mu_MLE,
               'S_MLE': S_ord,
               'C_MLE': C_MLE,
               'W_MLE': W_MLE,
               'D_MLE': D_MLE}
    
    return MLE, C_MLE_chol, C_MLE_inv

def noise_from_MLE(MLE):

    """
    Given dictionary of MLEs, returns 'sph' if 'sigma_sq_MLE' is a key in MLE
    and 'diag' if 'D_MLE' is a key in MLE. 
    """
    
    if 'sigma_sq_MLE' in MLE.keys(): 
        noise = 'sph'
    if 'D_MLE' in MLE.keys():
        noise = 'diag'
    return noise

def compute_log_det_Sigma_MLE(MLE, C_MLE_chol=None):
  
    """
    Returns log determinant of the MLE for Sigma implied by `MLE`, a dictionary of 
    maximum likelihood estimates under the spherical or digaonal Gaussian PCSS model. 
  
    Parameters
    ----------
	MLE : Dict[str, np.array]
        A dictionary containing the maximum likelihood estimates. The keys for the spherical Gaussian
        PCSS model are `'mu_MLE'`, `'S_MLE'`, `'C_MLE'`, `'W_MLE'` and `'sigma_sq_MLE'`. The keys for the diagonal
        Gaussian PCSS model are the same, but `'sigma_sq_MLE'` is replaced by `'D_MLE'`. 
    C_MLE_chol : np.array, default=`None` 
        The Cholesky decomposition of MLE for C
        
    Returns 
	-------
	float
        The log determinant of the MLE for Sigma
	"""

    noise = noise_from_MLE(MLE)

    if C_MLE_chol is None:
        C_MLE_chol = np.linalg.cholesky(MLE['C_MLE']) 
  
    if noise == 'sph':
        return np.sum(np.log(np.square(np.diag(C_MLE_chol)))) + MLE['W_MLE'].shape[0] * np.log(MLE['sigma_sq_MLE'])
    if noise == 'diag':
        return np.sum(np.log(np.square(np.diag(C_MLE_chol)))) + np.sum(np.log(MLE['D_MLE']))
    
def compute_Sigma_MLE_inv(MLE, C_MLE_inv=None):
    
    """
    Returns the inverse of the MLE for `(p, p)`-shaped Sigma implied by `MLE`, a dictionary of 
    maximum likelihood estimates under the spherical or digaonal Gaussian PCSS model, 
    in blocks.  
  
    Parameters
    ----------
	MLE : Dict[str, np.array]
        A dictionary containing the maximum likelihood estimates. The keys for the spherical Gaussian
        PCSS model are `'mu_MLE'`, `'S_MLE'`, `'C_MLE'`, `'W_MLE'` and `'sigma_sq_MLE'`. The keys for the diagonal
        Gaussian PCSS model are the same, but `'sigma_sq_MLE'` is replaced by `'D_MLE'`. 
    C_MLE_inv : np.array, default=`None` 
        The `(k, k)`-shaped inverse of the MLE for C
        
    Returns 
	-------
    top_left_block : np.array
        The `(k, k)`-shaped top left block of the inverse of the MLE for Sigma, supposing the variables are sorted 
        such that the selected subset is ordered and comes first, and then the remaining variables come in order after.
    bottom_left_block : np.array
        The `(p-k, k)`-shaped bottom left block of the inverse of the MLE for Sigma, supposing the same ordering for the variables
        as above.  
    bottom_right_block : np.array
        The `(p-k,)`-shaped diagonal of the bottom right block of the inverse of the MLE for Sigma, supposing the same ordering 
        for the variables as above.  
	"""
  
    noise = noise_from_MLE(MLE)

    if C_MLE_inv is None:
        C_MLE_inv = np.linalg.inv(MLE['C_MLE'])

    if noise == 'sph':
        bottom_right_block = 1/MLE['sigma_sq_MLE'] * np.ones(MLE['W_MLE'].shape[0])
    if noise == 'diag':
        bottom_right_block = 1/MLE['D_MLE']
  
    bottom_left_block = -1 * bottom_right_block[:, None] * MLE['W_MLE']
    top_left_block = C_MLE_inv -  MLE['W_MLE'].T @ bottom_left_block

    return top_left_block, bottom_left_block, bottom_right_block 

def compute_Sigma_MLE_chol(MLE, C_MLE_chol=None):

    """
    Returns the Cholesky of the MLE for `(p, p)`-shaped Sigma implied by `MLE`, a dictionary of 
    maximum likelihood estimates under the spherical or digaonal Gaussian PCSS model, in blocks.  
  
    Parameters
    ----------
	MLE : Dict[str, np.array]
        A dictionary containing the maximum likelihood estimates. The keys for the spherical Gaussian
        PCSS model are `'mu_MLE'`, `'S_MLE'`, `'C_MLE'`, `'W_MLE'` and `'sigma_sq_MLE'`. The keys for the diagonal
        Gaussian PCSS model are the same, but `'sigma_sq_MLE'` is replaced by `'D_MLE'`. 
    C_MLE_chol : np.array, default=`None` 
        The `(k, k)`-shaped cholesky of the MLE for C
        
    Returns 
	-------
    top_left_block : np.array
        The `(k, k)`-shaped top left block of the Cholesky of the MLE for Sigma, supposing the variables are sorted 
        such that the selected subset is ordered and comes first, and then the remaining variables come in order after.
    bottom_left_block : np.array
        The `(p-k, k)`-shaped bottom left block of the Cholesky of the MLE for Sigma, supposing the same ordering for the variables
        as above.  
    bottom_right_block : np.array
        The `(p-k,)`-shaped diagonal of the bottom right block of the Cholesky of the MLE for Sigma, supposing the same ordering 
        for the variables as above.  
	"""

    noise = noise_from_MLE(MLE)
  
    if C_MLE_chol is None:
        C_MLE_chol = np.linalg.cholesky(MLE['C_MLE'])
  
    if noise == 'sph':
        bottom_right_block = np.sqrt(MLE['sigma_sq_MLE']) * np.ones(MLE['W_MLE'].shape[0])
    if noise == 'diag':
        bottom_right_block = np.sqrt(MLE['D_MLE'])

    top_left_block = C_MLE_chol.copy()
    bottom_left_block = MLE['W_MLE'] @ C_MLE_chol

    return top_left_block, bottom_left_block, bottom_right_block 

def compute_in_sample_mean_log_likelihood(p, log_det_Sigma_MLE):
    """
    Computes the mean in sample log-likelihood of Gaussian data given the
    log determinant of the maximum likelihood estimate of the covariance
    and the dimension of the data.
  
    Parameters
    ----------
   	p : int
        Dimension of the covariance matrix. 
    log_det_Sigma_MLE : float
        The log determinant of the maximum likelihood estimate of the covariance. 
        
    Returns 
	-------
    float
        The mean in sample log-likelihood of the data under the Gaussian model. 
	"""
    return -1/2*(p * np.log(2 * np.pi) + p + log_det_Sigma_MLE)