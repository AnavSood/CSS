import warnings
import numpy as np
import itertools
import math 
import tqdm 
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
    in `idxs1` with those in `idxs2` in place. The i-th index in `idxs2` is
    gauarnteed to end up where the i-th index of `idx1` was. 

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
    
    if len(idxs1) == 0:
        return
    if disjoint:
        orig = np.concatenate([idxs1, idxs2])
        perm = np.concatenate([idxs2, idxs1])
    else:
        union = set(idxs1).union(idxs2)
        orig = np.concatenate([idxs1, list(set(union) - set(idxs1)) ]).astype(int)
        perm = np.concatenate([idxs2, list(set(union) - set(idxs2)) ]).astype(int)
    perm_in_place(Sigma, orig, perm, idx_order=idx_order, row=row, col=col)

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

def regress_one_off(Sigma, j, tol=TOL):
    
    '''
    Same as `regress_one_off_in_place` but not in place
    '''
    
    if Sigma[j, j] > tol:
        return Sigma - np.outer(Sigma[:, j], Sigma[:, j])/Sigma[j, j]   
    else:
        return Sigma 

def regress_off(Sigma, S, tol=TOL):
    
    '''
    Same as `regress_off_in_place` but not in place
    '''

    for j in S:
        Sigma = regress_one_off(Sigma, j, tol=tol)
    return Sigma 

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


def css_score(Sigma_R, tol=TOL):
    
    """
    Given a current residual covariance matrix `Sigma_R` computes 
    scores for each variable `Sigma_R`. The variable with the lowest 
    score will most reduce the CSS objective value when added to the 
    currently selected subset. 

    Parameters
	----------
	Sigma_R : np.array
	    A `(p, p)`-shaped current residual covariance matrix.
    tol : float, default=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
	
    Returns 
	-------
	np.array
        A  `(p,)`-shaped array of scores for each variable.
	"""
    
    diag = np.diag(Sigma_R)
    return -1 * np.divide(np.sum(np.square(Sigma_R), axis=1), diag, out=np.zeros_like(diag, dtype=float), where=(diag > tol))

def check_greedy_css_inputs(Sigma, k, cutoffs, include, exclude, tol):

    """
    Checks if the inputs to `greedy_css` meet the required specifications.
	"""

    n, p = Sigma.shape 

    if not n == p:
        raise ValueError("Sigma must be a square matrix.")
  
    if k is None and cutoffs is None:
        raise ValueError("One of k or cutoff must not be None.")
  
    if k is not None and cutoffs is not None:
        raise ValueError("Only one of k or cutoff can be None.")

    if cutoffs is not None:
        if (isinstance(cutoffs, (list, np.ndarray)) and not len(cutoffs) == p) or (not isinstance(cutoffs, (list, np.ndarray)) and  not isinstance(cutoffs, (int, np.integer, float)) ):
            raise ValueError("Cutoffs must be a single value or length p.")

    if k is not None and not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer.")
    if k is not None and (k <= 0 or k > p):
        raise ValueError("k must be > 0 and <= p.")

    set_include = set(include)
    set_exclude = set(exclude)
    if not isinstance(include, np.ndarray) or (include.dtype != 'int' and len(include) > 0) or not set_include.issubset(np.arange(p)): 
        raise ValueError('Include must be a numpy array of integers from 0 to p-1.')
    if not isinstance(exclude, np.ndarray) or (exclude.dtype != 'int' and len(exclude) > 0) or not set_exclude.issubset(np.arange(p)):
        raise ValueError('Exclude must be a numpy array of integers from 0 to p-1.')
    if len(set_exclude.intersection(set_include)) > 0:
        raise ValueError("Include and exclude must be disjoint.")
        
    if len(exclude) == p:
        raise ValueError("Cannot exclude everything.")
    if k is not None and len(include) > k:
        raise ValueError("Cannot include more than k.")
    if k is not None and len(exclude) > p - k:
        raise ValueError("Cannot exclude more than p-k.")

    return


def greedy_css(Sigma,
               k=None,
               cutoffs=None,
               include=np.array([]),
               exclude=np.array([]),
               tol=TOL):

    """
    Given a '(p, p)`-shaped covariance matrix `Sigma` finds the greedily
    selected subset of size k according to the CSS objective, or a large 
    enough greedily selected subset so that the CSS objective is sufficiently
    small. 

    Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix to perform subset selection with.
    k : int, default=`None`
        If not `None`, the number of variables to greedily select. Exactly one of
        `k` and `cutoffs` can and must be `None`. 
    cutoffs : float OR np.array, default=None
        If a single value then we greedily select variables until the CSS objective value 
        is <= this cutoff. If a `(p, )`-shaped array then the i-th entry is used as the cutoff for 
        the greedily selected size-i subset  Exactly one of`k` and `cutoffs` can and must be `None`. 
    include : np.array[int], default=np.array([])
        A list of variables that must be included. 
    exclude: np.array[int], default=np.array([])
        A list of variables that must not be included.
    tol : float, default=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
	
    Returns 
	-------
	S : np.array
        The greedily selected subset. 
    Sigma_R : np.array
        The '(p, p)'-shaped residual covariance corresponding to `S`. 

	"""

    check_greedy_css_inputs(Sigma=Sigma,
                            k=k, 
                            cutoffs=cutoffs, 
                            include=include, 
                            exclude=exclude, 
                            tol=tol)

    Sigma_R = Sigma.copy()
    p = Sigma.shape[0]
    S = -1 * np.ones(p).astype(int)

    if isinstance(cutoffs, (int, np.integer, float)):
        cutoffs = cutoffs * np.ones(p)

    idx_order = np.arange(p)
    num_active = p

    selected_enough = False
    num_selected = 0

    while not selected_enough:

        # subset to acvice variables
        Sigma_R_active = Sigma_R[:num_active, :num_active]

        if num_selected < len(include):
            j_star = np.where(idx_order == include[num_selected])[0][0]

            # If the include variables are colinear return a colinearity error
            if j_star > num_active - 1:
                warnings.warn("Variables " + str(include[:num_selected + 1]) + " that have been requested to be included are colinear.")
                S[num_selected] = idx_order[j_star]
                num_selected += 1
                continue

        else:
            # compute objective values
            obj_vals = css_score(Sigma_R_active, tol=tol)

            # set the exclude objective values to infinity
            obj_vals[np.in1d(idx_order[:num_active], exclude)] = np.inf
            # select next variable
            j_star = random_argmin(obj_vals)

        S[num_selected] = idx_order[j_star]
        num_selected += 1

        # regress off selected variable
        regress_one_off_in_place(Sigma_R_active, j_star, tol=tol)

        # swap so selected variable is in last active position
        swap_in_place(Sigma_R, [num_active - 1], [j_star], idx_order=idx_order)
        # decrement number active
        num_active -= 1

        # swap any variables with < tol variance to bottom and update num active
        zero_idxs = np.where(np.diag(Sigma_R_active)[:num_active] < tol)[0]
        num_zero_idxs = len(zero_idxs)
        idxs_to_swap = np.arange(num_active - num_zero_idxs, num_active)
        swap_in_place(Sigma_R, idxs_to_swap, zero_idxs, idx_order=idx_order)  # order of idxs1 and idxs2 is important!
        num_active -= num_zero_idxs

        # continue if not enough included
        if num_selected < len(include):
            continue
        # terminate if user requested k and k have been selected
        if k is not None and num_selected == k:
            selected_enough = True
        # terminate if below user's cutoff
        if cutoffs is not None and np.trace(Sigma_R) <= cutoffs[num_selected - 1]:
            selected_enough = True
         # terminate early if no variables left 
        if set(idx_order[:num_active]).issubset(exclude) and not selected_enough:
            if cutoffs is not None:
                warnings.warn("Cutoff was not obtained by the selected subset, but no more variables can be added.")
            if k is not None:
                warnings.warn("A smaller subset sufficiently explained all the not excluded variables.")
            selected_enough = True

    perm_in_place(Sigma_R, np.arange(p), np.argsort(idx_order))

    return S[:num_selected], Sigma_R

def check_swapping_css_inputs(Sigma,
                              k,
                              num_inits,
                              max_iter,
                              S_init,
                              include,
                              exclude,
                              tol):
    
    """
    Checks if the inputs to `swapping_css` meet the required specifications.
	"""
    
    n, p = Sigma.shape 

    if not n == p:
        raise ValueError("Sigma must be a square matrix.")

    if not isinstance(k, (int, np.integer)) or k <= 0 or k > p:
        raise ValueError("k must be an integer > 0 and <= p.")
    
    set_include = set(include)
    set_exclude = set(exclude)
    if not isinstance(include, np.ndarray) or (include.dtype != 'int' and len(include) > 0) or not set_include.issubset(np.arange(p)): 
        raise ValueError('Include must be a numpy array of integers from 0 to p-1.')
    if not isinstance(exclude, np.ndarray) or (exclude.dtype != 'int' and len(exclude) > 0) or not set_exclude.issubset(np.arange(p)):
        raise ValueError('Exclude must be a numpy array of integers from 0 to p-1.')
    if len(set_exclude.intersection(set_include)) > 0:
        raise ValueError("Include and exclude must be disjoint.")
    
    if S_init is not None:
        if not isinstance(S_init, np.ndarray) or S_init.dtype != 'int' or len(set(S_init)) != k or (not set(S_init).issubset(np.arange(p))):
            raise ValueError("S_init must be a numpy array of k integers from 0 to p-1 inclusive.")
        if not set(include).issubset(S_init):
            raise ValueError("Include must be a subset of S_init.")
        if len(set(exclude).intersection(S_init)) > 0:
            raise ValueError("S_init cannot contain any elements in exlcude.")

    if len(include) > k:
        raise ValueError("Cannot include more than k.")
    if len(exclude) > p - k:
        raise ValueError("Cannot exclude more than p-k.")

    
def swapping_css_with_init(Sigma,
                           S_init,
                           max_iter,
                           include,
                           exclude,
                           tol=TOL):
    
    '''
    Performs swapping CSS with a particular initialization. See `swapping_CSS` for a description 
    of inputs. 
    '''
    
    k = len(S_init)
    p = Sigma.shape[0]
    d = p-k
    include_set = set(include)

    idx_order = np.arange(p)

    Sigma_R = Sigma.copy()
    # these will always be the indices of the selected subset
    subset_idxs = np.arange(d, p)
    # swap initial variables to bottom of Sigma
    swap_in_place(Sigma_R, subset_idxs, S_init, idx_order=idx_order) #order of idxs1 and idxs2 is important!
    S = idx_order[d:].copy()
    Sigma_S = Sigma[:, S][S, :]
    invertible, Sigma_S_L = is_invertible(Sigma_S)   

    if not invertible:
        return None, None, None 

    #regress out original subset
    regress_off_in_place(Sigma_R, np.arange(d, p), tol=tol)

    # number of completed iterations
    N = 0
    # counter of how many consecutive times we have chose not to swap 
    not_replaced = 0
    # permutation which shifts the last variable in the subset to the top of the subset
    subset_idxs_permuted = np.concatenate([subset_idxs[1:], np.array([subset_idxs[0]])])
    converged = False

    while N < max_iter and (not converged):
        for i in range(k):
            S_0 = S[0]

            # Update cholesky after removing first variable from subset
            Sigma_T_L = update_cholesky_after_removing_first(Sigma_S_L)

            if S_0 not in include_set:
            
                # Subset with first variable removed  from selected subset
                T = S[1:].copy()

                # Update residual covariance after removing first variable from subset
                v = Sigma[:, S_0] - Sigma[:, T] @ solve_with_cholesky(Sigma_T_L, Sigma[T, S_0]) if k > 1 else Sigma[:, S_0]
                reordered_v = v[idx_order]
                Sigma_R = Sigma_R + np.outer(reordered_v, reordered_v)/v[S_0]
                
                # Swap first variable from subset to to top of residual matrix
                swap_in_place(Sigma_R, np.array([0]), np.array([d]), idx_order=idx_order)

                # find indices of variables with zero variance
                zero_idxs = np.where(np.diag(Sigma_R)[:(d + 1)] <= tol)[0]
                num_zero_idxs = len(zero_idxs)
                # In residual matrix, swap variables with zero indices to right above currently selected subset (of size k-1)
                swap_in_place(Sigma_R, np.arange(d + 1 - num_zero_idxs, d + 1), zero_idxs, idx_order=idx_order) #order of idxs1 and idxs2 is important!

                # update num_active
                num_active = d + 1 - num_zero_idxs

                # compute objectives and for active variables and find minimizers
                obj_vals = css_score(Sigma_R[:num_active, :num_active], tol=tol)

                # set the objective value to infinity for the excluded variables
                obj_vals[np.in1d(idx_order[:num_active], exclude)] = np.inf

                choices = np.flatnonzero(obj_vals == obj_vals.min())

                # if removed variable is a choice, select it, otherwise select a random choice
                if 0 in choices:
                    not_replaced += 1
                    j_star = 0
                else:
                    not_replaced = 0
                    j_star = np.random.choice(choices)
                
                S_new = idx_order[j_star]
                
                # In residual covariance, regress selected variable off the remaining
                #regress_one_off_in_place(Sigma_R[:(d+1), :(d+1)], j_star) #alternative option
                regress_one_off_in_place(Sigma_R[:num_active, :num_active], j_star)
                # In residual covariance swap new choice to top of selected subset 
                swap_in_place(Sigma_R, np.array([d]), np.array([j_star]), idx_order=idx_order)
              
            else:
                S_new = S_0 
            
            # Add new choice as the last variable in selected subset
            S[:k-1] = S[1:]
            S[k-1] = S_new

            # Update cholesky after adding new choice as last variable in selected subset
            Sigma_S_L = update_cholesky_after_adding_last(Sigma_T_L, Sigma[S_new, S])

            # permute first variables in selected subset to the last variable in the residual matrix
            perm_in_place(Sigma_R, subset_idxs,  subset_idxs_permuted, idx_order=idx_order)
                
            if not_replaced == k - len(include):
                converged=True
                break
       
        N += 1

    perm_in_place(Sigma_R, np.arange(p), np.argsort(idx_order))
    
    return S, Sigma_R, converged 

def swapping_css(Sigma,
                 k,
                 num_inits=1, 
                 max_iter=100,
                 S_init=None,
                 include=np.array([]),
                 exclude=np.array([]),
                 tol=TOL):
   
    """
    Given a `(p, p)`-covariance matrix `Sigma` uses iterative swapping to 
    approximately find a size k subset that minimizes the CSS objective. The swapping
    happens in the same order as the initial subset. 

    Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix to perform subset selection with.
    k : int, default=`None`
        The number of variables to select. 
    max_iter : int, default=100
        Maximum number of iterations for the swapping algorithm to achieve convergence. If 
        the algorithm does not achieve converge it returns the best subset till that point.
    num_inits : int, default=1
        Number of random initializations to try. Only relevant if `S_init` is not `None`.
    S_init : np.array[int] 
        Size `k` array of variables that serves as the initialization for the swapping algorithm.
        If `None` then `num_inits` random initializations are tried.   
    include : np.array[int], default=np.array([])
        A list of variables that must be included. 
    exclude: np.array[int], default=np.array([])
        A list of variables that must not be included.
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

    check_swapping_css_inputs(Sigma=Sigma,
                              k=k,
                              num_inits=num_inits,
                              max_iter=max_iter,
                              S_init=S_init,
                              include=include,
                              exclude=exclude,
                              tol=tol)
    
    best_converged = None
    best_S = None
    best_S_init = None 
    best_Sigma_R = None
    best_obj_val = np.inf 
    not_include = np.array([idx for idx in complement(Sigma.shape[0], include) if idx not in set(exclude)])
    
    if len(include) > 0:
        invertible, _ = is_invertible(Sigma[include, :][:, include], tol=tol)
        if not invertible:
            warnings.warn("The variables requested to be included are colinear.")
            return best_S, best_Sigma_R, best_S_init, best_converged   
    
    no_initialization = (S_init is None)
    if not no_initialization:
        num_inits = 1

    for _ in range(num_inits):
        if no_initialization:
            S_init = np.concatenate([include, np.random.choice(not_include, k-len(include), replace=False)]).astype(int)

        S, Sigma_R, converged  = swapping_css_with_init(Sigma=Sigma,
                                                        S_init=S_init,
                                                        max_iter=max_iter, 
                                                        include=include,
                                                        exclude=exclude,
                                                        tol=TOL)
        if S is None:
            continue 
      
        obj_val = np.trace(Sigma_R)
        if obj_val < best_obj_val:
            best_obj_val = obj_val 
            best_S = S.copy()
            best_S_init = S_init.copy()
            best_Sigma_R = Sigma_R.copy()
            best_converged = converged 

    if best_S is None:
        warnings.warn("All the initializations tried were colinear.")
    return best_S, best_Sigma_R, best_S_init, best_converged

def check_exhuastive_css_inputs(Sigma, 
                                k, 
                                include,
                                exclude,
                                show_progress,
                                tol):
    
    """
    Checks if the inputs to `exhaustive_css` meet the required specifications.
	"""
    
    n, p = Sigma.shape 

    if not n == p:
        raise ValueError("Sigma must be a square matrix.")

    if not isinstance(k, (int, np.integer)) or k <= 0 or k > p:
        raise ValueError("k must be an integer > 0 and <= p.")
        
    set_include = set(include)
    set_exclude = set(exclude)
    if not isinstance(include, np.ndarray) or (include.dtype != 'int' and len(include) > 0) or not set_include.issubset(np.arange(p)): 
        raise ValueError('Include must be a numpy array of integers from 0 to p-1.')
    if not isinstance(exclude, np.ndarray) or (exclude.dtype != 'int' and len(exclude) > 0) or not set_exclude.issubset(np.arange(p)):
        raise ValueError('Exclude must be a numpy array of integers from 0 to p-1.')
    if len(set_exclude.intersection(set_include)) > 0:
        raise ValueError("Include and exclude must be disjoint.")

    if len(include) > k:
        raise ValueError("Cannot include more than k.")
    if len(exclude) > p - k:
        raise ValueError("Cannot exclude more than p-k.")
    

def exhaustive_css(Sigma, 
                   k, 
                   include=np.array([]),
                   exclude=np.array([]),
                   show_progress=True,
                   tol=TOL):
    
    """
    Given a `(p, p)`-covariance matrix `Sigma` exhaustively searches
    for the size k that minimizes the CSS objective. 

    Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix to perform subset selection with.
    k : int, default=`None`
        The number of variables to select.   
    include : np.array[int], default=np.array([])
        A list of variables that must be included. 
    exclude: np.array[int], default=np.array([])
        A list of variables that must not be included.
    show_progress : bool
        If `True`, informs the user of the number of subsets being searched over
        and shows a progress bar.
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
    
    p = Sigma.shape[0]

    check_exhuastive_css_inputs(Sigma=Sigma, 
                                k=k, 
                                include=include,
                                exclude=exclude,
                                show_progress=show_progress,
                                tol=tol)

    best_S = None
    best_Sigma_R = None
    best_obj_val = np.inf 

    options = np.array([idx for idx in np.arange(p) if idx not in np.concatenate([include, exclude])])
    to_add = k - len(include)
    S = np.concatenate([include, -1*np.ones(to_add)]).astype(int)

    if show_progress:
        print("Iterating over " + str(math.comb(len(options), to_add)) + " different subsets...")
        iterator = tqdm.tqdm(itertools.combinations(options, to_add))
    else:
        iterator = itertools.combinations(options, to_add)

    for  remaining in iterator:
        S[len(include):] = np.array(remaining).astype(int)
        Sigma_R = regress_off(Sigma, S, tol=tol)
        obj_val = np.trace(Sigma_R)
        if obj_val < best_obj_val:
            best_obj_val = obj_val
            best_S = S.copy()
            best_Sigma_R = Sigma_R.copy()
    
    return best_S, best_Sigma_R 
    
    
def subset_factor_score(Sigma_R, tol=TOL):

    """
    Given a current residual covariance matrix `Sigma_R` computes 
    scores for each variable `Sigma_R`. The variable with the lowest 
    score will result in the smallest test statistic when added to the 
    currently selected subset. 

    Parameters
	----------
	Sigma_R : np.array
	    A `(p, p)`-shaped current residual covariance matrix.
    tol : float, default=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
	
    Returns 
	-------
	np.array
        A `(p,)-shaped` array of scores for each variable.
    (np.array, np.array)
        A tuple of arrays documenting any colinearity. Adding the i-th entry of the first array
        to the current subset would allow the user to perfectly explain the i-th entry of the
        second array. 
	"""

    diag = np.diag(Sigma_R)
    resids = diag - (1/diag)[:, None] * np.square(Sigma_R)
    np.fill_diagonal(resids, 1)

    if np.any(resids < tol):
        return None, np.where(resids < tol)
    else:
        objective_values = np.log(diag) + np.sum(np.log(resids), axis=1)
        return objective_values, (np.array([]), np.array([]))

def check_greedy_subset_factor_inputs(Sigma, cutoffs, include, exclude, tol):

    """
    Checks if the inputs to `greedy_subset_factor_selection` meet the required specifications.
	"""

    n, p = Sigma.shape 

    if not n == p:
        raise ValueError("Sigma must be a square matrix.")
  
    if not isinstance(cutoffs, (list, np.ndarray)) or not len(cutoffs) == p + 1:
        raise ValueError("Must provide p + 1 cutoffs.")

    set_include = set(include)
    set_exclude = set(exclude)
    if not isinstance(include, np.ndarray) or (include.dtype != 'int' and len(include) > 0) or not set_include.issubset(np.arange(p)): 
        raise ValueError('Include must be a numpy array of integers from 0 to p-1.')
    if not isinstance(exclude, np.ndarray) or (exclude.dtype != 'int' and len(exclude) > 0) or not set_exclude.issubset(np.arange(p)):
        raise ValueError('Exclude must be a numpy array of integers from 0 to p-1.')
    if len(set_exclude.intersection(set_include)) > 0:
        raise ValueError("Include and exclude must be disjoint.")

    if not isinstance(tol, float):
        raise ValueError("tol must be a float.")

    return

def greedy_subset_factor_selection(Sigma,
                                   cutoffs,
                                   include=np.array([]),
                                   exclude=np.array([]),
                                   tol=TOL,
                                   ):
    
    """
    Given a '(p, p)`-shaped covariance matrix `Sigma`, greedily selects a subset 
    until the log determinant of the diagonal of the residual covariance matrix plus 
    the log determinant  of the covariance of the selected subset is less than or 
    equal to the provided cutoffs.

    Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix to perform subset selection with.
    cutoffs : np.array
        A '(p+1, )'-shaped numpy array containing the cutoffs. The greedily selected size-i
        susbet is sufficient if the corresponding log determinant is less than or qual to 
        the i-th value of cutoffs. 
    include : np.array[int], default=np.array([])
        A list of variables that must be included. 
    exclude: np.array[int], default=np.array([])
        A list of variables that must not be included.
    tol : float, default=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
	
    Returns 
	-------
	S : np.array
        The greedily selected subset. 
    reject : bool
        Whether the subset's corresponding log determinant is less than or equal to 
        the corresponding cutoff. 

	"""

    check_greedy_subset_factor_inputs(Sigma, cutoffs, include, exclude, tol)
    
    Sigma_R = Sigma.copy()
    p = Sigma.shape[0]
    
    # check if size-0 subset is sufficient 
    reject = np.sum(np.log(np.diag(Sigma_R))) > cutoffs[0]
    if (not reject and len(include) == 0) or len(exclude) == p:
        return np.array([]), reject
        
    
    S = -1 * np.ones(p).astype(int)
    idx_order = np.arange(p)
    num_active = p

    selected_enough = False
    num_selected = 0

    running_residuals = -1 * np.ones(p)

    # subset to acvice variables
    Sigma_R_active = Sigma_R[:num_active, :num_active]
    while not selected_enough:

        if num_selected < len(include):
            j_star = np.where(idx_order == include[num_selected])[0][0]
        else:
            # compute objective values
            obj_vals, colinearity_errors = subset_factor_score(Sigma_R_active, tol=tol)

            # if adding a variable results in zero residual, warn the user and fail to reject
            if len(colinearity_errors[0]) > 0:
                warnings.warn("When you add variable "  + str(colinearity_errors[0]) + " to " + str(S[:num_selected]) + " it perfectly explains variable " + str(colinearity_errors[1]) + ".")
                reject = False
                return np.concatenate([S[:num_selected], np.array([colinearity_errors[0][1]])]), reject

            # set the exclude objective values to infinity
            obj_vals[np.in1d(idx_order[:num_active], exclude)] = np.inf
            # select next variable
            j_star = random_argmin(obj_vals)

        S[num_selected] = idx_order[j_star]
        running_residuals[num_selected] = Sigma_R_active[j_star, j_star]
        num_selected += 1

        # regress off selected variable
        regress_one_off_in_place(Sigma_R_active, j_star, tol=tol)

        # swap so selected variable is in last active position
        swap_in_place(Sigma_R, [num_active - 1], [j_star], idx_order=idx_order)
        # decrement number active
        num_active -= 1

        # subset_to_active_variables
        Sigma_R_active = Sigma_R[:num_active, :num_active]

        # when including variables that have been requested to be included, ensure that no residuals are zero
        if num_selected <= len(include):
            zeros = np.where(np.diag(Sigma_R_active) < tol)[0]
            if len(zeros) > 0:
                warnings.warn("Variables " + str(S[:num_selected]) + " perfectly explain " + str(idx_order[zeros]) + ".")
                reject = False
                return include, reject

        # continue if not enough included
        if num_selected < len(include):
            continue

        # if we fail to reject, terminate and return
        if np.sum(np.log(np.diag(Sigma_R_active))) + np.sum(np.log(running_residuals[:num_selected])) <= cutoffs[num_selected]:
            reject = False
            selected_enough = True

        # forcefully fail to reject once we've selected at least p-1 in case of numerical instability
        if num_selected >= p - 1:
            reject = False
            selected_enough = True

        # terminate if no variables are left to select
        if set(idx_order[:num_active]).issubset(exclude):
            selected_enough = True

    return S[:num_selected], reject

def check_swapping_subset_factor_inputs(Sigma,
                                        k,
                                        cutoff, 
                                        max_iter,
                                        num_inits, 
                                        S_init,
                                        find_minimizer,
                                        include,
                                        exclude,
                                        tol):
    """
    Checks if the inputs to `swapping_subset_factor_selection` meet the required specifications.
	"""

    n, p = Sigma.shape 

    if not n == p:
        raise ValueError("Sigma must be a square matrix.")

    if not isinstance(k, (int, np.integer)) or k < 0 or k > p:
        raise ValueError("k must be an integer between 0 and p (inclusive).")
    
    set_include = set(include)
    set_exclude = set(exclude)
    if not isinstance(include, np.ndarray) or (include.dtype != 'int' and len(include) > 0) or not set_include.issubset(np.arange(p)): 
        raise ValueError('Include must be a numpy array of integers from 0 to p-1.')
    if not isinstance(exclude, np.ndarray) or (exclude.dtype != 'int' and len(exclude) > 0) or not set_exclude.issubset(np.arange(p)):
        raise ValueError('Exclude must be a numpy array of integers from 0 to p-1.')
    if len(set_exclude.intersection(set_include)) > 0:
        raise ValueError("Include and exclude must be disjoint.")
    
    if S_init is not None:
        if not isinstance(S_init, np.ndarray) or S_init.dtype != 'int' or len(set(S_init)) != k or (not set(S_init).issubset(np.arange(p))):
            raise ValueError("S_init must be a numpy array of k integers from 0 to p-1 inclusive.")
        if not set(include).issubset(S_init):
            raise ValueError("Include must be a subset of S_init.")
        if len(set(exclude).intersection(S_init)) > 0:
            raise ValueError("S_init cannot contain any elements in exlcude.")
        

    if len(include) > k:
        raise ValueError("Cannot include more than k.")
    if k is not None and len(exclude) > p - k:
        raise ValueError("Cannot exclude more than p-k.")


    if not isinstance(tol, float):
        raise ValueError("tol must be a float.")
    
    return 

def swapping_subset_factor_with_init(Sigma, 
                                     S_init,
                                     find_minimizer,
                                     cutoff, 
                                     max_iter,
                                     include,
                                     exclude,
                                     tol=TOL):
    
    '''
    Performs swapping subset factor selection with a particular initialization.
    See `swapping_subset_factor_selection` for a description of inputs. 
    '''
    
    k = len(S_init)
    
    # handle case where subset must be empty 
    if k == 0:
        log_det = np.sum(np.log(np.diag(Sigma)))
        reject = log_det > cutoff
        return np.array([]), reject, log_det 
    
    p = Sigma.shape[0]
    d = p-k
    include_set = set(include)

    idx_order = np.arange(p)

    Sigma_R = Sigma.copy()
    # these will always be the indices of the selected subset
    subset_idxs = np.arange(d, p)
    # swap initial variables to bottom of Sigma
    swap_in_place(Sigma_R, subset_idxs, S_init, idx_order=idx_order) #order of idxs1 and idxs2 is important!
    S = idx_order[d:].copy()
    Sigma_S = Sigma[:, S][S, :]
    invertible, Sigma_S_L = is_invertible(Sigma_S)   

    if not invertible:
        warnings.warn("Variables " + str(S_init) + " are colinear." )
        reject = False
        return S_init, reject, -np.inf  

    regress_off_in_place(Sigma_R, np.arange(d, p), tol=tol)
    
    where_zeros = np.where(np.diag(Sigma_R)[:d] < tol)[0]
    if len(where_zeros > 0):
        warnings.warn("Variables " + str(S_init) + " perfectly explain " + str(idx_order[where_zeros]) )
        reject = False 
        return S_init, reject, -np.inf 
    

    # number of completed iterations
    N = 0
    # counter of how many consecutive times we have chose not to swap 
    not_replaced = 0
    # permutation which shifts the last variable in the subset to the top of the subset
    subset_idxs_permuted = np.concatenate([subset_idxs[1:], np.array([subset_idxs[0]])])
    converged = False

    while N < max_iter and (not converged):
        for i in range(k):
            S_0 = S[0]

            # Update cholesky after removing first variable from subset
            Sigma_T_L = update_cholesky_after_removing_first(Sigma_S_L)

            if S_0 not in include_set:
            
                # Subset with first variable removed from selected subset
                T = S[1:].copy()

                # Update residual covariance after removing first variable from subset
                v = Sigma[:, S_0] - Sigma[:, T] @ solve_with_cholesky(Sigma_T_L, Sigma[T, S_0]) if k > 1 else Sigma[:, S_0]
                reordered_v = v[idx_order]
                Sigma_R = Sigma_R + np.outer(reordered_v, reordered_v)/v[S_0]
                
                # Swap first variable from subset to to top of residual matrix
                swap_in_place(Sigma_R, np.array([0]), np.array([d]), idx_order=idx_order)
                
                # compute objectives and for active variables and find minimizers
                obj_vals, colinearity_errors = subset_factor_score(Sigma_R[:(d+1), :(d+1)], tol=tol)

                # if adding a variable results in zero residual, warn the user and fail to reject
                if len(colinearity_errors[0]) > 0:
                    warnings.warn("When you add variable "  + str(colinearity_errors[0]) + " to " + str(S[:num_selected]) + " it perfectly explains variable " + str(colinearity_errors[1]) + ".")
                    reject = False
                    return np.concatenate([T, np.array([colinearity_errors[0][1]])]), reject, -np.inf

                # set the objective value to infinity for the excluded variables
                obj_vals[np.in1d(idx_order[:(d+1)], exclude)] = np.inf

                choices = np.flatnonzero(obj_vals == obj_vals.min())

                # if removed variable is a choice, select it, otherwise select a random choice
                if 0 in choices:
                    not_replaced += 1
                    j_star = 0
                else:
                    not_replaced = 0
                    j_star = np.random.choice(choices)
                
                S_new = idx_order[j_star]
                
                # In residual covariance, regress selected variable off the remaining
                regress_one_off_in_place(Sigma_R[:(d+1), :(d+1)], j_star)
                # In residual covariance swap new choice to top of selected subset 
                swap_in_place(Sigma_R, np.array([d]),  np.array([j_star]), idx_order=idx_order)
              
            else:
                S_new = S_0 
            
            # Add new choice as the last variable in selected subset
            S[:k-1] = S[1:].copy()
            S[k-1] = S_new
            # Update cholesky after adding new choice as last variable in selected subset
            Sigma_S_L = update_cholesky_after_adding_last(Sigma_T_L, Sigma[S_new, S])
            
            # permute first variables in selected subset to the last variable in the residual matrix
            perm_in_place(Sigma_R, subset_idxs,  subset_idxs_permuted, idx_order=idx_order)
            
            # If you don't want to find the minimizer and log det is small enough, terminate now
            if not find_minimizer:
                log_det = np.sum(np.log(np.diag(Sigma_R)[:d])) + np.sum(np.log(np.square(np.diag(Sigma_S_L))))
                if log_det <= cutoff:
                    reject = False
                    return S, reject, log_det

            if not_replaced == k - len(include):
                converged=True
                break

        N += 1

    log_det = np.sum(np.log(np.diag(Sigma_R)[:d])) + np.sum(np.log(np.square(np.diag(Sigma_S_L))))
    reject = (log_det > cutoff)
    return S, reject, log_det

def swapping_subset_factor_selection(Sigma,
                                     k,
                                     cutoff,
                                     max_iter=100,
                                     num_inits=1, 
                                     S_init=None,
                                     find_minimizer=True, 
                                     include=np.array([]),
                                     exclude=np.array([]),
                                     tol=TOL):
    
    """
    Given a `(p, p)`-covariance matrix `Sigma` uses iterative swapping to 
    approximately find a size k subset that minimizes the sum of the log 
    determinant of the diagonal of the residual covariance matrix plus the 
    log determinant of the covariance matrix of the selected subset. The swapping
    happens in the same order as the initial subset. 

    Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix to perform subset selection with.
    k : int, default=`None`
        The number of variables to select. 
    cutoff : float
        The value we want the log determinant to be under.
    max_iter : int, default=100
        Maximum number of iterations for the swapping algorithm to achieve convergence. If 
        the algorithm does not achieve converge it returns the best subset till that point.
    num_inits : int, default=1
        Number of random initializations to try. Only relevant if `S_init` is not `None`.
    S_init : np.array[int] 
        Size `k` array of variables that serves as the initialization for the swapping algorithm.
        If `None` then `num_inits` random initializations are tried.   
    find_minimizer : bool
        Controls whether to terminante right when a subset that achieves a log determinant 
        under `cutoff` or whether to continue searching for better subsets. 
    include : np.array[int], default=np.array([])
        A list of variables that must be included. 
    exclude: np.array[int], default=np.array([])
        A list of variables that must not be included.
    tol : float, default=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
	
    Returns 
	-------
	S : np.array
        The greedily selected subset. 
    reject : bool
        Whether the subset's corresponding log determinant is less than or equal to 
        the corresponding cutoff. 
        
	"""
    
    check_swapping_subset_factor_inputs(Sigma,
                                        k,
                                        cutoff, 
                                        max_iter,
                                        num_inits, 
                                        S_init,
                                        find_minimizer,
                                        include,
                                        exclude,
                                        tol)
    
    reject = True
    best_S = None
    best_log_det = np.inf 
    not_include = np.array([idx for idx in complement(Sigma.shape[0], include) if idx not in set(exclude)])
    
    if len(include) > 0:
        invertible, _ = is_invertible(Sigma[include, :][:, include], tol=tol)
        if not invertible:
            warnings.warn("The variables that have been requested to be included are colinear.")
            reject = False
            return np.concatenate([include, not_include[:(k - len(include))]]), reject
    
    no_initialization = (S_init is None)
    if not no_initialization or k == 0 or k == 1:
        num_inits = 1

    for _ in range(num_inits):
        if no_initialization:
            S_init = np.concatenate([include, np.random.choice(not_include, k-len(include), replace=False)]).astype(int)

        S, reject, log_det = swapping_subset_factor_with_init(Sigma=Sigma,
                                                              S_init=S_init,
                                                              find_minimizer=find_minimizer,
                                                              cutoff=cutoff,
                                                              max_iter=max_iter, 
                                                              include=include,
                                                              exclude=exclude,
                                                              tol=TOL)
        if not find_minimizer and (not reject):
            return S, reject 
        
        if find_minimizer and (not reject):
            reject = reject

        if log_det < best_log_det:
            best_S = S.copy()
            best_log_det = log_det 

    return best_S, reject 

def check_exhuastive_subset_factor_inputs(Sigma, 
                                          k, 
                                          cutoff, 
                                          include,
                                          exclude,
                                          find_minimizer,
                                          show_progress,
                                          tol):
    
    """
    Checks if the inputs to `exhaustive_subset_factor_selection` meet the required specifications.
	"""
    
    n, p = Sigma.shape 

    if not n == p:
        raise ValueError("Sigma must be a square matrix.")

    if not isinstance(k, (int, np.integer)) or k <= 0 or k > p:
        raise ValueError("k must be an integer > 0 and <= p.")
        
    set_include = set(include)
    set_exclude = set(exclude)
    if not isinstance(include, np.ndarray) or (include.dtype != 'int' and len(include) > 0) or not set_include.issubset(np.arange(p)): 
        raise ValueError('Include must be a numpy array of integers from 0 to p-1.')
    if not isinstance(exclude, np.ndarray) or (exclude.dtype != 'int' and len(exclude) > 0) or not set_exclude.issubset(np.arange(p)):
        raise ValueError('Exclude must be a numpy array of integers from 0 to p-1.')
    if len(set_exclude.intersection(set_include)) > 0:
        raise ValueError("Include and exclude must be disjoint.")

    if len(include) > k:
        raise ValueError("Cannot include more than k.")
    if len(exclude) > p - k:
        raise ValueError("Cannot exclude more than p-k.")
    

def exhaustive_subset_factor_selection(Sigma, 
                                       k, 
                                       cutoff, 
                                       include=np.array([]),
                                       exclude=np.array([]),
                                       show_progress=True,
                                       find_minimizer=False, 
                                       tol=TOL):
    
    """
    Given a `(p, p)`-covariance matrix `Sigma` exhaustively searches
    for the size k that results in the lowest value of log determinant of 
    diagonal of residual covariance matrix plus log determinant of covariance
    of selected subset.  

    Parameters
	----------
	Sigma : np.array
	    A `(p, p)`-shaped covariance matrix to perform subset selection with.
    k : int, default=`None`
        The number of variables to select. 
    cutoff : float
        If the corresponding log determinant is larger than the cutoff then we reject  
    include : np.array[int], default=np.array([])
        A list of variables that must be included. 
    exclude: np.array[int], default=np.array([])
        A list of variables that must not be included.
    find_minimizer : bool
        If `True`, continues to search for minimizer if subset that achieves log determinant
        below the cutoff is found.
    show_progress : bool
        If `True`, informs the user of the number of subsets being searched over
        and shows a progress bar.
    tol : float, default=`TOL`
        Tolerance at which point we consider a variable to have zero variance.
	
    Returns 
	-------
	S : np.array
        The greedily selected subset. 
    reject : bool
        Whether the subset's corresponding log determinant is less than or equal to 
        the corresponding cutoff. 
        
	"""
    
    p = Sigma.shape[0]

    check_exhuastive_subset_factor_inputs(Sigma=Sigma, 
                                          k=k, 
                                          cutoff=cutoff, 
                                          include=include,
                                          exclude=exclude,
                                          find_minimizer=False,
                                          show_progress=show_progress,
                                          tol=tol)

    best_S = None
    best_log_det = np.inf 
    reject = True 

    options = np.array([idx for idx in np.arange(p) if idx not in np.concatenate([include, exclude])])
    to_add = k - len(include)
    S = np.concatenate([include, -1*np.ones(to_add)]).astype(int)

    if show_progress:
        print("Iterating over " + str(math.comb(len(options), to_add)) + " different subsets...")
        iterator = tqdm.tqdm(itertools.combinations(options, to_add))
    else:
        iterator = itertools.combinations(options, to_add)

    for  remaining in iterator:
        S[len(include):] = np.array(remaining).astype(int)
        Sigma_R = regress_off(Sigma, S, tol=tol)
        invertible, Sigma_S_L =  is_invertible(Sigma[:, S][ S, :])
        S_comp = complement(p, S)
        if  not invertible or not np.all(np.diag(Sigma_R)[S_comp]) > tol:
            warnings.warn("Subset " + str(S) + " is colinear or perfectly explains some other variables.")
            reject = False
            return S, reject
        
        log_det =  np.sum(np.log((np.square(np.diag(Sigma_S_L))))) + np.sum(np.log(np.diag(Sigma_R)[S_comp]))
        if log_det <= cutoff:
            reject = False
            if not find_minimizer:
                return S, reject

        if log_det < best_log_det:
            best_log_det = log_det
            best_S = S.copy() 
    
    return best_S, reject 