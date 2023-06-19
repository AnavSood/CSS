from pycss.CSS import *
from pycss.utils import *
from pycss.subset_selection import *

def test_CSS():

    p=20
    k=4
    Sigma = np.diag(np.arange(1, p + 1))

    include=np.array([0])
    exclude=np.array([19])
    
    css = CSS()
    css.select_subset_from_cov(Sigma=Sigma,
                               method='greedy',
                               k=k,
                               include=include,
                               exclude=exclude,
                               standardize=False)
    S = np.array([0, 18, 17, 16])
    Sigma_R = Sigma.copy()
    for i in S:
        Sigma_R[i, i] = 0
    
    assert(css.p == p)
    assert(np.all(css.Sigma == Sigma))
    assert(np.all(css.Sigma_R == Sigma_R))
    assert(np.all(css.S == S))
    assert(np.all(css.include == include))
    assert(np.all(css.exclude == exclude))
    assert(css.S_init is None)
    assert(css.converged is None)

    p = 15
    k = 4
    Sigma = np.diag(np.arange(1, p + 1))

    include=np.array([2, 3])
    exclude=np.array([13])

    css.select_subset_from_cov(Sigma=Sigma,
                               method='swap',
                               k=k,
                               include=include,
                               exclude=exclude,
                               standardize=False)
    S = np.array([2, 3, 12, 14])
    Sigma_R = Sigma.copy()
    for i in S:
        Sigma_R[i, i] = 0
    
    assert(css.p == p)
    assert(np.all(css.Sigma == Sigma))
    assert(np.all(css.Sigma_R == Sigma_R))
    assert(set(S) == set(css.S))
    assert(np.all(css.include == include))
    assert(np.all(css.exclude == exclude))
    assert(css.S_init is not None)
    assert(css.converged == True)

    p = 15
    k = 4
    Sigma = np.diag(np.arange(1, p + 1))

    include=np.array([4])
    exclude=np.array([6])

    css.select_subset_from_cov(Sigma=Sigma,
                               method='exhaustive',
                               k=k,
                               include=include,
                               exclude=exclude,
                               standardize=False,
                               show_progress=False)
    S = np.array([4, 12, 13, 14])
    Sigma_R = Sigma.copy()
    for i in S:
        Sigma_R[i, i] = 0
    
    assert(css.p == p)
    assert(np.all(css.Sigma == Sigma))
    assert(np.all(css.Sigma_R == Sigma_R))
    assert(set(S) == set(css.S))
    assert(np.all(css.include == include))
    assert(np.all(css.exclude == exclude))
    assert(css.S_init is None )
    assert(css.converged is None)


def test_from_data_vs_from_cov():
    
    p = 100
    k = 50
    np.random.seed(0)
    X = np.random.normal(0, 1, (p, p))
    X -= np.mean(X, axis=0)
    Sigma = 1/len(X) * X.T @ X

    include = np.array([5, 6])
    exclude = np.array([7, 8])

    css1 = CSS()
    css1.select_subset_from_data(X=X,
                                 method='greedy',
                                 k=k,
                                 standardize=False, 
                                 center=False,
                                 include=include,
                                 exclude=exclude)
    css2 = CSS()
    css2.select_subset_from_cov(Sigma=Sigma, 
                                method='greedy',
                                k=k,
                                standardize=False,
                                include=include,
                                exclude=exclude)
    
    assert(css1.p == css2.p)
    assert(np.all(css1.Sigma == css2.Sigma))
    assert(np.all(css1.Sigma_R == css2.Sigma_R))
    assert(set(css1.S) == set(css2.S))
    assert(np.all(css1.include == css2.include))
    assert(np.all(css1.exclude == css2.exclude))

def test_rsq_cutoffs():
    
    p = 100

    np.random.seed(0)
    X = np.random.normal(0, 1, (p, p))
    _, Sigma = get_moments(X)
    Sigma = standardize_cov(Sigma)

    css = CSS()
    css.select_subset_from_data(X=X,
                                method='greedy',
                                cutoff=0.1,
                                objective='rsq', 
                                center=True)
    
    Sigma_R = regress_off(Sigma, css.S)
    S_comp = complement(p, css.S)
    Sigma_R_comp_diag = np.diag(Sigma_R)[S_comp]
    assert(np.mean(1 - Sigma_R_comp_diag) >= 0.1) 

    Sigma_R = regress_off(Sigma, css.S[:len(css.S) - 1])
    S_comp = complement(p, css.S[:len(css.S) - 1])
    Sigma_R_comp_diag = np.diag(Sigma_R)[S_comp]
    assert(np.mean(1 - Sigma_R_comp_diag) < 0.1) 







