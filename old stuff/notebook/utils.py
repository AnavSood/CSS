import numpy as np

TOL = 1e-10

def regress_one_off(Sigma, j, tol=TOL):
    if Sigma[j, j] > tol:
        return Sigma - np.outer(Sigma[:, j], Sigma[:, j])/Sigma[j, j]   
    else:
        return Sigma 

def regress_off(Sigma, S, tol=TOL):
    for j in S:
        Sigma = regress_one_off(Sigma, j, tol)
    return Sigma 
