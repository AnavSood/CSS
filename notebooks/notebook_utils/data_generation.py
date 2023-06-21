import numpy as np
from choldate import cholupdate
from pycss.subset_selection import complement, perm_in_place


def get_equicorrelated_chol(k, off_diag, diag=1):
    
    chol = np.sqrt(diag-off_diag) * np.eye(k)
    cholupdate(chol, np.sqrt(off_diag) * np.ones(k))
    
    return chol.T

def get_block_W(p, k , num_blocks, block_size, overlap):
  
    W = np.zeros((p-k, k))
    row_start = 0
    col_start = 0
    row_move = int((p-k)/num_blocks)
    col_move = block_size - overlap
  
    for i in range(num_blocks):
        if i < num_blocks -1:
            W[row_start: row_start + row_move, col_start : col_start + block_size ] = 1
        else:
            W[row_start:, col_start:] = 1
    
        row_start += row_move
        col_start += col_move
    
    return W

def generate_gaussian_PCSS_sample_cov(n, C_chol, W, D=None, sigma_sq=None, S=None, B=None):
    
    if D is None and sigma_sq is None:
        raise ValueError("Both D and sigma_sq cannot be None.")
    if D is not None and sigma_sq is not None:
        raise ValueError("Both D and sigma_sq cannot be not None.")
    
    squeeze = False
    if B is None:
        B = 1
    squeeze = True
    
    k = C_chol.shape[0]
    p = k + W.shape[0]

    chis = np.random.chisquare(df=np.arange(n - 1, n - p - 1 , -1), size=(B,p))
    chi_sqrts = np.sqrt(chis)
    normals = np.random.normal(0, 1, (B, int(p*(p-1)/2)) )
    
    # Fill in the A matrix in the Bartlett decomposition from
    # https://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition
    
    A_top_left = np.zeros((B, k, k))
    A_top_left[:, np.tri(k, dtype=bool, k=-1)] = normals[:, :int(k *(k-1)/2)]
    i, j = np.diag_indices(k)
    A_top_left[:, i, j] = chi_sqrts[:, :k]

    A_bottom_left = normals[:, int(k *(k-1)/2) : int(k *(k-1)/2) + (p-k)*k].reshape((B, p-k, k))

    A_bottom_right = np.zeros((B, p-k, p-k))
    A_bottom_right[:, np.tri(p-k, dtype=bool, k=-1)] = normals[: , int(k *(k-1)/2) + (p-k)*k :]
    i, j =  np.diag_indices(p-k)
    A_bottom_right[:, i, j] = chi_sqrts[:, k:]

    # Left multiply by Cholesky decomposition of V as in 
    # https://en.wikipedia.org/wiki/Wishart_distribution#Bartlett_decomposition
    
    Sigma_hat_chol = np.zeros((B, p, p))
    Sigma_hat_chol[:, :k, :k] = C_chol[np.newaxis, :, :] @ A_top_left
  
    if sigma_sq is not None:
        sigma = np.sqrt(sigma_sq)
        Sigma_hat_chol[:, k:, :k] = W[np.newaxis, :, :] @ Sigma_hat_chol[:, :k, :k] + sigma * A_bottom_left
        Sigma_hat_chol[:, k:, k:] =  sigma * A_bottom_right
    if D is not None: 
        D_sqrt = np.sqrt(D)
        Sigma_hat_chol[:, k:, :k] = W[np.newaxis, :, :] @ Sigma_hat_chol[:, :k, :k] + D_sqrt[np.newaxis, :, np.newaxis] * A_bottom_left
        Sigma_hat_chol[:, k:, k:] =  D_sqrt[np.newaxis, :, np.newaxis] * A_bottom_right

    Sigma_hat = 1/n * Sigma_hat_chol @ np.transpose(Sigma_hat_chol, (0, 2, 1))
  
    if S is not None:
        S_comp = complement(p, S)
        perm_in_place(Sigma_hat, np.range(p), np.concatenate([S, S_comp]))
    
    return np.squeeze(Sigma_hat) if squeeze else Sigma_hat

def generate_PCSS_data(X_S, W, D=None, sigma_sq=None, mu_S = None, mu_S_comp=None, S=None):
    
    
    if D is None and sigma_sq is None:
        raise ValueError("Both D and sigma_sq cannot be None.")
    if D is not None and sigma_sq is not None:
        raise ValueError("Both D and sigma_sq cannot be not None.")
    
    squeeze = False
    if len(X_S.shape) == 2:
        X_S = X_S.reshape((1, X_S.shape[0], X_S.shape[1]))
        squeeze = True

    B, n, k = X_S.shape 

    if k != W.shape[1]:
        raise ValueError("Shapes of X_S and W do not agree")

    p = k + W.shape[0]

    noise = np.random.normal(0, 1, size=(B, n, p-k))
    if sigma_sq is not None:
        noise = noise * np.sqrt(sigma_sq)
    if D is not None:
        noise = noise *  np.sqrt(D)[np.newaxis , np.newaxis, :]

    if mu_S is None:
        mu_S = np.zeros(k)
    if mu_S_comp is None:
        mu_S_comp = np.zeros(p-k)

    X_S_comp = (X_S - mu_S[np.newaxis, np.newaxis, :] ) @ W.T[np.newaxis, :, :] + noise + mu_S_comp[np.newaxis, np.newaxis, :]
    X = np.dstack([X_S, X_S_comp]) 
    
    if S is not None:
        S_comp = complement(p, S)
        perm_in_place(X, np.range(p), np.concatenate([S, S_comp], row=False))

    return np.squeeze(X) if squeeze else X