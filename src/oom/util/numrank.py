import numpy as np


def spectral_norm_expectation(sqrt_V, n = 5):
    """
    Compute the expectation of the spectral norm of a random matrix with independently normally
    distributed entries with zero mean and standard deviation given by `sqrt_V` (i.e., the
    square root of the element-wise variances) from `n` samples.

    Source: m7thon/tom
    """
    if not sqrt_V.any(): return 0
    spec_norm = np.zeros(n)
    for i in range(n):
        spec_norm[i] = np.log(np.linalg.norm(np.multiply(sqrt_V, np.random.randn(*sqrt_V.shape)), ord=2))
    return np.exp(np.mean(spec_norm))


def numerical_rank_frob_mid_spec(F, seqlength, len_cwords, len_iwords, ret_bound: bool=False):
    """
    Numerical rank using unbiased variance estimate to determine error bound

    Adapted from m7thon/tom using the 'frob_mid_spec' case
    """
    # Compute singular values
    s = np.linalg.svd(F, compute_uv=False)

    # Estimate element-wise variance as for binomially distributed entries
    var_denom = seqlength - (len_cwords + len_iwords)
    var_convf = (var_denom + 1) / var_denom
    V = np.abs((F * var_convf - np.square(F) * var_convf**2) / var_denom)
    
    if np.any(V < 0):
        raise ValueError(f"Negative values #{np.sum(V < 0)}")
    
    # Compute for initial error estimate (sum of variances)
    e = np.sum(V)
    s2_tailsum = 0
    d_numrank = len(s)
    while d_numrank > 0 and s2_tailsum <= e:
        d_numrank -= 1
        s2_tailsum += s[d_numrank]**2
    
    e = s[d_numrank] - (s2_tailsum**0.5 - e**0.5) / s[d_numrank] * (s[d_numrank] - (0 if d_numrank == len(s) else s[d_numrank+1]))
    d_numrank += 1

    # Consider also this error metric which might fine-tune the result
    sqrt_V = np.sqrt(V)
    e = min(e, (spectral_norm_expectation(sqrt_V) * np.sum(sqrt_V) / V.size**0.5)**0.5)
    
    while d_numrank < len(s) and s[d_numrank] > e:
        d_numrank += 1

    return (d_numrank, e) if ret_bound else d_numrank



def numerical_rank_binomial(F, seqlength, ret_all: bool=False):
    """
    Compute the numerical rank of a matrix using the document's method.
    
    Parameters:
    - F (np.ndarray): Input matrix
    - seqlength (float): Training data size.
    
    Returns:
    - int: Numerical rank.
    - np.ndarray: Singular values.
    - float: Threshold.
    """
    # Normalize F such that sum(F) = seqlength
    current_sum = np.sum(F)
    F_normalized = F * (seqlength / current_sum)
    
    # Compute infinity norm (max row sum of absolute values)
    infinity_norm = np.max(np.sum(np.abs(F_normalized), axis=1))
    
    # Compute variance estimates (Equation 12)
    V_00 = np.nan_to_num(np.divide(
        np.sqrt(
            np.multiply(
                F_normalized,
                1 - F_normalized / seqlength
            )
        ),
        F_normalized,
        where = F_normalized > 0
    ))
    
    # Calculate epsilon and threshold
    dY, dX = F_normalized.shape  # Matrix dimension (Y x X)
    sum_alpha = np.sum(V_00)
    epsilon = sum_alpha / (dX * dY)
    threshold = epsilon * infinity_norm
    
    # Compute singular values and numerical rank
    S = np.linalg.svd(F_normalized, compute_uv=False)
    rank = np.sum(S > threshold)
    
    if ret_all:
        return rank, S, threshold
    return rank