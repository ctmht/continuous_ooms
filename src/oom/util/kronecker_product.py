# SOURCE: https://gist.github.com/ahwillia/9a81c0d091e39e319b4b9ab0919e304b
"""
References:

  - B. Plateau, On the stochastic structure of parallelism and synchronization models for distributed algorithms.
  Perform. Eval. Rev., 13 (1985), pp. 147–154.

  - Dayar, T., & Orhan, M. C. (2015). On vector-Kronecker product multiplication with rectangular factors.
  SIAM Journal on Scientific Computing, 37(5), S526-S543.
"""

from functools import reduce
import numpy as np
import numpy.random as npr
import numba
from numpy.testing import assert_array_almost_equal


def kron_vec_prod(Xs, v, out=None, side="right"):
    """
    Computes matrix-vector multiplication between
    matrix kron(X[0], X[1], ..., X[N]) and vector v.

    Parameters
    ----------
    Xs : list of ndarray
        List of square matrices defining Kronecker
        structure.
    v : ndarray
        Vector to multiply.
    out : ndarray or None
        Output vector.
    side : str
        Either "right" to specify kron(Xs...) @ v
        or "left" to specify v.T @ kron(Xs...).

    Returns
    -------
    out : ndarray
        Vector holding result.
    """

    if (out is None) and (side == "right"):
        out = np.empty(np.prod([X.shape[1] for X in Xs]))
    elif (out is None) and (side == "left"):
        out = np.empty(np.prod([X.shape[0] for X in Xs]))

    if side == "right":
        return _left_kron_vec_prod([X.T for X in Xs], v, out)
    elif side == "left":
        return _left_kron_vec_prod(Xs, v, out)
    else:
        raise ValueError(
            "Expected 'side' option to be 'left' or 'right'.")


@numba.jit(nopython=True)
def _left_kron_vec_prod(Xs, p, q):

    H = len(Xs)
    rs = [X.shape[0] for X in Xs]

    q[:] = p

    ileft = 1
    iright = 1
    for X in Xs[1:]:
        iright *= X.shape[0]

    for h in range(H):

        base_i = 0
        base_j = 0

        z = np.empty((1, rs[h]))
        zp = np.empty((1, rs[h]))

        for il in range(ileft):
            for ir in range(iright):
                ii = base_i + ir
                ij = base_j + ir

                for row in range(rs[h]):
                    z[0, row] = q[ii]
                    ii += iright

                np.dot(z, Xs[h], out=zp)

                for col in range(rs[h]):
                    q[ij] = zp[0, col]
                    ij += iright

            base_i += rs[h] * iright
            base_j += rs[h] * iright

        if (h + 1) != H:
            ileft *= rs[h]
            iright //= rs[h + 1]

    return q


# if __name__ == "__main__":
#     print("Testing...")
#     dims = (2, 3, 4, 3, 2)

#     # Left multiplication, square matrices
#     Xs = [npr.randn(s, s) for s in dims]
#     p = npr.randn(np.prod(dims))

#     expected = (p[None, :] @ reduce(np.kron, Xs)).ravel()
#     actual = kron_vec_prod(Xs, p, side="left")

#     assert_array_almost_equal(expected, actual)
#     print("* Passed 1 / 2")

#     # Right multiplication, square matrices
#     Xs = [npr.randn(s, s) for s in dims]
#     p = npr.randn(np.prod(dims))

#     expected = reduce(np.kron, Xs) @ p
#     actual = kron_vec_prod(Xs, p, side="right")

#     assert_array_almost_equal(expected, actual)
#     print("* Passed 2 / 2")