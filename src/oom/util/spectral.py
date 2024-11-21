from collections.abc import Sequence
from typing import Tuple

import numpy as np
import sklearn.utils.extmath as sklue

from ..observable import *

def _estimate_f(obs: ObsSequence, word: ObsSequence) -> float:
	""" Estimate the system function """
	return obs.count_sub(word) / (len(obs) - len(word) + 1)

def get_matrix_estimates(
	obs: ObsSequence,
	chr_w: Sequence[ObsSequence],
	ind_w: Sequence[ObsSequence],
) -> np.array:
	"""
	
	"""
	def _getmat(nrows: int, ncols: int) -> np.matrix:
		""" Shortened matrix creation code """
		return np.asmatrix(np.zeros([nrows, ncols]))
	
	# Define empty target matrices
	F_0J = _getmat(1, len(ind_w))
	F_I0 = _getmat(len(chr_w), 1)
	F_IJ = _getmat(len(chr_w), len(ind_w))
	F_IzJ = [_getmat(len(chr_w), len(ind_w)) for _ in obs.alphabet]
	
	for cidx, cword in enumerate(chr_w):
		print(f"{(cidx, cword)} " if cidx % 10 == 0 else '', end = '')			# debug verbosity
		
		# Compute one
		F_I0[cidx, 0] = _estimate_f(obs, cword)
		
		for iidx, iword in enumerate(ind_w):
			# Compute F_IJ[cidx, iidx]
			F_IJ[cidx, iidx] = _estimate_f(obs, iword + cword)
			
			# Compute F_IzJ[cidx, iidx] for z in possible_observations
			for zidx, z in enumerate(obs.alphabet):
				F_IzJ[zidx][cidx, iidx] = _estimate_f(obs, iword + z + cword)
	
	for iidx, iword in enumerate(ind_w):
		# Compute other
		F_0J[0, iidx] = _estimate_f(obs, iword)
	
	return F_0J, F_I0, F_IJ, F_IzJ


def get_CQ_by_svd(
	estimated_matrix: np.matrix,
	target_dimension: int
) -> Tuple[np.matrix, np.matrix]:
	"""
	
	"""
	# Get d-truncated SVD (U * S * V^T = estimated_matrix for d -> inf)
	U, S, Vt = sklue.randomized_svd(
		M = estimated_matrix,
		n_components = target_dimension,
		n_oversamples = max(10, estimated_matrix.shape[0] // 2 - target_dimension),
		n_iter = 10,
		power_iteration_normalizer = "QR",
	)
	U = np.asmatrix(U)
	S = np.asmatrix(np.diag(S))
	Vt = np.asmatrix(Vt)
	
	# Define characterizer matrix C and inductor matrix Q
	C = U.T
	Q = Vt.T * np.linalg.pinv(S)
	
	return C, Q