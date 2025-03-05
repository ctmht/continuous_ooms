from collections.abc import Callable
from typing import Optional, Any, Union

import numpy as np
import pandas as pd
from scipy.stats import rv_continuous
import sklearn.utils.extmath as sklue

from ..discrete_observable import DiscreteObservable
from ..ContinuousValuedOOM import ContinuousValuedOOM



def spectral_algorithm(
	estimation_routine: Callable[..., tuple[dict[Any, np.matrix], np.matrix, np.matrix]],
	obs: list[DiscreteObservable | str | int],
	target_dimension: int,
	estimated_matrices: Optional[tuple[np.matrix]] = None,
	**kwargs
) -> tuple:
	"""
	
	"""
	# STEPS 3 AND 4:
	# 3. GATHER LARGE MATRICES OF ESTIMATES (based on whatever desired criteria)
	# 4. APPLY FURTHER PROCESSING TO THE MATRICES (if applicable)
	if not estimated_matrices:
		estimated_matrices = estimation_routine(
			obs,
			**kwargs
		)
	F_IzJ, F_0J, F_I0 = estimated_matrices
	F_IJ = F_IzJ[0]
	
	# STEP 6
	# 6. CHOOSE CHARACTERIZER AND INDICATOR MATRICES C, Q s.t. C*F_IJ*Q INVERTIBLE
	# Get characterizer and inductor matrices spectrally
	C, Q = get_CQ_by_svd(F_IJ, target_dimension)
	
	# STEP 7
	# 7. APPLY LEARNING EQUATIONS TO GET MODEL COMPONENTS
	# Inverse matrix computation
	V = C * F_IJ * Q
	V_inv = np.linalg.inv(V)
	
	# Get linear functional (learning equation #1)
	# It holds that tvtype(sigma) == LinFunctional == np.matrix
	sigma = F_0J * Q * V_inv

	# Get discrete_observable operators (learning equation #2)
	tau_z = {}
	for obs, F_IkJ in F_IzJ.items():
		if obs == 0:
			continue
		
		operator_matrix = C * F_IkJ * Q * V_inv
		
		operator = operator_matrix
		
		tau_z[obs] = operator
	
	# Get state vector (learning equation #3)
	# It holds that type(sigma) == np.matrix
	omega = C * F_I0
	
	return sigma, tau_z, omega


def get_CQ_by_svd(
	estimated_matrix: np.matrix,
	target_dimension: int
) -> tuple[np.matrix, np.matrix]:
	"""
	
	"""
	# TODO: recheck parameters of SVD call
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


# def _estimate_f(
# 	obs: list[DiscreteObservable],
# 	word: list[DiscreteObservable]
# ) -> float:
# 	""" Estimate the system function """
# 	return -1 # obs.count_sub(word) / (len(obs) - len(word) + 1)
#
#
# def alphabet(
# 	seq: list[DiscreteObservable]
# ) -> list[DiscreteObservable]:
# 	return list(set(seq))
#
#
# def get_matrix_estimates(
# 	obs: list[DiscreteObservable],
# 	chr_w: list[list[DiscreteObservable]],
# 	ind_w: list[list[DiscreteObservable]],
# ) -> np.array:
# 	"""
#
# 	"""
# 	def _getmat(nrows: int, ncols: int) -> np.matrix:
# 		""" Shortened matrix creation code """
# 		return np.asmatrix(np.zeros([nrows, ncols]))
#
# 	obsalphabet = alphabet(obs)
#
# 	# Define empty target matrices
# 	F_0J = _getmat(1, len(ind_w))
# 	F_I0 = _getmat(len(chr_w), 1)
# 	F_IJ = _getmat(len(chr_w), len(ind_w))
# 	F_IzJ = [_getmat(len(chr_w), len(ind_w)) for _ in obsalphabet]
#
# 	for cidx, cword in enumerate(chr_w):
# 		# debug verbosity
# 		print(f"{(cidx, cword)} " if cidx % 10 == 0 else '', end = '')
#
# 		# Compute one
# 		F_I0[cidx, 0] = _estimate_f(obs, cword)
#
# 		for iidx, iword in enumerate(ind_w):
# 			# Compute F_IJ[cidx, iidx]
# 			F_IJ[cidx, iidx] = _estimate_f(obs, iword + cword)
#
# 			# Compute F_IzJ[cidx, iidx] for z in possible_observations
# 			for zidx, z in enumerate(obsalphabet):
# 				F_IzJ[zidx][cidx, iidx] = _estimate_f(obs, iword + z + cword)
#
# 	for iidx, iword in enumerate(ind_w):
# 		# Compute other
# 		F_0J[0, iidx] = _estimate_f(obs, iword)
#
# 	return F_0J, F_I0, F_IJ, F_IzJ