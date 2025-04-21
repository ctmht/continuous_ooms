import sys
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from ..discrete_observable import DiscreteObservable


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
		sys.stderr.write("Estimating matrices_bl - this should not happen in experiments.")
		estimated_matrices = estimation_routine(
			obs,
			**kwargs
		)
	F_zY_X, F_X_row, F_Y_col = estimated_matrices
	F_Y_X = F_zY_X[0]
	
	# STEP 6
	# 6. CHOOSE CHARACTERIZER AND INDICATOR MATRICES C, Q s.t. C*F_IJ*Q INVERTIBLE
	# Get characterizer and inductor matrices_bl spectrally
	C, Q = get_CQ_by_svd(F_Y_X, target_dimension)
	
	# STEP 7
	# 7. APPLY LEARNING EQUATIONS TO GET MODEL COMPONENTS
	# Inverse matrix computation
	V = C * F_Y_X * Q
	V_inv = np.linalg.inv(V)
	
	# Get linear functional (learning equation #1)
	# It holds that tvtype(sigma) == LinFunctional == np.matrix
	sigma = F_X_row * Q * V_inv

	# Get discrete_observable operators (learning equation #2)
	tau_z = {}
	for obs, F_zY_X_thisz in F_zY_X.items():
		if obs == 0:
			continue
		
		operator_matrix = C * F_zY_X_thisz * Q * V_inv
		
		operator = operator_matrix
		
		tau_z[obs] = operator
	
	# Get state vector (learning equation #3)
	# It holds that type(sigma) == np.matrix
	omega = C * F_Y_col
	
	return sigma, tau_z, omega


def get_CQ_by_svd(
	estimated_matrix: np.matrix,
	target_dimension: int
) -> tuple[np.matrix, np.matrix]:
	"""
	
	"""
	U, S, Vt = np.linalg.svd(
		a = estimated_matrix,
		full_matrices = True,
		compute_uv = True,
		hermitian = False
	)
	U = U[:, :target_dimension]
	S = S[:target_dimension]
	Vt = Vt[:target_dimension, :]
	
	U = np.asmatrix(U)
	S = np.asmatrix(np.diag(S))
	Vt = np.asmatrix(Vt)
	
	# Define characterizer matrix C and inductor matrix Q
	C = U.T
	Q = Vt.T * np.linalg.pinv(S)
	
	return C, Q