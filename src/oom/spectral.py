import re

import numpy as np
from sklearn.utils.extmath import randomized_svd

from src.oom.DiscreteValuedOOM import DiscreteValuedOOM
from src.oom.OOM import State, LinFunctional
from src.oom.observable import Observable
from src.oom.operator import Operator


def count_appearences(
	observation,
	*subwords
) -> int:
	word = "".join(subwords)
	pattern = "(?=(" + word + "))"
	count = len(re.findall(pattern, observation))
	return count


def f_estimate(
	observation,
	*subwords
):
	word = "".join(subwords)
	count = count_appearences(observation, word)
	return count / (len(observation) - len(word) + 1)


def get_matrix_estimates(
	observation,
	possible_observations,
	chr_w,
	ind_w,
) -> np.array:
	F_0J = np.asmatrix(np.zeros([1, len(ind_w)]))
	F_I0 = np.asmatrix(np.zeros([len(chr_w), 1]))
	F_IJ = np.asmatrix(np.zeros([len(chr_w), len(ind_w)]))
	F_IzJ = [np.asmatrix(np.zeros([len(chr_w), len(ind_w)])) for _ in possible_observations]
	
	for cidx, cword in enumerate(chr_w):
		# Compute one
		F_I0[cidx, 0] = f_estimate(observation, cword)
		
		for iidx, iword in enumerate(ind_w):
			# Compute F_IJ[cidx, iidx]
			F_IJ[cidx, iidx] = f_estimate(observation, iword, cword)
	
			# Compute F_IzJ[cidx, iidx] for z in possible_observations
			for zidx, z in enumerate(possible_observations):
				F_IzJ[zidx][cidx, iidx] = f_estimate(observation, iword, z, cword)
	
	for iidx, iword in enumerate(ind_w):
		# Compute other
		F_0J[0, iidx] = f_estimate(observation, iword)
	
	return F_0J, F_I0, F_IJ, F_IzJ


def get_CQ_by_svd(
	estimated_matrix,
	target_dimension
):
	F = estimated_matrix
	print()
	
	U, S, Vt = randomized_svd(F, n_components = target_dimension)
	U = np.asmatrix(U)
	S = np.asmatrix(np.diag(S))
	Vt = np.asmatrix(Vt)
	
	C = U.T
	Q = Vt.T * np.linalg.pinv(S)
	
	return C, Q

def estimate_OOM(
	observation,
	possible_observations,
	chr_w,
	ind_w,
	target_dimension
):
	F_0J, F_I0, F_IJ, F_IzJ = get_matrix_estimates(observation, possible_observations, chr_w, ind_w)
	
	C, Q = get_CQ_by_svd(F_IJ, target_dimension)
	
	V = C * F_IJ * Q
	V_inv = np.linalg.inv(V)

	# Get linear functional
	sigma = F_0J * Q * V_inv

	# Get observable operators
	tau_z = []
	for F_IkJ in F_IzJ:
		tau_z.append(C * F_IkJ * Q * V_inv)
	
	# Get state vector
	omega = C * F_I0
	
	return (sigma, tau_z, omega), F_IJ