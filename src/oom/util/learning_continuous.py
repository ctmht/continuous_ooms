from typing import Optional
import itertools

import numpy as np
import scipy as sp
from scipy.stats import rv_continuous

from ..discrete_observable import DiscreteObservable
from ..ContinuousValuedOOM import ContinuousValuedOOM
from .kronecker_product import kron_vec_prod
from .spectral import spectral_algorithm


def learn_continuous_valued_oom(
	obs: list[DiscreteObservable | str | int],
	target_dimension: int,
	len_cwords: int,
	len_iwords: int,
	mfn_dict: dict[DiscreteObservable, rv_continuous],
	estimated_matrices: Optional[tuple[np.matrix]] = None
) -> 'ContinuousValuedOOM':
	"""
	
	"""
	# Estimate large matrices
	sigma, tau_z, omega = spectral_algorithm(
		estimation_routine = estimate_matrices_continuous,
		obs                = obs,
		target_dimension   = target_dimension,
		estimated_matrices = estimated_matrices,
		len_cwords         = len_cwords,
		len_iwords         = len_iwords,
		mfn_dict           = mfn_dict
	)
	
	learned_oom = ContinuousValuedOOM(
		dim                  = target_dimension,
		linear_functional    = sigma,
		obs_ops              = tau_z,
		start_state          = omega,
		membership_functions = list(mfn_dict.values())
	)
	learned_oom.normalize()
	
	return learned_oom


def estimate_matrices_continuous(
	sequence,
	len_cwords,
	len_iwords,
	mfn_dict
):
	"""
	
	"""
	alphabet = list(mfn_dict.keys())
	
	F_IJ_bl = dict(zip(
		[0] + alphabet,
		[np.zeros(shape = (len(alphabet) ** len_iwords, len(alphabet) ** len_cwords))
		 for _ in range(len(alphabet) + 1)]
	))
	
	F_0J_bl = np.zeros(shape = (1, len(alphabet) ** len_cwords))
	
	F_I0_bl = np.zeros(shape = (len(alphabet) ** len_iwords, 1))
	
	# Loop over all possible characteristic words of given length len_cwords
	for idxc, cword in enumerate(
		itertools.product(alphabet, repeat = len_cwords)
	):
		# Estimate entries of large row-matrix (gives linear functional)
		F_0J_bl[0, idxc] = estimate(
			sequence.sequence_cont, cword, mfn_dict
		)
		
		# Loop over all possible indicative words of given length len_iwords
		for idxq, qword in enumerate(
			itertools.product(alphabet, repeat = len_iwords)
		):
			# Estimate entries of large matrix
			F_IJ_bl[0][idxq, idxc] = estimate(
				sequence.sequence_cont, cword + qword, mfn_dict
			)
			# print(f"{estimate_matrices_bl[0][idxq, idxc] : <.10f}", cword, qword)
			
			# Loop over all possible discrete observables
			for obs in alphabet:
				# Estimate entries of large observable-indexed matrix
				# (gives operators)
				F_IJ_bl[obs][idxq, idxc] = estimate(
					sequence.sequence_cont, cword + (obs,) + qword, mfn_dict
				)
				# print(f"{estimate_matrices_bl[obs][idxq, idxc] : >12} ", cword, (obs,), qword)
	
	# Loop over all possible indicative words of given length len_iwords
	for idxq, qword in enumerate(
		itertools.product(alphabet, repeat = len_iwords)
	):
		# Estimate entries of large column-matrix (gives starting state)
		F_I0_bl[idxq, 0] = estimate(
			sequence.sequence_cont, qword, mfn_dict
		)
	
	# Applies processing to the blended-case estimate matrices
	estimate_matrices, estimate_row, estimate_column = process_matrices_continuous(
		len_iwords,
		len_cwords,
		mfn_dict,
		F_IJ_bl,
		F_0J_bl,
		F_I0_bl
	)
	
	return estimate_matrices, estimate_row, estimate_column


def estimate(cv_sequence, zbar, memberships):
	"""
	Computes the system function estimate for a given continuous-valued sequence and known
	membership functions, using the Ergodic Theorem for Strictly Stationary Processes
	
	Args:
		cv_sequence: the continuous-valued sequence
		zbar: a word formed of finitely many objects in the alphabet of a symbolic process
		memberships: dictionary linking each alphabet entry to its known membership function
	"""
	est = 0
	
	for start in range(len(cv_sequence) - len(zbar) + 1):
		est_here = 1

		# print(f"est_here: {est_here} ", end='')
		
		for idx in range(0, len(zbar)):
			mf = memberships[zbar[idx]]
			x = cv_sequence[start + idx]
			est_here *= mf.pdf(x)
			# print(f"{est_here} ", end='')

		est += est_here / (len(cv_sequence) - len(zbar) + 1)

	return est


def process_matrices_continuous(
	len_iwords,
	len_cwords,
	mfn_dict,
	F_IJ_bl,
	F_0J_bl,
	F_I0_bl
):
	alphabet = list(mfn_dict.keys())
	T = np.zeros(shape = (len(alphabet), len(alphabet)))

	for idx in range(len(alphabet)):
		for jdx in range(len(alphabet)):
			T[idx, jdx] = sp.integrate.quad(
				lambda x: mfn_dict[alphabet[idx]].pdf(x) * mfn_dict[alphabet[jdx]].pdf(x),
				0, 1
			)[0]
	
	F_IJ = {}
	for key, mat in F_IJ_bl.items():
		mat = mat.flatten()
		reps = len_cwords + len_iwords
		transformed = kron_vec_prod([T for _ in range(reps)], mat, side = "right")
		transformed = transformed.reshape(len(alphabet) ** len_iwords, len(alphabet) ** len_cwords)
		F_IJ[key] = transformed
	
	F_0J = kron_vec_prod(
		[T for _ in range(len_cwords)],
		F_0J_bl.flatten(),
		side = "right"
	).reshape(
		1,
		len(alphabet) ** len_cwords
	)
	
	F_I0 = kron_vec_prod(
		[T for _ in range(len_iwords)],
		F_I0_bl.flatten(),
		side = "right"
	).reshape(
		len(alphabet) ** len_iwords,
		1
	)
	
	return F_IJ, F_0J, F_I0