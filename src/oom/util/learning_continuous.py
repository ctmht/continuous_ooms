from typing import Optional
import itertools
import math

from scipy.stats import rv_continuous
from tqdm import tqdm
import numpy as np
import scipy as sp

from ..discrete_observable import DiscreteObservable
from .kronecker_product import kron_vec_prod
from .spectral import spectral_algorithm
from .mf_lookup import _MfLookup
import src.oom


def learn_continuous_valued_oom(
	obs: list[DiscreteObservable | str | int],
	target_dimension: int,
	len_cwords: int,
	len_iwords: int,
	membership_functions: list[rv_continuous],
	observables: Optional[list[DiscreteObservable]] = None,
	estimated_matrices: Optional[tuple[np.matrix]] = None
) -> 'src.oom.ContinuousValuedOOM':
	"""
	
	"""
	# Estimate large matrices
	sigma, tau_z, omega = spectral_algorithm(
		estimation_routine   = estimate_matrices_continuous,
		obs                  = obs,
		target_dimension     = target_dimension,
		estimated_matrices   = estimated_matrices,
		len_cwords           = len_cwords,
		len_iwords           = len_iwords,
		membership_functions = membership_functions,
		observables			 = observables
	)
	
	learned_oom = src.oom.ContinuousValuedOOM(
		dim                  = target_dimension,
		linear_functional    = sigma,
		obs_ops              = tau_z,
		start_state          = omega,
		membership_functions = membership_functions
	)
	learned_oom.normalize()
	
	return learned_oom


def estimate_matrices_continuous(
	sequence,
	len_cwords,
	len_iwords,
	membership_functions,
	observables
):
	"""
	
	"""
	# Create or format observables and membership functions
	if observables is not None:
		alphabet = observables
	else:
		alphabet = []
		for uidx in range(len(membership_functions)):
			name = chr(ord("a") + uidx)
			observable = DiscreteObservable(name)
			alphabet.append(observable)
	mfn_dict = dict(zip(alphabet, membership_functions))
	
	# Replaces mf_dict[observable].pdf(sequence_item) sequential computation with
	# accessing precomputed PDF values by mf_lookup[observable, index(sequence_item)]
	mf_lookup = _MfLookup(mfn_dict, steps_pdfs= 10000)
	
	# Simplify notation a bit
	n = len(alphabet)
	N = len(sequence)
	Lc = len_cwords
	Li = len_iwords
	
	# Create large matrices F_I(z)J, column vector F_0J, and row vector F_I0
	F_IJ_bl = dict(zip(
		[0] + alphabet,
		[np.zeros(shape = (n ** Li, n ** Lc))
		 for _ in range(n + 1)]
	))
	F_0J_bl = np.zeros(shape = (1, n ** Lc))
	F_I0_bl = np.zeros(shape = (n ** Li, 1))
	
	# Go through sequence
	for k0 in tqdm(range(0, N - (Lc + Li + 1)), position=0, leave=True):
		# Ensure lookup contains all values of the current subsequence
		if not mf_lookup.holds_pdfs(start = k0, end = k0 + (Lc + Li + 1)):
			mf_lookup.update_pdfs(sequence, start = k0)
		
		# Go through all possible characteristic words at this sequence step
		for idxc, cword in enumerate(
			itertools.product(alphabet, repeat = Lc)
		):
			# Compute the Lc-length membership of cword as the product of all
			# memberships sequentially in cword of the respective sequence elements
			c_base = math.prod([
				mf_lookup[cword[idx], k0 + idx]
				for idx in range(0, len(cword))
			])
			
			# Add indicative count in column vector (only once)
			F_0J_bl[0, idxc] += c_base / (N - Lc + 1)
			
			# Go through all possible observables
			for idxo, obs in enumerate(alphabet):
				o_incr = mf_lookup[obs, k0 + Lc]
				
				# Go through all possible "indicative" events (trick)
				k0co = k0 + Lc + 1
				for idxq, iword in enumerate(
					itertools.product(alphabet, repeat = Li - 1)
				):
					i_incr = math.prod([
						mf_lookup[iword[idx], k0co + idx]
						for idx in range(0, Li - 1)
					])
					
					# Add count in F_IJ_bl[0]
					_prod0 = c_base * o_incr * i_incr
					_idxi = idxo * n ** (Li - 1) + idxq
					F_IJ_bl[0][_idxi, idxc] += _prod0 / (N - (Lc + Li) + 1)
					
					# Add indicative count in row vector (only once)
					if idxc == 0:
						_prod_tot_iword = o_incr * i_incr
						F_I0_bl[_idxi] += _prod_tot_iword / (N - Li + 1)
					
					k0coi = k0co + Li
					# Go through all possible observables again
					for idxo2, obs2 in enumerate(alphabet):
						o2_incr = mf_lookup[obs2, k0coi]
						
						# Add count in F_IJ_bl[obs]
						_prodobs = _prod0 * o2_incr
						_idxi2 = idxq * n + idxo2
						F_IJ_bl[obs][_idxi2, idxc] += _prodobs / (N - (Lc + Li + 1) + 1)
	
	# The last subsequence of length (Lc + Li) is unaccounted for
	k0 = N - (Lc + Li)
	# Go through all possible characteristic words at this sequence step
	for idxc, cword in enumerate(
		itertools.product(alphabet, repeat = Lc)
	):
		# Compute the Lc-length membership of cword as the product of all
		# memberships sequentially in cword of the respective sequence elements
		c_base = math.prod([
			mf_lookup[cword[idx], k0 + idx]
			for idx in range(0, len(cword))
		])
		
		# Add indicative count in column vector (only once)
		F_0J_bl[0, idxc] += c_base / (N - Lc + 1)
		
		# Go through all possible indicative events
		for idxq, iword in enumerate(
			itertools.product(alphabet, repeat = Li)
		):
			i_incr = math.prod([
				mf_lookup[iword[idx], k0 + Lc + idx]
				for idx in range(0, len(iword))
			])
			
			# Add count in F_IJ_bl[0]
			_prod0 = c_base * i_incr
			F_IJ_bl[0][idxq, idxc] += _prod0 / (N - (Lc + Li) + 1)
			
			# Add indicative count in row vector (only once)
			if idxc == 0:
				F_I0_bl[idxq, 0] += i_incr / (N - Li + 1)
	
	# F_I0_bl /= (N - Li + 1)
	# F_0J_bl /= (N - Lc + 1)
	# F_IJ_bl[0] /= (N - (Lc + Li) + 1)
	# for obs in alphabet:
	# 	F_IJ_bl[obs] /= (N - (Lc + Li + 1) + 1)
	
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



def estimate_matrices_continuous_old(
	sequence,
	len_cwords,
	len_iwords,
	membership_functions,
	observables
):
	"""
	
	"""
	if observables is None:
		alphabet = []
		for uidx in range(len(membership_functions)):
			name = chr(ord("a") + uidx)
			observable = DiscreteObservable(name)
			alphabet.append(observable)
	else:
		alphabet = observables
	mfn_dict = dict(zip(alphabet, membership_functions))
	
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
			sequence, cword, mfn_dict
		)
		
		# Loop over all possible indicative words of given length len_iwords
		for idxq, qword in enumerate(
			itertools.product(alphabet, repeat = len_iwords)
		):
			# Estimate entries of large matrix
			F_IJ_bl[0][idxq, idxc] = estimate(
				sequence, cword + qword, mfn_dict
			)
			# print(f"{estimate_matrices_bl[0][idxq, idxc] : <.10f}", cword, qword)
			
			# Loop over all possible discrete observables
			for obs in alphabet:
				# Estimate entries of large observable-indexed matrix
				# (gives operators)
				F_IJ_bl[obs][idxq, idxc] = estimate(
					sequence, cword + (obs,) + qword, mfn_dict
				)
				# print(f"{estimate_matrices_bl[obs][idxq, idxc] : >12} ", cword, (obs,), qword)
		print('x')
	
	# Loop over all possible indicative words of given length len_iwords
	for idxq, qword in enumerate(
		itertools.product(alphabet, repeat = len_iwords)
	):
		# Estimate entries of large column-matrix (gives starting state)
		F_I0_bl[idxq, 0] = estimate(
			sequence, qword, mfn_dict
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