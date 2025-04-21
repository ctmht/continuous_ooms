import itertools
import math
from typing import Optional

import numpy as np
import scipy as sp
from scipy.stats import rv_continuous
from tqdm import tqdm

import src.oom
from .kronecker_product import kron_vec_prod
from .mf_lookup import _MfLookup
from .numrank import numerical_rank_frob_mid_spec
from .spectral import spectral_algorithm
from ..discrete_observable import DiscreteObservable


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
	# Estimate large matrices_bl
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
	len_cwords: int,
	len_iwords: int,
	membership_functions,
	observables,
	ret_numrank: bool=False,
	T_inv: Optional[np.matrix]=None
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
	seq_l = len(sequence)
	max_ci_l = len_cwords + len_iwords
	
	# Create large matrices_bl F_I(z)J, column vector F_0J, and row vector F_I0
	estimate_matrices_bl = dict(zip(
		[0] + alphabet,
		[np.zeros(shape = (n ** len_iwords, n ** len_cwords))
		 for _ in range(n + 1)]
	))
	estimate_row_bl = np.zeros(shape = (1, n ** len_cwords))
	estimate_column_bl = np.zeros(shape = (n ** len_iwords, 1))
	
	denom_rowvec_entries = (seq_l - len_iwords + 1)
	denom_colvec_entries = (seq_l - len_cwords + 1)
	denom_matraw_entries = (seq_l - max_ci_l + 1)
	denom_matobs_entries = (seq_l - (max_ci_l + 1) + 1)
	
	#################################################################################
	
	# iword <-> indicative word (past), cword <-> characteristic word (future)
	# Go through all iwords xbar in X, cwords ybar in Y of their respective lengths
	#     to construct matrices_bl F_X_bl (row), F_Y_bl (column), F_Y,X_bl , F_zY,X_bl
	for start in tqdm(range(0, seq_l - (max_ci_l + 1) + 1), position=0, leave=True):
		# Ensure lookup contains all values of the current subsequence
		if not mf_lookup.holds_pdfs(start = start, end = start + (max_ci_l + 1) + 1):
			mf_lookup.update_pdfs(sequence, start = start)
		
		# Go through all possible indicative words at this sequence step
		for idxi, iword in enumerate(
			itertools.product(alphabet, repeat = len_iwords)
		):
			# Compute the len_iwords-length membership of iword as the product of all
			# memberships sequentially in iword of the respective sequence elements
			i_base = math.prod([
				mf_lookup[iword[idx], start + idx]
				for idx in range(0, len_iwords)
			])
			
			# Add iword to the row estimates matrix (F_X_bl)
			estimate_row_bl[0, idxi] += (i_base / denom_rowvec_entries)
			
			# Go through all possible observables -> either our z or part of cword
			for idxo, obs in enumerate(alphabet):
				o_incr = mf_lookup[obs, start + len_iwords]
				
				# Go through all possible "characteristic" events (trick, take length-1)
				start_after_i_o = start + len_iwords + 1
				for idxc, cword_reduced in enumerate(
					itertools.product(alphabet, repeat = len_cwords - 1)
				):
					c_incr = math.prod([
						mf_lookup[cword_reduced[idx], start_after_i_o + idx]
						for idx in range(0, len_cwords - 1)
					])
					
					# iword (iword), cword (obs + cword_reduced)
					# Add to the regular estimate matrix (F_Y,X_bl <-> estimate_matrices_bl[0])
					_prod0 = i_base * o_incr * c_incr
					_idxc = idxo * n ** (len_cwords - 1) + idxc
					estimate_matrices_bl[0][_idxc, idxi] += (_prod0 / denom_matraw_entries)
					
					# Add cword to the column estimates matrix (F_Y_bl) (only once for any iword)
					if idxi == 0:
						_prod_tot_cword = o_incr * c_incr
						estimate_column_bl[_idxc] += (_prod_tot_cword / denom_colvec_entries)
					
					start_after_i_o_c0 = start_after_i_o + len_cwords - 1
					# Go through all possible observables again
					for idxo2, obs2 in enumerate(alphabet):
						o2_incr = mf_lookup[obs2, start_after_i_o_c0]
						
						# iword (iword), z (obs), cword (cword_reduced + obs2)
						# Add to the discrete_observable's estimate matrices_bl (F_zY,X_bl <-> estimate_matrices_bl[z])
						_prodobs = _prod0 * o2_incr
						_idxi2 = idxc * n + idxo2
						estimate_matrices_bl[obs][_idxi2, idxi] += (_prodobs / denom_matobs_entries)
	
	# The last subsequence of length (Lc + Li) is unaccounted for
	
	start = seq_l - max_ci_l
	for idxi, iword in enumerate(
		itertools.product(alphabet, repeat = len_iwords)
	):
		# Ensure lookup contains all values of the current subsequence
		if not mf_lookup.holds_pdfs(start = start, end = start + max_ci_l + 1):
			mf_lookup.update_pdfs(sequence, start = start)
		
		# Get all iword + cword combinations in the sequence
		
		i_base = math.prod([
			mf_lookup[iword[idx], start + idx]
			for idx in range(0, len_iwords)
		])
		
		# Add iword to the row estimates matrix (F_X_bl)
		estimate_row_bl[0, idxi] += (i_base / denom_rowvec_entries)
		
		# Go through all possible characteristic events
		for idxc, cword in enumerate(
			itertools.product(alphabet, repeat = len_cwords)
		):
			# Get char word
			c_incr = math.prod([
				mf_lookup[cword[idx], start + len_iwords + idx]
				for idx in range(0, len_cwords)
			])
			
			# Add to the regular estimate matrix (F_Y,X_bl <-> estimate_matrices_bl[0])
			_prod0 = i_base * c_incr
			estimate_matrices_bl[0][idxc, idxi] += (_prod0 / denom_matraw_entries)
			
			# Add cword to the column estimates matrix (F_Y_bl) (only once for any iword)
			if idxi == 0:
				estimate_column_bl[idxc, 0] += (c_incr / denom_colvec_entries)
	
	# The first characteristic and last indicative sequences are unaccounted for
	
	for start in range(0, len_iwords):
		# Ensure lookup contains all values of the current subsequence
		if not mf_lookup.holds_pdfs(start = start, end = start + len_cwords + 1):
			mf_lookup.update_pdfs(sequence, start = start)
		
		# Get uncounted cwords at the start of the sequence
		for idxc, cword in enumerate(
			itertools.product(alphabet, repeat = len_cwords)
		):
			# Get numerator as product of memberships
			c_base = math.prod([
				mf_lookup[cword[idx], start + idx]
				for idx in range(0, len_cwords)
			])
			# Add cword to the column estimates matrix (F_Y_bl)
			estimate_column_bl[idxc, 0] += (c_base / denom_colvec_entries)
	
	
	for start in range(seq_l - max_ci_l + 1, seq_l - len_iwords + 1):
		# Ensure lookup contains all values of the current subsequence
		if not mf_lookup.holds_pdfs(start = start, end = start + len_iwords + 1):
			mf_lookup.update_pdfs(sequence, start = start)
		
		# Get uncounted iwords at the end of the sequence
		for idxi, iword in enumerate(
			itertools.product(alphabet, repeat = len_iwords)
		):
			# Get numerator as product of memberships
			i_base = math.prod([
				mf_lookup[iword[idx], start + idx]
				for idx in range(0, len_iwords)
			])
			# Add iword to the row estimates matrix (F_X_bl)
			estimate_row_bl[0, idxi] += (i_base / denom_rowvec_entries)
	
	#################################################################################
	
	if ret_numrank:
		numrank = numerical_rank_frob_mid_spec(estimate_matrices_bl[0],
											   seqlength = seq_l,
											   len_cwords = len_cwords,
											   len_iwords = len_iwords)
		# numrank = numerical_rank_binomial(estimate_matrices_bl[0],
		# 								  seqlength = seq_l)
	
	# for obs, mat in estimate_matrices_bl.items():
	# 	print(obs)
	# 	print(mat)
	# print(estimate_row_bl)
	# print(estimate_column_bl)
	
	# Applies processing to the blended-case estimate matrices_bl
	estimate_matrices, estimate_row, estimate_column = process_matrices_continuous(
		matrices_bl= estimate_matrices_bl,
		rowvec_bl = estimate_row_bl,
		colvec_bl = estimate_column_bl,
		mfn_dict = mfn_dict,
		len_iwords = len_iwords,
		len_cwords = len_cwords,
		T_inv = T_inv
	)
	
	if ret_numrank:
		return estimate_matrices, estimate_row, estimate_column, numrank
	return estimate_matrices, estimate_row, estimate_column


def process_matrices_continuous(
	matrices_bl,
	rowvec_bl,
	colvec_bl,
	mfn_dict,
	len_iwords: int,
	len_cwords: int,
	T_inv: Optional[np.matrix]=None
):
	alphabet = list(mfn_dict.keys())
	
	if T_inv is None:
		T_inv = get_transfer_matrix(mfn_dict)
	
	matrices = {}
	for key, mat in matrices_bl.items():
		matrices[key] = kron_vec_prod(
			[T_inv for _ in range(len_iwords + len_cwords)],
			mat.flatten(),
			side = "right"
		).reshape(
			len(alphabet) ** len_cwords,
			len(alphabet) ** len_iwords
		)
	
	rowvec = kron_vec_prod(
		[T_inv for _ in range(len_iwords)],
		rowvec_bl.flatten(),
		side = "right"
	).reshape(
		1,
		len(alphabet) ** len_iwords
	)
	
	colvec = kron_vec_prod(
		[T_inv for _ in range(len_cwords)],
		colvec_bl.flatten(),
		side = "right"
	).reshape(
		len(alphabet) ** len_cwords,
		1
	)
	
	return matrices, rowvec, colvec


def get_transfer_matrix(
	mfn_dict,
	save_to_files: bool=False,
) -> np.matrix:
	# TODO: remove hardcoding of density function supports
	# TODO: extend sp.integrate.quad to nquad for higher-dim observations (e.g. in R^2)
	_TOTAL_SUPPORT = [-np.inf, +np.inf]
	_INF_LIM = 1000
	_SKIP = 1
	
	# Create transfer matrix T with entries given by inner products of memberships
	alphabet = list(mfn_dict.keys())
	T = np.zeros(shape = (len(alphabet), len(alphabet)))

	for idx in range(len(alphabet)):
		for jdx in range(len(alphabet)):
			# Integrate nu_i(x) * nu_j(x) between 0 and 1
			T[idx, jdx] = sp.integrate.quad(
				lambda x: mfn_dict[alphabet[idx]].pdf(x) * mfn_dict[alphabet[jdx]].pdf(x),
				_TOTAL_SUPPORT[0], -_INF_LIM
			)[0] + sp.integrate.quad(
				lambda x: mfn_dict[alphabet[idx]].pdf(x) * mfn_dict[alphabet[jdx]].pdf(x),
				_INF_LIM, _TOTAL_SUPPORT[1]
			)[0] + sum([
				sp.integrate.quad(
					lambda x: mfn_dict[alphabet[idx]].pdf(x) * mfn_dict[alphabet[jdx]].pdf(x),
					a, a + _SKIP
				)[0]
				for a in range(-_INF_LIM, _INF_LIM, _SKIP)
			])
	
	if save_to_files:
		with open(f'T_dump_{len(alphabet)}.txt', 'w') as f:
			f.write(str(T))
	
	T_inv = np.linalg.inv(T)
	
	if save_to_files:
		with open(f'T_inv_dump_{len(alphabet)}.txt', 'w') as f:
			f.write(str(T_inv))
	
	return T_inv



# ###################################################################################
# def estimate_matrices_continuous_old(
# 	sequence,
# 	len_cwords,
# 	len_iwords,
# 	membership_functions,
# 	observables
# ):
# 	"""
#
# 	"""
# 	if observables is None:
# 		alphabet = []
# 		for uidx in range(len(membership_functions)):
# 			name = chr(ord("a") + uidx)
# 			observable = DiscreteObservable(name)
# 			alphabet.append(observable)
# 	else:
# 		alphabet = observables
# 	mfn_dict = dict(zip(alphabet, membership_functions))
#
# 	F_IJ_bl = dict(zip(
# 		[0] + alphabet,
# 		[np.zeros(shape = (len(alphabet) ** len_iwords, len(alphabet) ** len_cwords))
# 		 for _ in range(len(alphabet) + 1)]
# 	))
#
# 	F_0J_bl = np.zeros(shape = (1, len(alphabet) ** len_cwords))
#
# 	F_I0_bl = np.zeros(shape = (len(alphabet) ** len_iwords, 1))
#
# 	# Loop over all possible characteristic words of given length len_cwords
# 	for idxc, cword in enumerate(
# 		itertools.product(alphabet, repeat = len_cwords)
# 	):
# 		# Estimate entries of large row-matrix (gives linear functional)
# 		F_0J_bl[0, idxc] = estimate(
# 			sequence, cword, mfn_dict
# 		)
#
# 		# Loop over all possible indicative words of given length len_iwords
# 		for idxq, qword in enumerate(
# 			itertools.product(alphabet, repeat = len_iwords)
# 		):
# 			# Estimate entries of large matrix
# 			F_IJ_bl[0][idxq, idxc] = estimate(
# 				sequence, cword + qword, mfn_dict
# 			)
# 			# print(f"{estimate_matrices_bl[0][idxq, idxc] : <.10f}", cword, qword)
#
# 			# Loop over all possible discrete observables
# 			for obs in alphabet:
# 				# Estimate entries of large observable-indexed matrix
# 				# (gives operators)
# 				F_IJ_bl[obs][idxq, idxc] = estimate(
# 					sequence, cword + (obs,) + qword, mfn_dict
# 				)
# 				# print(f"{estimate_matrices_bl[obs][idxq, idxc] : >12} ", cword, (obs,), qword)
# 		print('x')
#
# 	# Loop over all possible indicative words of given length len_iwords
# 	for idxq, qword in enumerate(
# 		itertools.product(alphabet, repeat = len_iwords)
# 	):
# 		# Estimate entries of large column-matrix (gives starting state)
# 		F_I0_bl[idxq, 0] = estimate(
# 			sequence, qword, mfn_dict
# 		)
#
# 	# Applies processing to the blended-case estimate matrices_bl
# 	estimate_matrices, estimate_row, estimate_column = process_matrices_continuous(
# 		len_iwords,
# 		len_cwords,
# 		mfn_dict,
# 		F_IJ_bl,
# 		F_0J_bl,
# 		F_I0_bl
# 	)
#
# 	return estimate_matrices, estimate_row, estimate_column
#
#
# def estimate(cv_sequence, zbar, memberships):
# 	"""
# 	Computes the system function estimate for a given continuous-valued sequence and known
# 	membership functions, using the Ergodic Theorem for Strictly Stationary Processes
#
# 	Args:
# 		cv_sequence: the continuous-valued sequence
# 		zbar: a word formed of finitely many objects in the alphabet of a symbolic process
# 		memberships: dictionary linking each alphabet entry to its known membership function
# 	"""
# 	est = 0
#
# 	for start in range(len(cv_sequence) - len(zbar) + 1):
# 		est_here = 1
#
# 		# print(f"est_here: {est_here} ", end='')
#
# 		for idx in range(0, len(zbar)):
# 			mf = memberships[zbar[idx]]
# 			x = cv_sequence[start + idx]
# 			est_here *= mf.pdf(x)
# 			# print(f"{est_here} ", end='')
#
# 		est += est_here / (len(cv_sequence) - len(zbar) + 1)
#
# 	return est