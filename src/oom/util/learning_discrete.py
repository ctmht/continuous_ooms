from typing import Optional

import numpy as np
import pandas as pd

from ..discrete_observable import DiscreteObservable
from ..DiscreteValuedOOM import DiscreteValuedOOM
from .spectral import spectral_algorithm


def learn_discrete_valued_oom(
	obs: list[DiscreteObservable | str | int],
	target_dimension: int,
	len_cwords: Optional[int] = None,
	len_iwords: Optional[int] = None,
	max_length: Optional[int] = None,
	estimated_matrices: Optional[tuple[np.matrix]] = None,
) -> 'DiscreteValuedOOM':
	"""
	
	"""
	if (len_cwords is None or len_iwords is None) and max_length is not None:
		sigma, tau_z, omega = spectral_algorithm(
			estimation_routine = estimate_matrices_discrete,
			obs                = obs,
			target_dimension   = target_dimension,
			estimated_matrices = estimated_matrices,
			max_length         = max_length
		)
	elif max_length is None:
		sigma, tau_z, omega = spectral_algorithm(
			estimation_routine = estimate_matrices_discrete_fixed,
			obs                = obs,
			target_dimension   = target_dimension,
			estimated_matrices = estimated_matrices,
			len_cwords 		   = len_cwords,
			len_iwords         = len_iwords
		)
	else:
		raise ValueError("Please provide either fixed characteristic and indicative "
						 "word lengths, or a maximum concatenated substring length")
	
	learned_oom = DiscreteValuedOOM(
		dim               = target_dimension,
		linear_functional = sigma,
		obs_ops           = tau_z,
		start_state       = omega
	)
	learned_oom.normalize()
	
	return learned_oom
	# # Estimate large matrices
	# if not estimated_matrices:
	# 	estimated_matrices = estimate_matrices_discrete(
	# 		obs,
	# 		max_length
	# 	)
	# F_IzJ, F_0J, F_I0 = estimated_matrices
	# F_IJ = F_IzJ[0]
	#
	# # Get characterizer and inductor matrices spectrally
	# C, Q = get_CQ_by_svd(F_IJ, target_dimension)
	#
	# # Inverse matrix computation
	# V = C * F_IJ * Q
	# V_inv = np.linalg.inv(V)
	#
	# # Get linear functional (learning equation #1)
	# # It holds that tvtype(sigma) == LinFunctional == np.matrix
	# sigma = F_0J * Q * V_inv
	#
	# # Get discrete_observable operators (learning equation #2)
	# tau_z = {}
	# for obsname, F_IkJ in F_IzJ.items():
	# 	if obsname == 0:
	# 		continue
	#
	# 	operator_matrix = C * F_IkJ * Q * V_inv
	#
	# 	operator = operator_matrix
	# 	obs = DiscreteObservable(obsname[1:])
	#
	# 	tau_z[obs] = operator
	#
	# # Get state vector (learning equation #3)
	# # It holds that type(sigma) == np.matrix
	# omega = C * F_I0


def estimate_matrices_discrete(
	sequence,
	max_length: int
):
	"""
	
	"""
	def alphabet(
		seq: list[DiscreteObservable]
	) -> list[DiscreteObservable]:
		return list(set(seq))
	myobsalphabet = alphabet(sequence)
	
	seq_l = len(sequence)
	max_ci_l = max_length
	
	estimate_matrices = dict(
		zip(
			[0] + [obs.uid for obs in myobsalphabet],
			[{} for _ in range(len(myobsalphabet) + 1)]
		)
	)
	estimate_column = {}
	estimate_row = {}
	
	# Go through all char/ind word combinations such that their concatenated length
	# adds to substr_len
	for substr_len in range(2, 2 * max_ci_l+1 + 1):
		# print(substr_len, end = ' ')
		
		# Min/max ranges for 2-splits
		char_lmin2 = substr_len - max_ci_l - 1
		char_lmax2 = max_ci_l
		if char_lmin2 <= 0:
			char_lmax2 = max_ci_l + char_lmin2 - 1
			char_lmin2 = 1
		ind_lmax2 = char_lmax2
		ind_lmin2 = char_lmin2

		# Min/max ranges for 1-splits
		char_lmin1 = substr_len - max_ci_l
		char_lmax1 = max_ci_l
		if char_lmin1 <= 0:
			char_lmax1 = max_ci_l + char_lmin1
			char_lmin1 = 1
		ind_lmax1 = char_lmax1
		ind_lmin1 = char_lmin1
		
		for start in range(0, seq_l - substr_len):
			# Get each valid 2-split of total length substr_len
			for clen, ilen in zip(
				range(char_lmin2, char_lmax2 + 1),
				range(ind_lmax2, ind_lmin2 - 1, -1)
			):
				# Get char word (xj), discrete_observable (z), and ind word (xi)
				xj = "".join([obs.uid for obs in
							  sequence[start: start + clen]])
				z = sequence[start + clen].uid
				xi = "".join([obs.uid for obs in
							  sequence[start + clen + 1: start + clen + 1 + ilen]])
	
				# Add to the discrete_observable's estimate matrices (F_IzJ <-> F_IzJ[z])
				if xi not in estimate_matrices[z]:
					estimate_matrices[z][xi] = {}
				if xj not in estimate_matrices[z][xi]:
					estimate_matrices[z][xi][xj] = 0.0
				estimate_matrices[z][xi][xj] += 1 / (seq_l - len(xj) - len(xi) + 1)
			
			# Get each valid 1-split of total length substr_len
			for clen, ilen in zip(
				range(char_lmin1, char_lmax1 + 1),
				range(ind_lmax1, ind_lmin1 - 1, -1)
			):
				# Get char word (xj) and ind word (xi)
				xj = "".join([obs.uid for obs in
							  sequence[start: start + clen]])
				xi = "".join([obs.uid for obs in
							  sequence[start + clen: start + clen + ilen]])
		
				# Add to the regular estimate matrix (F_IJ <-> F_IzJ[0])
				if xi not in estimate_matrices[0]:
					estimate_matrices[0][xi] = {}
				if xj not in estimate_matrices[0][xi]:
					estimate_matrices[0][xi][xj] = 0.0
				estimate_matrices[0][xi][xj] += 1 / (seq_l - len(xj) - len(xi) + 1)

				# Add to the row estimates matrix (F_0J)
				if xj not in estimate_row:
					estimate_row[xj] = 0.0
				estimate_row[xj] += 1 / (seq_l - len(xj) + 1)

				# Add to the column estimates matrix (F_I0)
				if xi not in estimate_column:
					estimate_column[xi] = 0.0
				estimate_column[xi] += 1 / (seq_l - len(xi) + 1)
	
	# print("| Done", end = '')
	
	# Convert to dataframes / series
	for key, entry in estimate_matrices.items():
		estimate_matrices[key] = pd.DataFrame.from_dict(
			entry, orient='index', dtype=float
		).fillna(0)
	estimate_column = pd.Series(estimate_column)
	estimate_row = pd.Series(estimate_row)
	
	# Keep only char/ind words that are common between all estimate matrices
	reduce_to_common(
		matrices = list(estimate_matrices.values()),
		series_i0 = estimate_column,
		series_0j = estimate_row,
		max_length = max_ci_l
	)
	
	# Convert to NumPy matrices
	for obs, matrix in estimate_matrices.items():
		estimate_matrices[obs] = np.asmatrix(matrix.values)
	estimate_column = np.asmatrix(estimate_column.values).T
	estimate_row = np.asmatrix(estimate_row.values)
	
	return estimate_matrices, estimate_row, estimate_column


def estimate_matrices_discrete_fixed(
	sequence,
	len_cwords: int,
	len_iwords: int
):
	"""
	
	"""
	def alphabet(
		seq: list[DiscreteObservable]
	) -> list[DiscreteObservable]:
		return list(set(seq))
	myobsalphabet = alphabet(sequence)
	
	seq_l = len(sequence)
	max_ci_l = len_cwords + len_iwords
	
	estimate_matrices = dict(
		zip(
			[0] + [obs.uid for obs in myobsalphabet],
			[{} for _ in range(len(myobsalphabet) + 1)]
		)
	)
	estimate_column = {}
	estimate_row = {}
	
	# Go through all char/ind words of desired fixed lengths
	for start in range(0, seq_l - max_ci_l):
		# Get all cword + z + iword combinations in the sequence
		
		# Get char word (xj), discrete_observable (z), and ind word (xi)
		xj = "".join([obs.uid for obs in
					  sequence[start : start + len_cwords]])
		z = sequence[start + len_cwords].uid
		xi = "".join([obs.uid for obs in
					  sequence[start + len_cwords + 1 : start + len_cwords + 1 + len_iwords]])

		# Add to the discrete_observable's estimate matrices (F_IzJ <-> F_IzJ[z])
		if xi not in estimate_matrices[z]:
			estimate_matrices[z][xi] = {}
		if xj not in estimate_matrices[z][xi]:
			estimate_matrices[z][xi][xj] = 0.0
		estimate_matrices[z][xi][xj] += 1 / (seq_l - max_ci_l + 1)
	
	for start in range(0, seq_l - max_ci_l + 1):
		# Get all cword + iword combinations in the sequence
		
		# Get char word (xj) and ind word (xi)
		xj = "".join([obs.uid for obs in
					  sequence[start: start + len_cwords]])
		xi = "".join([obs.uid for obs in
					  sequence[start + len_cwords: start + len_cwords + len_iwords]])

		# Add to the regular estimate matrix (F_IJ <-> F_IzJ[0])
		if xi not in estimate_matrices[0]:
			estimate_matrices[0][xi] = {}
		if xj not in estimate_matrices[0][xi]:
			estimate_matrices[0][xi][xj] = 0.0
		estimate_matrices[0][xi][xj] += 1 / (seq_l - max_ci_l + 1)

		# Add to the row estimates matrix (F_0J)
		if xj not in estimate_row:
			estimate_row[xj] = 0.0
		estimate_row[xj] += 1 / (seq_l - len(xj) + 1)

		# Add to the column estimates matrix (F_I0)
		if xi not in estimate_column:
			estimate_column[xi] = 0.0
		estimate_column[xi] += 1 / (seq_l - len(xi) + 1)
	
	# print("| Done", end = '')
	
	# Convert to dataframes / series
	for key, entry in estimate_matrices.items():
		estimate_matrices[key] = pd.DataFrame.from_dict(
			entry, orient='index', dtype=float
		).fillna(0)
	estimate_column = pd.Series(estimate_column)
	estimate_row = pd.Series(estimate_row)
	
	# Keep only char/ind words that are common between all estimate matrices
	reduce_to_common(
		matrices = list(estimate_matrices.values()),
		series_i0 = estimate_column,
		series_0j = estimate_row,
		max_length = max_ci_l
	)
	
	# Convert to NumPy matrices and rename keys to observables instead of their uids
	estimate_matrices[0] = np.asmatrix(estimate_matrices[0])
	for obs in myobsalphabet:
		estimate_matrices[obs] = np.asmatrix(estimate_matrices.pop(obs.uid))
	estimate_column = np.asmatrix(estimate_column.values).T
	estimate_row = np.asmatrix(estimate_row.values)
	
	return estimate_matrices, estimate_row, estimate_column


def reduce_to_common(
	matrices: list[pd.DataFrame],
	series_i0,
	series_0j,
	max_length
):
	"""
	
	"""
	# Create the set of mutual char and ind words between all estimate matrices
	index_set = set(series_i0.index)
	column_set = set(series_0j.index)

	for matrix in matrices:
		index_set &= set(matrix.index)
		column_set &= set(matrix.columns)
	
	for matrix in matrices:
		# Drop uncommon rows
		matrix_index = set(matrix.index)
		to_drop = matrix_index - index_set
		matrix.drop(to_drop, axis = 0, inplace = True)

		# Drop uncommon columns
		matrix_columns = set(matrix.columns)
		to_drop = matrix_columns - column_set
		matrix.drop(to_drop, axis = 1, inplace = True)

		# Reorder indices alphabetically and by word length
		matrix.sort_index(
			axis = 1, key = lambda x: x.str.rjust(max_length, 'Z'), inplace = True
		)
		matrix.sort_index(
			axis = 0, key = lambda x: x.str.rjust(max_length, 'Z'), inplace = True
		)
	
	# Drop rows of uncommon ind words from the estimate column vector
	series_index = set(series_i0.index)
	to_drop = series_index - index_set
	series_i0.drop(to_drop, inplace = True)
	series_i0.sort_index(
		key = lambda x: x.str.rjust(max_length, 'Z'), inplace = True
	)
	
	# Drop rows of uncommon char words from the estimate row vector
	series_index = set(series_0j.index)
	to_drop = series_index - column_set
	series_0j.drop(to_drop, inplace = True)
	series_0j.sort_index(
		key = lambda x: x.str.rjust(max_length, 'Z'), inplace = True
	)
	
	return