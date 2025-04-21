from typing import Any, Optional

import numpy as np
import pandas as pd

from ..discrete_observable import DiscreteObservable
from ..discrete_valued_oom import DiscreteValuedOOM
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
		# Matrices estimated using all c/i words of given maximum length
		sigma, tau_z, omega = spectral_algorithm(
			estimation_routine = estimate_matrices_discrete,
			obs                = obs,
			target_dimension   = target_dimension,
			estimated_matrices = estimated_matrices,
			max_length         = max_length
		)
	elif max_length is None:
		# Matrices estimated using all c/i words of fixed lengths
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
	
				# Add to the discrete_observable's estimate matrices_bl (F_IzJ <-> F_IzJ[z])
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
	
	# Keep only char/ind words that are common between all estimate matrices_bl
	pad_to_make_similar(
		matrices = list(estimate_matrices.values()),
		series_colvec= estimate_column,
		series_rowvec= estimate_row,
		maxlength = max_ci_l
	)
	
	# Convert to NumPy matrices_bl
	for obs, matrix in estimate_matrices.items():
		estimate_matrices[obs] = np.asmatrix(matrix.values)
	estimate_column = np.asmatrix(estimate_column.values).T
	estimate_row = np.asmatrix(estimate_row.values)
	
	return estimate_matrices, estimate_row, estimate_column


def estimate_matrices_discrete_fixed(
	sequence,
	len_cwords: int,
	len_iwords: int,
	indexing: bool = False
):
	"""
	
	"""
	def alphabet(
		seq: list[DiscreteObservable]
	) -> list[DiscreteObservable]:
		return list(sorted(set(seq), key=lambda x: x.uid))
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
	
	denom_rowvec_entries = (seq_l - len_iwords + 1)
	denom_colvec_entries = (seq_l - len_cwords + 1)
	denom_matraw_entries = (seq_l - max_ci_l + 1)
	denom_matobs_entries = (seq_l - (max_ci_l + 1) + 1)
	
	#################################################################################
	
	# iword <-> indicative word (past), cword <-> characteristic word (future)
	# Go through all iwords xbar in X, cwords ybar in Y of their respective lengths
	#     to construct matrices_bl F_X (row), F_Y (column), F_Y,X, F_zY,X
	for start in range(0, seq_l - (max_ci_l + 1) + 1):
		# Get all iword + z + cword combinations in the sequence
		
		# Get iword (xj), discrete_observable (z), and cword (yi)
		xj = "".join([obs.uid for obs in
					  sequence[start : start + len_iwords]])
		z = sequence[start + len_iwords].uid
		yi = "".join([obs.uid for obs in
					  sequence[start + len_iwords + 1 : start + len_iwords + 1 + len_cwords]])

		# Add to the discrete_observable's estimate matrices_bl (F_zY,X <-> estimate_matrices[z])
		if yi not in estimate_matrices[z]:
			estimate_matrices[z][yi] = {}
		if xj not in estimate_matrices[z][yi]:
			estimate_matrices[z][yi][xj] = 0.0
		estimate_matrices[z][yi][xj] += (1 / denom_matobs_entries)
	
	
	for start in range(0, seq_l - max_ci_l + 1):
		# Get all iword + cword combinations in the sequence
		
		# Get ind word (xj) and char word (yi)
		xj = "".join([obs.uid for obs in
					  sequence[start: start + len_iwords]])
		yi = "".join([obs.uid for obs in
					  sequence[start + len_iwords: start + len_iwords + len_cwords]])

		# Add to the regular estimate matrix (F_Y,X <-> estimate_matrices[0])
		if yi not in estimate_matrices[0]:
			estimate_matrices[0][yi] = {}
		if xj not in estimate_matrices[0][yi]:
			estimate_matrices[0][yi][xj] = 0.0
		estimate_matrices[0][yi][xj] += (1 / denom_matraw_entries)

		# Add iword to the row estimates matrix (F_X)
		if xj not in estimate_row:
			estimate_row[xj] = 0.0
		estimate_row[xj] += (1 / denom_rowvec_entries)

		# Add cword to the column estimates matrix (F_Y)
		if yi not in estimate_column:
			estimate_column[yi] = 0.0
		estimate_column[yi] += (1 / denom_colvec_entries)
	
	# The first characteristic and last indicative sequences are unaccounted for
	
	for start in range(0, len_iwords):
		# Get uncounted cwords at the start of the sequence
		yi = "".join([obs.uid for obs in
					  sequence[start: start + len_cwords]])
		# Add cword to the column estimates matrix (F_Y)
		if yi not in estimate_column:
			estimate_column[yi] = 0.0
		estimate_column[yi] += (1 / denom_colvec_entries)
	
	
	for start in range(seq_l - max_ci_l + 1, seq_l - len_iwords + 1):
		# Get uncounted iwords at the end of the sequence
		xj = "".join([obs.uid for obs in
					  sequence[start: start + len_iwords]])
		# Add iword to the row estimates matrix (F_X)
		if xj not in estimate_row:
			estimate_row[xj] = 0.0
		estimate_row[xj] += (1 / denom_rowvec_entries)
	
	#################################################################################
	
	# Convert to dataframes / series
	for key, entry in estimate_matrices.items():
		estimate_matrices[key] = pd.DataFrame.from_dict(
			entry, orient='index', dtype=float
		).fillna(0)
	estimate_column = pd.Series(estimate_column)
	estimate_row = pd.Series(estimate_row)
	
	# Keep only char/ind words that are common between all estimate matrices_bl
	estimate_matrices, estimate_column, estimate_row = pad_to_make_similar(
		matrices = estimate_matrices,
		series_colvec = estimate_column,
		series_rowvec = estimate_row,
		maxlength = max_ci_l
	)
	if indexing: indexes = (estimate_column.index, estimate_row.index)
	
	# Convert to NumPy matrices_bl and rename keys to observables instead of their uids
	estimate_matrices[0] = np.asmatrix(estimate_matrices[0])
	for obs in myobsalphabet:
		estimate_matrices[obs] = np.asmatrix(estimate_matrices.pop(obs.uid))
	estimate_column = np.asmatrix(estimate_column.values).T
	estimate_row = np.asmatrix(estimate_row.values)
	
	if indexing: return estimate_matrices, estimate_row, estimate_column, indexes
	else: return estimate_matrices, estimate_row, estimate_column


def pad_to_make_similar(
	matrices: dict[Any, pd.DataFrame],
	series_colvec: pd.Series,
	series_rowvec: pd.Series,
	maxlength: int
) -> tuple[dict, pd.Series, pd.Series]:
	"""
	
	"""
	# sort_key = lambda x: (x.str.len(), x)  # Tuple (length, word)
	# old_sort_key = lambda x: x.str.rjust(maxlength, 'Z')
	new_sort_key = lambda x: x.str.ljust(maxlength, ' ')
	
	# Create the set of mutual char and ind words between all estimate matrices_bl
	cwords_ex_set = set(series_colvec.index)
	iwords_ex_set = set(series_rowvec.index)

	for idx, matrix in matrices.items():
		iwords_ex_set |= set(matrix.columns)
		cwords_ex_set |= set(matrix.index)
	
	for idx, matrix in matrices.items():
		# Add uncommon rows and columns as 0-rows and 0-columns
		matrices[idx] = matrices[idx].reindex(columns = iwords_ex_set, fill_value = 0)
		matrices[idx] = matrices[idx].reindex(index = cwords_ex_set, fill_value = 0)
		
		# Reorder indices alphabetically and by word length
		matrices[idx].sort_index(axis = 1, key = new_sort_key, inplace = True)
		matrices[idx].sort_index(axis = 0, key = new_sort_key, inplace = True)
	
	# Pad rows with missing ind words in the estimate column vector
	series_colvec = series_colvec.reindex(cwords_ex_set, fill_value = 0)
	series_colvec.sort_index(key = new_sort_key, inplace = True)
	
	# Pad rows with missing char words in the estimate row vector
	series_rowvec = series_rowvec.reindex(iwords_ex_set, fill_value = 0)
	series_rowvec.sort_index(key = new_sort_key, inplace = True)
	
	return matrices, series_colvec, series_rowvec