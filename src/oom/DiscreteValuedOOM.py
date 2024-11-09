from collections.abc import Sequence, Callable
from typing import Optional, Union, Self

import numpy as np
import pandas as pd
import scipy as sp

from .observable import Observable, ObsSequence
from .operator import Operator
from .util import get_matrix_estimates, get_CQ_by_svd
from .OOM import ObservableOperatorModel, State, LinFunctional
from .ContinuousValuedOOM import ContinuousValuedOOM

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class DiscreteValuedOOM(ObservableOperatorModel):
	""" n/a """
	
	def __init__(
		self,
		dim: int,
		linear_functional: LinFunctional,
		operators: Sequence[Operator],
		start_state: State,
		invalidity_adjustment: Optional[dict[str, Union[int, float]]] = None
	):
		"""
		Initialize an OOM for a discrete-valued process
		
		Args:
			dim: the dimension of the stochastic process to be modelled
				by this OOM
			linear_functional: the linear functional associated with the
				OOM, in the form of a 1 x dim matrix (row vector)
			operators: the observable operators corresponding 1:1 with the
				possible observations, given as a list of Operator objects
			start_state: the starting state of the OOM, in the form of
				a dim x 1 matrix (column vector)
			invalidity_adjustment: dictionary containing the margin,
				setbackMargin, and setbackLength parameters for running OOMs
				as invalid observations generators (stepping back in the generation
				process whenever negative probabilities become an issue)
		"""
		super().__init__(
			dim,
			linear_functional,
			operators,
			start_state,
			invalidity_adjustment
		)
	
	
	def blend(
		self,
		membership_functions: Optional[Sequence[Callable[[np.array], np.array]]]
	) -> 'ContinuousValuedOOM':
		# Blend using membership_functions if given
		# Learn membership_functions otherwise
		pass
		# unblend
		# pass # return ContinuousValuedOOM() #
	
	
	def __repr__(
		self
	) -> str:
		"""
		
		"""
		return f"<DiscreteValuedOOM object with dimension {self.dim} "\
			   f"and alphabet size {len(self.observables)}>"
	
	
	def __str__(
		self
	) -> str:
		strrep = self.__repr__() + '\n'
		strrep += f"functional = {self.lin_func.flatten()}\n"
		strrep += f"start state = {self.start_state.flatten()}^T\n"
		strrep += f"alphabet = [" + ', '.join([o.uid for o in self.observables])
		strrep += f"]\n"
		
		for op in self.operators:
			strrep += f"    {op.observable.uid} operator matrix:\n{op.mat}\n"
		
		return strrep
	
	
	@staticmethod
	def from_data(
		obs: ObsSequence,
		target_dimension: int,
		max_length: int = 50,
		estimated_matrices: Optional[tuple[np.matrix]] = None
	) -> 'DiscreteValuedOOM':
		"""
		
		"""
		print(f"{target_dimension=}, ", end='')
		
		# Estimate large matrices
		if not estimated_matrices:
			estimated_matrices = get_matrices(obs, max_length)
		F_IzJ, F_0J, F_I0 = estimated_matrices
		F_IJ = F_IzJ[0]
		
		# Get characterizer and inductor matrices spectrally
		C, Q = get_CQ_by_svd(F_IJ, target_dimension)
		
		# Inverse matrix computation
		V = C * F_IJ * Q
		V_inv = np.linalg.inv(V)
	
		# Get linear functional (learning equation #1)
		# It holds that type(sigma) == LinFunctional == np.matrix
		sigma = F_0J * Q * V_inv
	
		# Get observable operators (learning equation #2)
		tau_z = []
		for obsname, F_IkJ in F_IzJ.items():
			if obsname == 0:
				continue
			
			operator_matrix = C * F_IkJ * Q * V_inv
			operator = Operator(Observable(obsname[1 :]),
								range_dimension = target_dimension,
								matrix_rep = operator_matrix)
			tau_z.append(operator)
		# Get state vector (learning equation #3)
		# It holds that type(sigma) == State == np.matrix
		omega = C * F_I0
		
		learned_oom = DiscreteValuedOOM(
			dim               = target_dimension,
			linear_functional = sigma,
			operators         = tau_z,
			start_state       = omega
		)
		learned_oom.normalize()
		return learned_oom
	
	
	@staticmethod
	def from_sparse(
		dimension: int,
		density: float,
		alphabet: Optional[Sequence[Observable]] = None,
		alphabet_size: Optional[int] = None,
		deterministic_functional: bool = True,
		random_state: Optional[int] = None
	) -> 'DiscreteValuedOOM':
		"""
		
		"""
		if alphabet is None and alphabet_size is None:
			raise ValueError("Either an alphabet or alphabet size must be given.")
		if density < 0 or density > 1:
			raise ValueError("Sparsity must be a real number between 0 and 1.")
		
		# Create linear functional
		sigma: LinFunctional = np.asmatrix(
			np.ones(shape = (1, dimension)) if deterministic_functional
											else np.random.rand(1, dimension)
		)
		
		# Use given alphabet or create one
		if alphabet is not None:
			alphabet_size = len(alphabet)
		else:
			alphabet = []
			for uidx in range(alphabet_size):
				name = chr(ord("a") + uidx)
				observable = Observable(name)
				alphabet.append(observable)
		
		# Create operators
		while True:
			operators = []
			for idx, obs in enumerate(alphabet):
				matrix_rep = np.asmatrix(
					sp.sparse.random(
						m = dimension,
						n = dimension,
						density = density,
						random_state = random_state + idx if random_state else None
					).toarray()
				)
				operator = Operator(obs, dimension, matrix_rep)
				operators.append(operator)
			
			# Normalize operators to get valid HMM
			mu: np.matrix = sum([op.mat for op in operators])
			mu_notsums = sigma * mu
			
			# Need irreducible HMM => no columns of probability 0
			# (and it cant be normalized anyways)
			if np.sum(np.isclose(mu_notsums, 0)) != 0:
				continue
			
			for op in operators:
				for col in range(dimension):
					op.mat[:, col] = (op.mat[:, col]
									  * sigma[0, col]
									  / mu_notsums[0, col])
			break
		
		# Get stationary distribution of transition matrix
		mu: np.matrix = sum([op.mat for op in operators])
		eigenvals, eigenvects = np.linalg.eig(mu)
		
		# Find eigenvectors for eigenvalues close to one, use first for stationary
		# (and there is only one for irreducible aperiodic HMMs... assume it's ok)
		close_to_1_idx = np.isclose(eigenvals, 1)
		target_eigenvect = eigenvects[:, close_to_1_idx]
		target_eigenvect = target_eigenvect[:, 0]
		
		# Turn the eigenvector elements into probabilites
		omega = target_eigenvect / sum(target_eigenvect)
		omega: State = np.asmatrix(omega.real)
		
		random_oom = DiscreteValuedOOM(
			dim               = dimension,
			linear_functional = sigma,
			operators         = operators,
			start_state       = omega
		)
		random_oom.normalize()
		return random_oom



def reduce_to_common(matrices: list[pd.DataFrame], series_i0, series_0j, max_length):
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


def get_matrices(myobs, max_length):
	"""
	
	"""
	seq_l = len(myobs)
	max_ci_l = max_length
	
	estimate_matrices = dict(
		zip(
			[0] + [obs for obs in myobs.alphabet],
			[{} for _ in range(len(myobs.alphabet) + 1)]
		)
	)
	estimate_column = {}
	estimate_row = {}
	
	print(f"{max_length=}: substr_len = ", end='')
	for substr_len in range(2, 2 * max_ci_l+1 + 1):
		print(substr_len, end = ' ')
		
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
				# Get char word (xj), observable (z), and ind word (xi)
				xj = "".join(myobs[start : start + clen])
				z = "".join(myobs[start + clen])
				xi = "".join(myobs[start + clen + 1 : start + clen + 1 + ilen])
	
				# Add to the observable's estimate matrices (F_IzJ <-> F_IzJ[z])
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
				xj = "".join(myobs[start : start + clen])
				xi = "".join(myobs[start + clen : start + clen + ilen])
		
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
	
	print("| Done")
	
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