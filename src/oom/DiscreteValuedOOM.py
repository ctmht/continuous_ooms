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
	
	default_adjustment: dict[str, Union[int, float]] = {
		"margin": 0.005,
		"setbackMargin": -0.3,
		"setbackLength": 2
	}
	
	
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
		super().__init__(dim, linear_functional, operators, start_state)
		self._invalidity_adj = self._set_invalidity_adjustment(invalidity_adjustment)
	
	
	def _set_invalidity_adjustment(
		self,
		invalidity_adjustment: Optional[dict[str, Union[int, float]]]
	) -> dict[str, Union[int, float]]:
		"""
		Set the dictionary of parameters corresponding to the adjustment
		required to run OOMs as invalid observations generators
		"""
		if invalidity_adjustment is None:
			invalidity_adjustment = {}
		inv = {
			"margin": invalidity_adjustment.get(
				"margin", self.default_adjustment["margin"]
			),
			"setbackMargin": invalidity_adjustment.get(
				"setbackMargin", self.default_adjustment["setbackMargin"]
			),
			"setbackLength": invalidity_adjustment.get(
				"setbackLength", self.default_adjustment["setbackLength"]
			)
		}
		return inv
	
	
	def normalize(
		self
	) -> None:
		"""

		"""
		# Normalize start state
		err = self.lin_func * self.start_state
		self.start_state /= err
	
		# Normalize operators
		mu = sum([op.mat for op in self.operators])
		lin_func_err = self.lin_func * mu
		
		for op in self.operators:
			for col in range(op.mat.shape[1]):
				op.mat[:, col] = (op.mat[:, col]
								* self.lin_func[0, col]
								/ lin_func_err[0, col])
	
	
	def generate(
		self,
		length: int = 100
	) -> tuple[Sequence[State], ObsSequence, Sequence[float]]:
		"""
		Use the OOM to generate a observations of a given length from the
		discrete-valued, stationary, ergodic process it models
		
		Args:
			length: the desired generation length
		Returns:
			(statelist, observations): tuple of the visited states 'statelist'
				and corresponding observations 'observations'; every item
				statelist[index] corresponds to observations[index - 1], given
				the starting state statelist[0] corresponds to the "empty
				obs"
		"""
		# Generated observations
		state = self.start_state
		statelist: list[State] = [state]
		observations: list[Observable] = []
		nlls: list[float] = []
		nll: float = 0
		
		for time_step in range(length):
			p_vec = self.lf_on_operators * state
			
			# Invalidity adjustment
			p_vec = np.array(p_vec).flatten()
			delta = np.sum(
				self._invalidity_adj["margin"] - p_vec, where = p_vec < 0
			)
			p_plus = np.sum(
				p_vec, where = p_vec > 0
			)
			nu_ratio = 1 - delta / p_plus
			# print(time_step, p_vec, delta)
			if delta < self._invalidity_adj["setbackMargin"]:
				# Reset by setbackLength and discard what comes after
				time_step -= self._invalidity_adj["setbackLength"]
				state = statelist[-self._invalidity_adj["setbackLength"]]
				statelist = statelist[:-self._invalidity_adj["setbackLength"]]
				observations = observations[:-self._invalidity_adj["setbackLength"]]
	
			# Set negatives to "margin" and adjust valid probabilities
			p_vec[p_vec > 0] *= nu_ratio
			p_vec[p_vec <= 0] = self._invalidity_adj["margin"]
			
			# Choose next obs randomly
			op: Operator = np.random.choice(self.operators, p = p_vec)
			observation = op.observable
			
			# Apply operator to get next state
			state = op(state)
			state = state / (self.lin_func * state)
			
			# Save state and obs
			statelist.append(state)
			observations.append(observation)
			
			nll_step = - p_vec.dot(np.log2(p_vec))
			nll = nll + (nll_step - nll)/max(1, time_step)
			nlls.append(nll)
	
		return statelist, nlls, ObsSequence(observations)
	
	
	def compute(
		self,
		sequence: ObsSequence,
		length_max: Optional[int] = None
	) -> tuple[Sequence[State], Sequence[float]]:
		"""
		
		"""
		# Generated observations
		state = self.start_state
		statelist: list[State] = [state]
		nlls: list[np.array] = []
		nll: float = 0
		
		stop = min(len(sequence), length_max) if length_max else len(sequence)
		for time_step in range(stop):
			p_vec = self.lf_on_operators * state
			
			# Invalidity adjustment
			p_vec = np.array(p_vec).flatten()
			delta = np.sum(
				self._invalidity_adj["margin"] - p_vec, where = p_vec < 0
			)
			p_plus = np.sum(
				p_vec, where = p_vec > 0
			)
			nu_ratio = 1 - delta / p_plus
			# print(time_step, p_vec, delta)
			if delta < self._invalidity_adj["setbackMargin"]:
				# Reset by setbackLength and discard what comes after
				time_step -= self._invalidity_adj["setbackLength"]
				state = statelist[-self._invalidity_adj["setbackLength"]]
				statelist = statelist[:-self._invalidity_adj["setbackLength"]]
	
			# Set negatives to "margin" and adjust valid probabilities
			p_vec[p_vec > 0] *= nu_ratio
			p_vec[p_vec <= 0] = self._invalidity_adj["margin"]
			
			# Choose operator knowing the next obs
			obs_now = sequence[time_step]
			op = self.operators[self.obsnames.index(obs_now)]
			
			# Apply operator to get next state
			state = op(state)
			state = state / (self.lin_func * state)
			
			# Save state and obs
			statelist.append(state)
			
			nll_step = - p_vec.dot(np.log2(p_vec))
			nll = nll + (nll_step - nll)/max(1, time_step)
			nlls.append(nll)
	
		return statelist, nlls
	
	
	def blend(
		self,
		membership_functions: Optional[Sequence[Callable[[np.array], np.array]]]
	) -> 'ContinuousValuedOOM':
		# Blend using membership_functions if given
		# Learn membership_functions otherwise
		
		# unblend
		pass # return ContinuousValuedOOM() #
	
	
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
		strrep += f"alphabet = [" + ', '.join([o.name for o in self.observables])
		strrep += f"]\n"
		
		for op in self.operators:
			strrep += f"    {op.observable.name} operator matrix:\n{op.mat}\n"
		
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
	index_set = set(series_i0.index)
	column_set = set(series_0j.index)

	for matrix in matrices:
		index_set &= set(matrix.index)
		column_set &= set(matrix.columns)
	
	for matrix in matrices:
		# Drop uncommon rows
		matrix_index = set(matrix.index)
		to_drop = matrix_index - index_set
		matrix.drop(to_drop, axis=0, inplace=True)

		# Drop uncommon columns
		matrix_columns = set(matrix.columns)
		to_drop = matrix_columns - column_set
		matrix.drop(to_drop, axis=1, inplace=True)

		# Reorder indices alphabetically and by word length
		matrix.sort_index(axis=1, key=lambda x: x.str.rjust(max_length, 'Z'), inplace=True)
		matrix.sort_index(axis=0, key=lambda x: x.str.rjust(max_length, 'Z'), inplace=True)

	series_index = set(series_i0.index)
	to_drop = series_index - index_set
	series_i0.drop(to_drop, inplace=True)
	series_i0.sort_index(key=lambda x: x.str.rjust(max_length, 'Z'), inplace=True)

	series_index = set(series_0j.index)
	to_drop = series_index - column_set
	series_0j.drop(to_drop, inplace=True)
	series_0j.sort_index(key=lambda x: x.str.rjust(max_length, 'Z'), inplace=True)


def get_matrices(myobs, max_length):
	N = len(myobs)
	L = max_length
	
	F_IzJ = dict(
		zip(
			[0] + [obsname for obsname in myobs.alphabet],
			[{} for _ in range(len(myobs.alphabet) + 1)]
		)
	)
	F_I0 = {}
	F_0J = {}
	
	for k in range(0, 2*L-1 + 1):
		print(k, end = ' ')
		
		# Min/max ranges for 2-splits
		A0 = L - k
		A1 = L
		if A0 <= 0:
			A1 = L + A0 - 1
			A0 = 1
		B0 = A1
		B1 = A0

		# Min/max ranges for 1-splits
		C0 = L - (k-1)
		C1 = L
		if C0 <= 0:
			C1 = L + C0
			C0 = 1
		D0 = C1
		D1 = C0
		
		for start in range(0, N - (2*L + 1 - k)):
			
			for A, B in zip(range(A0, A1+1), range(B0, B1 - 1, -1)):
				# Get each 2-split
				xj = "".join(myobs[start : start + A])
				z = myobs[start + A]
				xi = "".join(myobs[start + A+1 : start + A+1 + B])
				# print(len(xj+z+xi))
	
				# Add to F_IzJ = F_IzJ[z]
				if xi not in F_IzJ[z]:
					F_IzJ[z][xi] = {}
				if xj not in F_IzJ[z][xi]:
					F_IzJ[z][xi][xj] = 0.0
				F_IzJ[z][xi][xj] += 1 / (N - len(xj) - len(xi) + 1)
			
			for C, D in zip(range(C0, C1+1), range(D0, D1 - 1, -1)):
				# Get each 1-split
				xj = "".join(myobs[start : start + C])
				xi = "".join(myobs[start + C : start + C + D])
		
				# Add to F_IJ = F_IzJ[0]
				if xi not in F_IzJ[0]:
					F_IzJ[0][xi] = {}
				if xj not in F_IzJ[0][xi]:
					F_IzJ[0][xi][xj] = 0.0
				F_IzJ[0][xi][xj] += 1 / (N - len(xj) - len(xi) + 1)

				# Add to F_0J
				if xj not in F_0J:
					F_0J[xj] = 0.0
				F_0J[xj] += 1 / (N - len(xj) + 1)

				# Add to F_I0
				if xi not in F_I0:
					F_I0[xi] = 0.0
				F_I0[xi] += 1 / (N - len(xi) + 1)
	
			# Get each characteristic word

	for key, entry in F_IzJ.items():
		F_IzJ[key] = pd.DataFrame.from_dict(
			entry, orient='index', dtype=float
		).fillna(0)
	
	F_I0 = pd.Series(F_I0)
	F_0J = pd.Series(F_0J)
	
	reduce_to_common(list(F_IzJ.values()), F_I0, F_0J, L)

	for obs, matrix in F_IzJ.items():
		F_IzJ[obs] = np.asmatrix(matrix.values)
	F_I0 = np.asmatrix(F_I0.values).T
	F_0J = np.asmatrix(F_0J.values)
	
	return F_IzJ, F_0J, F_I0