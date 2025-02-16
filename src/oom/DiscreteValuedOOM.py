from typing import Optional, Union, override
from warnings import simplefilter

import numpy as np
import pandas as pd
import scipy as sp

from .OOM import ObservableOperatorModel
from .discrete_observable import DiscreteObservable

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class DiscreteValuedOOM(ObservableOperatorModel):
	""" n/a """
	
	@override
	def __init__(
		self,
		dim: int,
		linear_functional,
		obs_ops: dict[DiscreteObservable | str | int, np.matrix],
		start_state: np.matrix,
		ia: Optional[dict[str, Union[int, float]]] = None,
	):
		"""
		Initialize an OOM for a discrete-valued process
		"""
		super().__init__(
			dim,
			linear_functional,
			obs_ops,
			start_state,
			ia
		)
	
	
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
		
		for obs, op in zip(self.observables, self.operators):
			strrep += f"    {obs} operator matrix:\n{op}\n"
		
		return strrep
	
	
	@staticmethod
	def from_sparse(
		dimension: int,
		density: float,
		alphabet: Optional[list[DiscreteObservable]] = None,
		alphabet_size: Optional[int] = None,
		deterministic_functional: bool = True,
		stationary_state: bool = True,
		seed: Optional[int] = None,
		fix_cols = True,
		fix_rows = True
	) -> 'ObservableOperatorModel':
		"""
		
		"""
		from .util import random_discrete_valued_oom
		
		return random_discrete_valued_oom(
			dimension,
			density,
			alphabet,
			alphabet_size,
			deterministic_functional,
			stationary_state,
			seed,
			fix_cols,
			fix_rows
		)
		# # Use given alphabet or create one
		# if alphabet is None and alphabet_size is None:
		# 	raise ValueError("Either an alphabet or alphabet size must be given.")
		# if alphabet is None:
		# 	alphabet = []
		# 	for uidx in range(alphabet_size):
		# 		name = chr(ord("a") + uidx)
		# 		observable = DiscreteObservable(name)
		# 		alphabet.append(observable)
		#
		# if density < 0 or density > 1:
		# 	raise ValueError("Density must be a real number between 0 and 1.")
		#
		# # Warn about possible issues
		# if not fix_cols or not fix_rows:
		# 	print("The resulting model might not be irreducible!")
		#
		# # Create RNG
		# rng = np.random.default_rng(seed = seed)
		# rvs = sp.stats.uniform(loc = 0.01, scale = 1).rvs
		#
		# _max_attempts = 1000 if not seed else 1
		# _attempt_all = 0
		#
		# while True:
		# 	# Create linear functional (always deterministic at this step)
		# 	sigma: np.matrix = np.asmatrix(np.ones(shape = (1, dimension)))
		#
		# 	# Create dictionary of observables and operators
		# 	obs_ops: dict[DiscreteObservable, np.matrix] = dict.fromkeys(alphabet, None)
		#
		# 	# Give up after _max_attempts
		# 	if _attempt_all == _max_attempts - 1:
		# 		msg = (
		# 			f"Maximum creation attempts reached ({_max_attempts}). Try a "
		# 			f"lower sparsity. OOM generated with modified matrices to yield "
		# 			f"valid operators - this will result in lower sparsities than "
		# 			f"attempted."
		# 		)
		# 		print(msg)
		# 	_attempt_all += 1
		#
		# 	# Create observable operators
		# 	for idx, obs in enumerate(obs_ops.keys()):
		# 		# # Skip observables whose operators have successfully been created
		# 		# if obs_ops[obs] is not None:
		# 		# 	continue
		#
		# 		# Try generating a sparse matrix
		# 		matrix_rep = np.asmatrix(
		# 			sp.sparse.random(
		# 				m = dimension,
		# 				n = dimension,
		# 				density = density,
		# 				random_state = rng,
		# 				data_rvs = rvs
		# 			).toarray()
		# 		)
		#
		# 		# Ensure no columns are entirely 0 inside each operator
		# 		zero_cols = np.isclose(np.sum(matrix_rep, axis = 0), 0).flatten()
		# 		zero_rows = np.isclose(np.sum(matrix_rep, axis = 1), 0).flatten()
		#
		# 		# Select one element on all-zero columns to make nonzero (if fix)
		# 		if fix_cols:
		# 			for colidx in range(dimension):
		# 				if zero_cols[0, colidx]:
		# 					chosen_row = -1
		# 					for rowidx in range(dimension):
		# 						if zero_rows[0, rowidx]:
		# 							chosen_row = rowidx
		# 							zero_rows[0, rowidx] = False
		# 					if chosen_row == -1:
		# 						chosen_row = np.random.randint(
		# 							low = 0,
		# 							high = dimension
		# 						)
		#
		# 					matrix_rep[chosen_row, colidx] = rvs()
		# 					zero_cols[0, colidx] = False
		#
		# 		# Select one element on all-zero rows to make nonzero (if fix)
		# 		if fix_rows:
		# 			for rowidx in range(dimension):
		# 				if zero_rows[0, rowidx]:
		# 					chosen_col = np.random.randint(low = 0, high = dimension)
		# 					matrix_rep[rowidx, chosen_col] = rvs()
		# 					zero_rows[0, rowidx] = False
		#
		# 		# Create operator
		# 		obs_ops[obs] = matrix_rep
		#
		# 	# Valid HMMs have sigma * mu = sigma (for sigma = (1 1 ...))
		# 	mu: np.matrix = np.sum(list(obs_ops.values()), axis = 0)
		# 	mu_notsums = sigma * mu
		#
		# 	for op in obs_ops.values():
		# 		for col in range(dimension):
		# 			if mu_notsums[0, col] != 0:
		# 				op[:, col] /= mu_notsums[0, col]
		#
		# 	if not stationary_state:
		# 		omega = np.asmatrix(np.random.rand(dimension, 1))
		# 		omega /= np.sum(omega)
		# 	else:
		# 		# Get stationary distribution of transition matrix
		# 		mu: np.matrix = np.sum(list(obs_ops.values()), axis = 0)
		# 		eigenvals, eigenvects = np.linalg.eig(mu)
		#
		# 		# Find eigenvectors for eigenvalues close to 1, use first for stationary
		# 		# (there is only one for irreducible aperiodic HMMs... assume it's ok)
		# 		close_to_1_idx = np.isclose(eigenvals, 1)
		# 		target_eigenvect = eigenvects[:, close_to_1_idx]
		# 		target_eigenvect = target_eigenvect[:, 0]
		#
		# 		# Turn the eigenvector elements into probabilites
		# 		omega = target_eigenvect / sum(target_eigenvect)
		# 		omega: np.matrix = np.asmatrix(omega.real).T
		# 		omega[omega == 0] = 1e-5
		#
		# 	if not deterministic_functional:
		# 		warpmat = np.asmatrix(
		# 			sp.stats.special_ortho_group(dimension).rvs()
		# 		)
		# 		sigma = sigma * warpmat.T
		# 		omega = warpmat * omega
		# 		for obs, op in obs_ops.items():
		# 			obs_ops[obs] = warpmat * op * warpmat.T
		#
		# 	random_oom = DiscreteValuedOOM(
		# 		dim               = dimension,
		# 		linear_functional = sigma,
		# 		obs_ops           = obs_ops,
		# 		start_state       = omega
		# 	)
		#
		# 	random_oom.normalize(ones_row = deterministic_functional)
		#
		# 	# One last check
		# 	p_vec_0_sum = np.sum(random_oom.lf_on_operators * random_oom.start_state)
		# 	if not np.isclose(p_vec_0_sum, 1):
		# 		continue
		#
		# 	break
		#
		# return random_oom
	
	
	@override
	@staticmethod
	def from_data(
		obs: list[DiscreteObservable | str | int],
		target_dimension: int,
		len_cwords: Optional[int] = None,
		len_iwords: Optional[int] = None,
		max_length: Optional[int] = None,
		estimated_matrices: Optional[tuple[np.matrix]] = None
	) -> 'DiscreteValuedOOM':
		"""
		
		"""
		from .util import learn_discrete_valued_oom
		
		return learn_discrete_valued_oom(
			obs,
			target_dimension,
			len_cwords,
			len_iwords,
			max_length,
			estimated_matrices
		)
		# # Estimate large matrices
		# if not estimated_matrices:
		# 	estimated_matrices = get_matrices(obs, max_length)
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
		# # Get state vector (learning equation #3)
		# # It holds that type(sigma) == np.matrix
		# omega = C * F_I0
		#
		# learned_oom = DiscreteValuedOOM(
		# 	dim               = target_dimension,
		# 	linear_functional = sigma,
		# 	obs_ops           = tau_z,
		# 	start_state       = omega
		# )
		# learned_oom.normalize()
		#
		# return learned_oom