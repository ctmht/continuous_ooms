from typing import Optional

import numpy as np
import scipy as sp

from ..discrete_observable import DiscreteObservable
from ..DiscreteValuedOOM import DiscreteValuedOOM


def random_discrete_valued_oom(
	dimension: int,
	density: float,
	alphabet: Optional[list[DiscreteObservable]] = None,
	alphabet_size: Optional[int] = None,
	deterministic_functional: bool = True,
	stationary_state: bool = True,
	seed: Optional[int] = None,
	fix_cols = True,
	fix_rows = True
) -> DiscreteValuedOOM:
	"""
	
	"""
	
	# Use given alphabet or create one
	if alphabet is None and alphabet_size is None:
		raise ValueError("Either an alphabet or alphabet size must be given.")
	if alphabet is None:
		alphabet = []
		for uidx in range(alphabet_size):
			name = chr(ord("a") + uidx)
			observable = DiscreteObservable(name)
			alphabet.append(observable)
	
	if density < 0 or density > 1:
		raise ValueError("Density must be a real number between 0 and 1.")
	
	# Warn about possible issues
	if not fix_cols or not fix_rows:
		print("The resulting model might not be irreducible!")
	
	# Create RNG
	rng = np.random.default_rng(seed = seed)
	rvs = sp.stats.uniform(loc = 0.01, scale = 1).rvs
	
	_max_attempts = 1000 if not seed else 1
	_attempt_all = 0
	
	while True:
		# Create linear functional (always deterministic at this step)
		sigma: np.matrix = np.asmatrix(np.ones(shape = (1, dimension)))
		
		# Create dictionary of observables and operators
		obs_ops: dict[DiscreteObservable, np.matrix] = dict.fromkeys(alphabet, None)
		
		# Give up after _max_attempts
		if _attempt_all == _max_attempts - 1:
			msg = (
				f"Maximum creation attempts reached ({_max_attempts}). Try a "
				f"lower sparsity. OOM generated with modified matrices to yield "
				f"valid operators - this will result in lower sparsities than "
				f"attempted."
			)
			print(msg)
		_attempt_all += 1
		
		# Create observable operators
		for idx, obs in enumerate(obs_ops.keys()):
			# # Skip observables whose operators have successfully been created
			# if obs_ops[obs] is not None:
			# 	continue
			
			# Try generating a sparse matrix
			matrix_rep = np.asmatrix(
				sp.sparse.random(
					m = dimension,
					n = dimension,
					density = density,
					random_state = rng,
					data_rvs = rvs
				).toarray()
			)
			
			# Ensure no columns are entirely 0 inside each operator
			zero_cols = np.isclose(np.sum(matrix_rep, axis = 0), 0).flatten()
			zero_rows = np.isclose(np.sum(matrix_rep, axis = 1), 0).flatten()
			
			# Select one element on all-zero columns to make nonzero (if fix)
			if fix_cols:
				for colidx in range(dimension):
					if zero_cols[0, colidx]:
						chosen_row = -1
						for rowidx in range(dimension):
							if zero_rows[0, rowidx]:
								chosen_row = rowidx
								zero_rows[0, rowidx] = False
						if chosen_row == -1:
							chosen_row = np.random.randint(
								low = 0,
								high = dimension
							)
						
						matrix_rep[chosen_row, colidx] = rvs()
						zero_cols[0, colidx] = False
			
			# Select one element on all-zero rows to make nonzero (if fix)
			if fix_rows:
				for rowidx in range(dimension):
					if zero_rows[0, rowidx]:
						chosen_col = np.random.randint(low = 0, high = dimension)
						matrix_rep[rowidx, chosen_col] = rvs()
						zero_rows[0, rowidx] = False
			
			# Create operator
			obs_ops[obs] = matrix_rep
		
		# Valid HMMs have sigma * mu = sigma (for sigma = (1 1 ...))
		mu: np.matrix = np.sum(list(obs_ops.values()), axis = 0)
		mu_notsums = sigma * mu
		
		for op in obs_ops.values():
			for col in range(dimension):
				if mu_notsums[0, col] != 0:
					op[:, col] /= mu_notsums[0, col]
		
		if not stationary_state:
			omega = np.asmatrix(np.random.rand(dimension, 1))
			omega /= np.sum(omega)
		else:
			# Get stationary distribution of transition matrix
			mu: np.matrix = np.sum(list(obs_ops.values()), axis = 0)
			eigenvals, eigenvects = np.linalg.eig(mu)
			
			# Find eigenvectors for eigenvalues close to 1, use first for stationary
			# (there is only one for irreducible aperiodic HMMs... assume it's ok)
			close_to_1_idx = np.isclose(eigenvals, 1)
			target_eigenvect = eigenvects[:, close_to_1_idx]
			target_eigenvect = target_eigenvect[:, 0]
			
			# Turn the eigenvector elements into probabilites
			omega = target_eigenvect / sum(target_eigenvect)
			omega: np.matrix = np.asmatrix(omega.real).T
			omega[omega == 0] = 1e-5
		
		if not deterministic_functional:
			warpmat = np.asmatrix(
				sp.stats.special_ortho_group(dimension).rvs()
			)
			sigma = sigma * warpmat.T
			omega = warpmat * omega
			for obs, op in obs_ops.items():
				obs_ops[obs] = warpmat * op * warpmat.T
		
		random_oom = DiscreteValuedOOM(
			dim               = dimension,
			linear_functional = sigma,
			obs_ops           = obs_ops,
			start_state       = omega
		)
		
		random_oom.normalize(ones_row = deterministic_functional)
		
		# One last check
		p_vec_0_sum = np.sum(random_oom.lf_on_operators * random_oom.start_state)
		if not np.isclose(p_vec_0_sum, 1):
			continue
		
		break
	
	return random_oom