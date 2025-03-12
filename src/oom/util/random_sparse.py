from typing import Optional

import numpy as np
import scipy as sp
from sklearn.utils.extmath import density

from src.oom.discrete_observable import DiscreteObservable
from src.oom.discrete_valued_oom import DiscreteValuedOOM


def random_discrete_valued_oom_OLD(
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


def random_discrete_valued_hmm(
	dimension: int,
	density: float,
	alphabet: Optional[list[DiscreteObservable]] = None,
	alphabet_size: Optional[int] = None,
	seed: Optional[int] = None
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
	
	# Create RNG
	rng = np.random.default_rng(seed = seed)
	rvs = sp.stats.uniform(loc = 0.01, scale = 1).rvs
	
	# Create linear functional (always deterministic at this step)
	sigma: np.matrix = np.asmatrix(np.ones(shape = (1, dimension)))
	
	# Create dictionary of observables and operators
	obs_ops: dict[DiscreteObservable, np.matrix] = dict.fromkeys(alphabet, None)
	
	# Create HMM transition matrix of given density (1 - sparsity)
	mu = _generate_sparse_full_rank_matrix(
		dim = dimension,
		sparsity = 1 - density,
		rng = rng,
		rvs = rvs
	)
	mu = np.asmatrix(mu).T
	
	# Generate observable compound matrix and create observable operators
	Os = _generate_observable_compound(
		nrows = alphabet_size,
		ncols = dimension,
		sparsity = 1 - density,
		rng = rng,
		rvs = rvs
	)
	for idx, obs in enumerate(obs_ops.keys()):
		O_obs = np.asmatrix(np.diag(Os[idx, :]))
		obs_ops[obs] = mu * O_obs
		obs_ops[obs] = np.asmatrix(obs_ops[obs])
	
	# Get stationary distribution of transition matrix
	omega = _get_stationary_state(mu)
	
	random_oom = DiscreteValuedOOM(
		dim               = dimension,
		linear_functional = sigma,
		obs_ops           = obs_ops,
		start_state       = omega
	)
	random_oom.normalize(ones_row = True)
	
	return random_oom
	


def _generate_sparse_full_rank_matrix(
	dim: int,
	sparsity: float,
	rng: np.random.Generator,
	rvs: sp.stats.rv_continuous,
	ret_effective: bool=False
) -> np.matrix | tuple[np.matrix, float]:
	"""
	Generate a full-rank matrix of given sparsity.
	
	Args:
		dim:
		sparsity:
		ret_effective:
	"""
	if sparsity > 1 - 2/dim:
		print("Sparsity may be too high for the generated matrix to have meaningful "
			  "structure")
	
	# Start with a random diagonal matrix (full-rank, sparsity = 1 - 1/d)
	diagonal_entries = rvs(size = dim, random_state = rng)
	M = np.diag(diagonal_entries)
	
	# Calculate the current and target number of non-zero entries
	current_non_zeros = dim
	target_non_zeros = (1 - sparsity) * dim * dim

	# Randomly add non-zero entries until the target is reached
	while current_non_zeros < target_non_zeros:
		# Select and check random row and column
		i, j = rng.integers(0, dim, 2)
		if M[i, j] == 0:
			M[i, j] = rvs(random_state = rng)
			current_non_zeros += 1
			
			# Ensure the matrix remains full-rank or revert change
			if np.linalg.matrix_rank(M) < dim:
				M[i, j] = 0
				current_non_zeros -= 1
	
	# Apply random row and column permutations to vary structure
	row_perm = rng.permutation(dim)
	col_perm = rng.permutation(dim)
	M = M[row_perm, :]  # Permute rows
	M = M[:, col_perm]  # Permute columns
	
	# Make matrix stochastic
	M = M / M.sum(axis = 1, keepdims = True)
	
	if ret_effective:
		eff_sparsity = np.sum(M == 0) / (dim * dim)
		return M, eff_sparsity
	
	return M


def _generate_observable_compound(
	nrows: int,
	ncols: int,
	sparsity: float,
	rng: np.random.Generator,
	rvs: sp.stats.rv_continuous
):
	"""
	Returns the data needed to create the observable matrices
	(column-stochastic matrix)
	
	Args:
		nrows:
		ncols:
	
	(Completed using DeepSeek to get the sparsity calculations and fixing quicker :D)
	"""
	# Step 1: Generate the matrix and apply sparsity
	Os = rvs(size=(nrows, ncols), random_state=rng)
	Os[Os < sparsity] = 0

	# Step 2: Fix fully-zero rows
	while np.any(np.sum(Os, axis=1) == 0):  # Check for fully-zero rows
		zero_row_indices = np.where(np.sum(Os, axis=1) == 0)[0]  # Find fully-zero rows

		# Step 2a: Check if any of these rows correspond to fully-zero columns
		zero_col_indices = np.where(np.sum(Os, axis=0) == 0)[0]  # Find fully-zero columns
		overlap = np.intersect1d(zero_row_indices, zero_col_indices)  # Rows with fully-zero columns

		if len(overlap) > 0:
			# Fix rows that correspond to fully-zero columns
			for row_idx in overlap:
				col_idx = rng.choice(zero_col_indices)  # Randomly select a fully-zero column
				Os[row_idx, col_idx] = rvs(size=1, random_state=rng)  # Add a random value
		else:
			# Step 2b: If no overlap, fix fully-zero rows randomly
			for row_idx in zero_row_indices:
				col_idx = rng.integers(0, ncols)  # Random column index
				Os[row_idx, col_idx] = rvs(size=1, random_state=rng)  # Add a random value

	# Step 3: Fix fully-zero columns (after rows are fixed)
	while np.any(np.sum(Os, axis=0) == 0):  # Check for fully-zero columns
		zero_col_indices = np.where(np.sum(Os, axis=0) == 0)[0]  # Find fully-zero columns
		for col_idx in zero_col_indices:
			row_idx = rng.integers(0, nrows)  # Random row index
			Os[row_idx, col_idx] = rvs(size=1, random_state=rng)  # Add a random value

	# Step 4: Normalize the matrix (optional)
	Os = Os / Os.sum(axis=0, keepdims=True)  # Normalize columns to sum to 1

	return Os


def _get_stationary_state(
	mat: np.matrix
) -> np.matrix:
	"""
	
	"""
	eigenvals, eigenvects = np.linalg.eig(mat)
	
	# Find eigenvectors for eigenvalues close to 1, use first for stationary
	# (there is only one for irreducible aperiodic HMMs... assume it's ok)
	close_to_1_idx = np.isclose(eigenvals, 1)
	target_eigenvect = eigenvects[:, close_to_1_idx]
	target_eigenvect = target_eigenvect[:, 0]
	
	# Turn the eigenvector elements into probabilites
	stat = target_eigenvect / sum(target_eigenvect)
	stat = np.asmatrix(stat.real)
	
	return stat


if __name__ == '__main__':
	with np.printoptions(suppress=True, linewidth=200):
		hmm = random_discrete_valued_hmm(
			dimension = 10,
			density = 0.20,
			alphabet_size = 2,
			seed = 57
		)
		print(hmm)
		
		gen = hmm.generate(100)
		print(gen.sequence[:10])