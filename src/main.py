import numpy as np
import pandas as pd

from src.oom import *
from src.oom.DiscreteValuedOOM import get_matrices


def experiment_dimension(
	n_obs: int,
	src_dim: int,
	trainlen: int,
	testlen: int,
	dim_search: list[int],
	maxlen_ciw: int,
	sparsity: float,
	repetitions: int = 1,
	verbose: bool = False
):
	# NLL-holding variables
	nll_mean = pd.Series(dict.fromkeys(dim_search, 0), dtype = float)
	nll_var = pd.Series(dict.fromkeys(dim_search, 0), dtype = float)
	
	nll_values = pd.DataFrame(
		index = dim_search,
		columns = range(repetitions),
		dtype = float
	)
	nll_values['sparsity'] = sparsity
	
	rep = 0
	while rep < repetitions:
		try:
			# Create ground OOM
			random_oom = DiscreteValuedOOM.from_sparse(
				alphabet_size = n_obs,
				dimension = src_dim,
				density = 1 - sparsity,
				deterministic_functional = False
			)
			if verbose:
				print("C1: creation, ", end='')
			
			# Generate new sequence
			generate_result_g = random_oom.generate(length = trainlen + testlen)
			seq = generate_result_g.sequence
			if verbose:
				print("C2: generation, ", end='')
			
			# Compute-tv over test sequence
			# (even though we can use the last testlen: of the generation results)
			compute_result_g = random_oom.compute(sequence = seq[trainlen:])
			if verbose:
				print("C3: computation, ", end='')
			
			# Precompute matrix estimates
			# (since all dimensions share them for constant maxlen_ciw)
			preestimated = get_matrices(myobs = seq, max_length = maxlen_ciw)
			if verbose:
				print("C4: matrices", end='')
			
			# Result dictionaries (only for last repetition as of now)
			ss_l_test_alldims = {}
			nlls_l_test_alldims = {}
			ps_l_test_alldims = {}
			
			if verbose:
				print(f"\n    repetition = {rep} | target dimension = ", end='')
			for idx, target_dim in enumerate(dim_search):
				if verbose:
					print(target_dim, end = ' ')
				
				# Learn a model with desired parameters
				learned_oom = DiscreteValuedOOM.from_data(
					obs = seq,
					target_dimension = target_dim,
					max_length = maxlen_ciw,
					estimated_matrices = preestimated
				)
				
				# Compute-tv over test sequence
				compute_result_l = learned_oom.compute(seq[trainlen:])
				ss_l_test = compute_result_l.state_list
				nlls_l_test = compute_result_l.nll_list
				ps_l_test = compute_result_l.p_vecs
				
				# # Delete learned model
				# del learned_oom
				
				# Save to result dictionaries
				ss_l_test_alldims[target_dim] = ss_l_test
				nlls_l_test_alldims[target_dim] = nlls_l_test
				ps_l_test_alldims[target_dim] = ps_l_test
			
			# Add repetition NLL results to mean and (unscaled) variance (online)
			nll_rep = pd.Series(nlls_l_test_alldims).apply(lambda nlls: nlls[-1])
			old_mean = nll_mean
			nll_mean = nll_mean + (nll_rep - nll_mean) / (rep + 1)
			nll_var = nll_var + (nll_rep - nll_mean) * (nll_rep - old_mean)
			
			nll_values[rep] = nll_rep
			
			if verbose:
				print("| Done")
			
			rep += 1
		except ValueError as err:
			if err.args[0] != "probabilities do not sum to 1":
				raise
		except TimeoutError:
			pass
		except:
			print("------------------------------------------------!!!!!!!!")
			raise
	
	# Scale and take sqrt to get standard deviation, if possible
	if repetitions > 1:
		nll_var = nll_var / (repetitions - 1)
		nll_std = nll_var ** 0.5
	else:
		nll_std = pd.Series(dict.fromkeys(dim_search, 0))
	
	return (
		seq,
		generate_result_g,
		compute_result_g,
		(ss_l_test_alldims, nlls_l_test_alldims, ps_l_test_alldims),
		nll_mean,
		nll_std,
		nll_values
	)


if __name__ == '__main__':
	n_obs = 7
	src_dim = 8
	trainlen = 70000
	testlen = 30000
	dim_search = [2, 4, 6, 8, 10, 12, 14]
	maxlen_ciw = 4
	
	results = experiment_dimension(
		n_obs = n_obs,
		src_dim = src_dim,
		trainlen = trainlen,
		testlen = testlen,
		dim_search = dim_search,
		maxlen_ciw = maxlen_ciw,
		sparsity = 0.8,
		repetitions = 10
	)
	
	# n_obs = 4
	# src_dim = 8
	# learnlen = 70000
	# testlen = 30000
	#
	# random_oom = DiscreteValuedOOM.from_sparse(
	# 	src_dim,
	# 	density = 0.3,
	# 	alphabet_size = n_obs,
	# 	deterministic_functional = False
	# )
	# print("\nGenerated OOM\n", random_oom)
	# _, nlls_ground, seq, ps_ground = random_oom.generate(length = learnlen + testlen)
	# _, nlls_ground_test, ps_ground_test = random_oom.compute(seq[learnlen:])
	# del random_oom
	#
	# dim_search = [4, 8, 12, 16, 20, 24, 32]
	# max_length = int(max(np.log2(dim_search))) - 3
	#
	# nlls_err_learned = []
	#
	# for idx, target_dim in enumerate(dim_search):
	# 	learned_oom = DiscreteValuedOOM.from_data(
	# 		seq,
	# 		target_dim,
	# 		max_length = max_length
	# 	)
	#
	# 	# Save difference in nll from ground (min) to learned
	# 	_, nlls_learned_test, ps_learned_test = learned_oom.compute(seq[learnlen:])
	# 	nlls_err_learned.append(nlls_learned_test[-1])
	#
	# 	del learned_oom
	# 	del nlls_learned_test
	#
	# import matplotlib as mpl
	# from matplotlib import pyplot as plt
	# xlims = [min(dim_search), max(dim_search)]
	#
	# fig = plt.figure()
	# ax = plt.gca()
	#
	# # Entropy estimates of learned OOMs on test sequence (by dimension)
	# ax.plot(dim_search, nlls_err_learned,
	# 		 marker='o', markersize=3,
	# 		 label = r"$\hat{H}(P_d \Vert P_\text{true})$")
	#
	# # Entropy of uniform process
	# ax.hlines(y = np.log2(n_obs), xmin=xlims[0], xmax=xlims[1],
	# 		   color='r', ls=(1, (5, 3)), linewidth=2, alpha=0.6,
	# 		   label = r"$\hat{H}(P_\text{uniform})$")
	#
	# # Entropy estimates of ground OOM on test sequence
	# ax.hlines(y = nlls_ground[-1], xmin=xlims[0], xmax=xlims[1],
	# 		   color='g', ls=(0, (4, 3)), linewidth=2, alpha=0.6,
	# 		   label = r"$\hat{H}(P_\text{true}$)")
	#
	# ax.set_xscale('log', base=2)
	# ax.set_xticks(dim_search)
	# ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	# ax.set_xlim(xlims)
	# ax.set_xlabel("Learned OOM dimension ($d$)")
	#
	# ax.set_ylabel("Entropy estimates ($\\text{NLL} \\approx \\hat{H}$)")
	#
	# ax.legend(loc="center right", title=f"True OOM $d = {src_dim}, \\vert\\Sigma\\vert = {n_obs}$")
	# ax.grid(True, alpha=0.5)
	#
	# fig.set_layout_engine("constrained")
	#
	# plt.show()