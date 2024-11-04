import numpy as np

from oom import *
from src.oom.DiscreteValuedOOM import get_matrices


if __name__ == '__main__':
	n_obs = 3
	src_dim = 64
	learnlen = 70000
	testlen = 30000
	
	random_oom = DiscreteValuedOOM.from_sparse(
		src_dim,
		density = 0.8,
		alphabet_size = n_obs,
		deterministic_functional = False
	)
	print("\nGenerated OOM\n", random_oom)
	
	dim_search = [8, 16, 32, 64, 96, 128, 160]
	nlls_err_learned = []
	
	_, nlls_ground, seq = random_oom.generate(length = learnlen + testlen)
	_, nlls_ground_test = random_oom.compute(seq[learnlen:])
	
	for idx, target_dim in enumerate(dim_search):
		# Get estimate matrices
		print(f"Target_dim = {target_dim}: k = ", end='')
		
		learned_oom = DiscreteValuedOOM.from_data(
			seq,
			target_dim,
			max_length = int(np.log2(target_dim))
		)
		print("| Done")
		
		# Save difference in nll from ground (min) to learned
		_, nlls_learned_test = learned_oom.compute(seq[learnlen:])
		nlls_err_learned.append(nlls_learned_test[-1])
	
	xlims = [min(dim_search), max(dim_search)]
	
	from matplotlib import pyplot as plt
	plt.figure()
	plt.plot(dim_search, nlls_err_learned,
			 label = r"$\text{NLL}(P_d, P_\text{true})$")
	plt.hlines(y = np.log2(n_obs), xmin=xlims[0], xmax=xlims[1],
			   color='r', ls='--',
			   label = r"$\text{NLL}(P_\text{uniform}, P_\text{true})$")
	plt.hlines(y = nlls_ground_test, xmin=xlims[0], xmax=xlims[1],
			   color='g', ls='--',
			   label = r"$\text{NLL}(P_\text{true}$)")
	plt.xlim(xlims)
	plt.legend()
	plt.show()
		