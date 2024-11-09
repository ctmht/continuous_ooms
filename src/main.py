import numpy as np

from oom import *


if __name__ == '__main__':
	n_obs = 4
	src_dim = 8
	learnlen = 70000
	testlen = 30000
	
	random_oom = DiscreteValuedOOM.from_sparse(
		src_dim,
		density = 0.3,
		alphabet_size = n_obs,
		deterministic_functional = False
	)
	print("\nGenerated OOM\n", random_oom)
	_, nlls_ground, seq, ps_ground = random_oom.generate(length = learnlen + testlen)
	_, nlls_ground_test, ps_ground_test = random_oom.compute(seq[learnlen:])
	del random_oom
	
	dim_search = [4, 8, 12, 16, 20, 24, 32]
	max_length = int(max(np.log2(dim_search))) - 3
	
	nlls_err_learned = []
	
	for idx, target_dim in enumerate(dim_search):
		learned_oom = DiscreteValuedOOM.from_data(
			seq,
			target_dim,
			max_length = max_length
		)
		
		# Save difference in nll from ground (min) to learned
		_, nlls_learned_test, ps_learned_test = learned_oom.compute(seq[learnlen:])
		nlls_err_learned.append(nlls_learned_test[-1])
		
		del learned_oom
		del nlls_learned_test
	
	import matplotlib as mpl
	from matplotlib import pyplot as plt
	xlims = [min(dim_search), max(dim_search)]

	fig = plt.figure()
	ax = plt.gca()
	
	# Entropy estimates of learned OOMs on test sequence (by dimension)
	ax.plot(dim_search, nlls_err_learned,
			 marker='o', markersize=3,
			 label = r"$\hat{H}(P_d \Vert P_\text{true})$")
	
	# Entropy of uniform process
	ax.hlines(y = np.log2(n_obs), xmin=xlims[0], xmax=xlims[1],
			   color='r', ls=(1, (5, 3)), linewidth=2, alpha=0.6,
			   label = r"$\hat{H}(P_\text{uniform})$")
	
	# Entropy estimates of ground OOM on test sequence
	ax.hlines(y = nlls_ground[-1], xmin=xlims[0], xmax=xlims[1],
			   color='g', ls=(0, (4, 3)), linewidth=2, alpha=0.6,
			   label = r"$\hat{H}(P_\text{true}$)")
	
	ax.set_xscale('log', base=2)
	ax.set_xticks(dim_search)
	ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
	ax.set_xlim(xlims)
	ax.set_xlabel("Learned OOM dimension ($d$)")
	
	ax.set_ylabel("Entropy estimates ($\\text{NLL} \\approx \\hat{H}$)")
	
	ax.legend(loc="center right", title=f"True OOM $d = {src_dim}, \\vert\\Sigma\\vert = {n_obs}$")
	ax.grid(True, alpha=0.5)
	
	fig.set_layout_engine("constrained")
	
	plt.show()
		