import sys

import numpy as np
import numpy.linalg
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.oom.util.learning_discrete import estimate_matrices_discrete_fixed
from util_experiments import *


def experiment_discrete_dataset(
	name: str
):
	"""
	
	"""
	_results = []
	
	# Make source
	source_oom = make_source(name)
	d0 = source_oom.dim
	n0 = len(source_oom.observables)
	
	# Generation and lengths at which to assess algorithm
	lengths = [10 ** x for x in range(3, 7 + 1)]
	test_len = 50_000
	
	# Printers
	pbarprint = PbarPrinter(
		indicator = name,
		total = len(lengths) * (2 * d0 - 1),
		desc = f"{name:>20}: Experimental progress",
		position = 0,
		leave = False,
		file = sys.stdout
	)
	pbarprint(f"Generating {max(lengths) + test_len} observations")
	gen = source_oom.generate(max(lengths) + test_len)
	pbarprint(f"Generated")
	
	# Estimate entropy rate of source OOM on test data
	comp_source = source_oom.compute(gen.sequence[-test_len:])
	nll_source  = comp_source.nll_list[-1]
	pbarprint(f"Computed source entropy rate {nll_source}")
	
	for length in lengths:
		# Set up parameters for OOMs to be learned
		params_target = {
			"target_dimension": list(range(2, 2 * d0 + 1))
		}
		
		# All possible options for the length of c/i words to consider
		len_ciw_options = [
			i for i in range(8, 2, -1)
			if 2 * d0 <= n0 ** (2 * i) <= 4e+8
		]
		
		for len_ciw in len_ciw_options:
			# Estimate large matrices
			pbarprint(f"Estimating matrices using L=L_c=L_i={len_ciw} "
					  f"at length {length}")
			estimated_matrices = estimate_matrices_discrete_fixed(
				sequence   = gen.sequence[:length],
				len_cwords = len_ciw,
				len_iwords = len_ciw
			)
			pbarprint(f"Estimated matrices")
			
			# len_ciw might be too large to give good estimates -> linalg error
			_LA_ERR_FLAG = False
			
			for target_pgrid in ParameterGrid(params_target):
				# Learn OOM using these parameters
				d1 = target_pgrid["target_dimension"]
				
				try:
					pbarprint("Learning")
					learned = DiscreteValuedOOM.from_data(
						gen.sequence[:length],
						target_dimension   = d1,
						len_cwords         = len_ciw,
						len_iwords         = len_ciw,
						estimated_matrices = estimated_matrices
					)
				except (numpy.linalg.LinAlgError, ValueError) as e:
					pbarprint(f"Learning error: {e}")
					learned = None
					if "argmax" in str(e):
						# len_ciw indeed too large, try (len_ciw - 1)
						_LA_ERR_FLAG = True
						pbarprint("LinAlg or value Error")
				if _LA_ERR_FLAG:
					pbarprint("Breaking")
					break
				
				assert learned is not None, "OOM not learned"
				
				# Estimate entropy rate of this learned OOM on test data
				with np.errstate(divide='ignore'):
					comp_learned = learned.compute(gen.sequence[-test_len:])
				nll_learned = comp_learned.nll_list[-1]
				
				# Save results for this learned OOM
				here_results = dict(
					name = name, length = length, ci_length = len_ciw,
					**target_pgrid,
					nll_source = nll_source.item(), nll_learned = nll_learned.item()
				)
				_results.append(here_results)
				
				# Print results for this learned OOM
				pbarprint.update(1)
				with np.printoptions(legacy='1.21'):
					pbarprint(f"Results {here_results}")
			
			if _LA_ERR_FLAG:
				# If LinAlgError, then try other len_ciw
				pbarprint("Continuing")
				continue
			else:
				# No LinAlgError, then no need to try other len_ciw
				pbarprint(f"{len_ciw=} is fine")
				break
	
	return _results


if __name__ == '__main__':
	# Get output file name/path
	print("Please remove unnecessary pbarprints first")
	exit(0)
	relative_path = "../../data/experiment_results/"
	outpath = outfile(relative_path, expcase = "discrete", exptype = "dataset")
	
	# Define the sources to be used
	names = ["BINS_2_2", "BINL_2_10", "OCTL_8_10"]
	
	# Run the three sources in parallel
	all_results = []
	try:
		all_results.extend(joblib.Parallel(
			n_jobs = 1
		)(
			joblib.delayed(
				experiment_discrete_dataset
			)(
				name
			)
			for name in names
		))
	finally:
		if all_results:
			res = pd.DataFrame.from_records(
				sum(all_results, [])
			)
			res.to_csv(outpath)