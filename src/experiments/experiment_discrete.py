import sys

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
	lengths = [10 ** x for x in range(3, 8)]
	test_len = 50_000
	
	# Printers
	pbarprint = PbarPrinter(
		total = len(lengths) * (2 * d0 - 1),
		desc = f"{name:>20}: Experimental progress",
		position = 0,
		leave = False,
		file = sys.stdout
	)
	pbarprint(f"{name}: Generating {max(lengths) + test_len} observations")
	gen = source_oom.generate(max(lengths) + test_len)
	pbarprint(f"{name}: Generated")
	
	# Estimate entropy rate of source OOM on test data
	comp_source = source_oom.compute(gen.sequence[-test_len:])
	nll_source  = comp_source.nll_list[-1]
	pbarprint(f"{name}: Computed source entropy rate {nll_source}")
	
	try:
		for length in lengths:
			# Set up parameters for OOMs to be learned
			params_target = {
				"target_dimension": list(range(2, 2 * d0 + 1))
			}
			
			# All possible options for the length of c/i words to consider
			len_ciw_options = [
				i for i in range(8, 2, -1)
				if 2 * d0 <= n0 ** (2 * i) <= 1e+8
			]
			for len_ciw in len_ciw_options:
				# Estimate large matrices
				pbarprint(f"{name}: Estimating matrices using L=L_c=L_i={len_ciw} "
						  f"at length {length}")
				estimated_matrices = estimate_matrices_discrete_fixed(
					sequence   = gen.sequence[:length],
					len_cwords = len_ciw,
					len_iwords = len_ciw
				)
				pbarprint(f"{name}: Estimated matrices")
				
				# len_ciw might be too large to give good estimates -> linalg error
				try:
					for target_pgrid in ParameterGrid(params_target):
						# Learn OOM using these parameters
						d1 = target_pgrid["target_dimension"]
						learned = DiscreteValuedOOM.from_data(
							gen.sequence[:length],
							target_dimension   = d1,
							len_cwords         = len_ciw,
							len_iwords         = len_ciw,
							estimated_matrices = estimated_matrices
						)
						
						# Estimate entropy rate of this learned OOM on test data
						with np.errstate(divide='ignore'):
							comp_learned = learned.compute(gen.sequence[-test_len:])
						nll_learned = comp_learned.nll_list[-1]
						
						# Save results for this learned OOM
						here_results = dict(
							name = name, length = length, ci_length = len_ciw,
							**target_pgrid,
							nll_source = nll_source, nll_learned = nll_learned
						)
						_results.append(here_results)
						
						# Print results for this learned OOM
						pbarprint.update(1)
						pbarprint(here_results)
				
				except (numpy.linalg.LinAlgError, ValueError) as e:
					if "argmax" in str(e):
						# len_ciw indeed too large, try the next largest (len_ciw-1)
						continue
					else:
						raise
				
				# If no LinAlgError, then no need to try other len_ciw
				break
	finally:
		return _results


if __name__ == '__main__':
	# Get output file name/path
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