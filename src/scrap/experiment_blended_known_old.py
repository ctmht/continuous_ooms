from typing import Callable
import sys

import numpy as np
import numpy.linalg
import pandas as pd
from sklearn.model_selection import ParameterGrid

import src.oom
from src.oom.util.learning_continuous import estimate_matrices_continuous
from src.experiments.util_experiments import *


def experiment_continuous_dataset(
	name: str,
	mf_getter_func: Callable
):
	"""
	
	"""
	_results = []
	_estimated_matrices_all = []
	
	# Make source
	discname = "_".join(name.split('_')[:-1])
	source_oom = make_source(discname)
	d0 = source_oom.dim
	n0 = len(source_oom.observables)
	mfs = mf_getter_func(n0)
	source_oom = src.oom.ContinuousValuedOOM.from_discrete_valued_oom(source_oom, mfs)
	
	# Generation and lengths at which to assess algorithm
	lengths = [10 ** x for x in range(3, 6 + 1)]
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
	comp_source = source_oom.compute(gen.sequence_cont[-test_len:])
	nll_source  = comp_source.nll_list[-1]
	pbarprint(f"Computed source entropy rate {nll_source}")
	
	try:
		for length in lengths:
			# Set up parameters for OOMs to be learned
			params_target = {
				"target_dimension": list(range(2, 2 * d0 + 1))
			}
			
			# All possible options for the length of c/i words to consider
			len_ciw_options = [
				i for i in range(6, 2, -1)
				if 2 * d0 <= n0 ** (2 * i) <= 1e+8
			]
			
			for len_ciw in len_ciw_options:
				# Estimate large matrices_bl
				pbarprint(f"Estimating matrices_bl using L=L_c=L_i={len_ciw} "
						  f"at length {length}")
				estimated_matrices = estimate_matrices_continuous(
					sequence             = gen.sequence_cont[:length],
					len_cwords           = len_ciw,
					len_iwords 			 = len_ciw,
					membership_functions = mfs,
					observables          = source_oom.observables
				)
				pbarprint("Estimated matrices_bl")
				_estimated_matrices_all.append(estimated_matrices)
				
				# len_ciw might be too large to give good estimates -> linalg error
				_LA_ERR_FLAG = False
				
				# len_ciw might be too large to give good estimates -> linalg error
				for target_pgrid in ParameterGrid(params_target):
					# Learn OOM using these parameters
					d1 = target_pgrid["target_dimension"]
					
					try:
						pbarprint("Learning")
						learned = ContinuousValuedOOM.from_data(
							gen.sequence_cont[:length],
							target_dimension     = d1,
							len_cwords           = len_ciw,
							len_iwords           = len_ciw,
							membership_functions = mfs,
							observables          = source_oom.observables,
							estimated_matrices   = estimated_matrices
						)
					except (numpy.linalg.LinAlgError, ValueError) as e:
						pbarprint(f"Learning error: {e}")
						learned = None
						if "argmax" in str(e):
							# len_ciw indeed too large, try (len_ciw - 1)
							pbarprint("Empty matrices_bl, len_ciw too large")
							_LA_ERR_FLAG = True
						if "Singular matrix" in str(e):
							pbarprint("Singular matrix, target dimension too large")
							continue
					if _LA_ERR_FLAG:
						pbarprint("Breaking")
						break
					
					assert learned is not None, "OOM not learned"
					
					# Estimate entropy rate of this learned OOM on test data
					with np.errstate(divide='ignore'):
						comp_learned = learned.compute(gen.sequence_cont[-test_len:])
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
	except Exception as e:
		import traceback
		traceback_str = ''.join(traceback.format_tb(e.__traceback__))
		pbarprint(traceback_str)
	finally:
		return _results, _estimated_matrices_all


if __name__ == '__main__':
	# Get output file name/path
	relative_path = "../../data/experiment_results_OLD/"
	outpath = outfile(relative_path, expcase = "blended-known", exptype = "dataset")
	
	# Define the sources to be used
	names = ["BIN2_S2_D100_2G", "BIN2_L10_D100_2G",
			 "BINL2_L10_S20_2G", "OCT8_L10_S20"]
	mf_getter = get_gaussian(seed = 100)
	
	# Run the three sources in parallel
	all_results = []
	try:
		results_here, em_here = joblib.Parallel(
			n_jobs = 1
		)(
			joblib.delayed(
				experiment_continuous_dataset
			)(
				name,
				mf_getter
			)
			for name in names
		)
		all_results.extend(results_here)
	finally:
		if len(all_results) > 0:
			res = pd.DataFrame.from_records(
				sum(all_results, [])
			)
			res.to_csv(outpath)