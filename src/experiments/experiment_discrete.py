import sys
import random

import pandas as pd

from src.oom.util import numerical_rank_frob_mid_spec
from src.oom.util.few_step_prediction import fix_pvec, kl_divergence, mse, \
	quantify_distribution
from src.oom.util.learning_discrete import estimate_matrices_discrete_fixed
from src.experiments.util_experiments import *


_RESULTS = []													# Final results


def experiment_discrete(
	name: Literal["S_3", "S_5"],
	cword_length: int,
	iword_length: int,
	lengths: list[int],
	test_len: int = 50_000,
	qdist_length: int = 3,
	repetition: int = 0
):
	"""
	
	"""
	# Printers
	pbarprint = PbarPrinter(
		indicator = name + f", rep_{repetition}",
		total = len(lengths),
		desc = f"{name:>10}, rep{repetition}: Experimental progress",
		position = 0,
		leave = False,
		file = sys.stdout
	)
	
	global _RESULTS
	
	# Make source
	case = "discrete"
	source_oom: DiscreteValuedOOM = make_source(name, case="discrete")
	d0 = source_oom.dim
	n0 = len(source_oom.observables)
	
	_SOURCE_PARS = dict(case=case, name=name, d0=d0, n0=n0)			# Dict of stats
	
	# Perform repetition
	pbarprint(f"Repetition {repetition} started")
	
	# Generate sequence of the maximum length (all steps will be on subsequences)
	pbarprint(f"Generating {max(lengths) + test_len} observations")
	gen = source_oom.generate(max(lengths) + test_len)
	pbarprint(f"Generating {max(lengths) + test_len} observations: Finished")
	
	# Estimate entropy rate of source OOM on test data
	pbarprint(f"Computing source OOM entropy rate")
	comp_source = source_oom.compute(gen.sequence[-test_len:])
	nll_source  = comp_source.nll_list[-1]
	pbarprint(f"Computing source OOM entropy rate: Finished, NLL_source = {nll_source}")
	
	# Compute distribution of few-step prediction for source OOM
	pbarprint(f"Computing source OOM few-step prediction distribution at {qdist_length}-steps")
	fsp_vec_source = quantify_distribution(
		steps = qdist_length,
		state = source_oom.start_state,
		operators = source_oom.operators,
		lin_func = source_oom.lin_func
	)
	fsp_vec_source = fix_pvec(fsp_vec_source)
	pbarprint(f"Computing source OOM few-step prediction distribution at {qdist_length}-steps: Finished")
	
	_SOURCE_METRICS = dict(repetition=repetition, nll_source=nll_source,
						   steps_fsp=qdist_length, fsp_source=fsp_vec_source)
	
	for length in lengths:
		# Estimate large matrices
		pbarprint(f"Estimating matrices using L_c = {cword_length}, L_i = {iword_length} at length {length}")
		estimated_matrices = estimate_matrices_discrete_fixed(
			sequence   = gen.sequence[:length],
			len_cwords = cword_length,
			len_iwords = iword_length
		)
		rankmax = np.linalg.matrix_rank(estimated_matrices[0][0])
		ranknum = numerical_rank_frob_mid_spec(estimated_matrices[0][0],
											   seqlength = length,
											   len_cwords = cword_length,
											   len_iwords = iword_length)
		pbarprint(f"Estimating matrices using L_c = {cword_length}, L_i = {iword_length} at length {length}: "
				  f"Finished, rank(F_IJ) = {rankmax}, numrank(F_IJ) = {ranknum}")
		
		# d1_range = range(2, rankmax + 1)
		# d1_range = [x for x in range(ranknum-3, ranknum+3) if 2 <= x <= rankmax]
		d1_range = [x for x in range(2, 15) if 2 <= x <= rankmax]
		for d1 in d1_range:
			# Learn OOM using these parameters
			pbarprint(f"Learning")
			learned = DiscreteValuedOOM.from_data(
				gen.sequence[:length],
				target_dimension   = d1,
				len_cwords         = cword_length,
				len_iwords         = iword_length,
				estimated_matrices = estimated_matrices
			)
			assert learned is not None, "OOM not learned"
			pbarprint(f"Learning: Finished")
			
			_LEARNED_PARS = dict(seqlength=length, cwlen=cword_length,
								 iwlen=iword_length, rankmax=rankmax,
								 ranknum=ranknum, d1=d1, n1=n0)
			
			# Estimate entropy rate of this learned OOM on test data
			pbarprint(f"Computing learned OOM entropy rate")
			with np.errstate(divide='ignore'):
				comp_learned = learned.compute(gen.sequence[-test_len:])
			nll_learned = comp_learned.nll_list[-1]
			pbarprint(f"Computing learned OOM entropy rate: Finished, NLL_learned = {nll_learned}")
			
			# Compute distribution of few-step prediction for learned OOM
			pbarprint(f"Computing learned OOM few-step prediction distribution at {qdist_length}-steps")
			fsp_vec_learned = quantify_distribution(
				steps = qdist_length,
				state = learned.start_state,
				operators = learned.operators,
				lin_func = learned.lin_func
			)
			fsp_vec_learned = fix_pvec(fsp_vec_learned)
			kldiv = kl_divergence(fsp_vec_source, fsp_vec_learned)
			mse_fsp = mse(fsp_vec_source, fsp_vec_learned)
			pbarprint(f"Computing learned OOM few-step prediction distribution at {qdist_length}-steps: "
					  f"Finished, KL(source, learned) = {kldiv}, MSE = {mse_fsp}")
			
			_LEARNED_METRICS = dict(nll_learned=nll_learned,
									fsp_learned=fsp_vec_learned,
									kl_divergence=kldiv,
									mse=mse_fsp)
			
			# Save results for this learned OOM
			_HERE_RESULTS = dict(**_SOURCE_PARS, **_SOURCE_METRICS,
								 **_LEARNED_PARS, **_LEARNED_METRICS)
			_RESULTS.append(_HERE_RESULTS)
			
			# Print results for this learned OOM
			pbarprint.update(1)
			with np.printoptions(legacy='1.21'):
				pbarprint(f"Results {without(_HERE_RESULTS, 'fsp_source', 'fsp_learned')}")
	
	return _RESULTS


if __name__ == '__main__':
	# Get output file name/path
	# print("Please remove unnecessary pbarprints first")
	# exit(0)
	relative_path = "../../data/experiment_results/"
	outpath = outfile(relative_path, expcase = "discrete", exptype = "all")
	
	# Define the sources to be used
	names = ["S_3", "S_5"]
	params = {
		"S_3": dict(
			cword_length = 3,
			iword_length = 3,
		),
		"S_5": dict(
			cword_length = 2,
			iword_length = 2,
		),
		"all": dict(
			lengths = [int(10 ** (k/2)) for k in range(6, 14+1)],
			test_len = 50_000,
			qdist_length = 5
		)
	}
	
	repetitions = 10
	rep_idxs = list(range(repetitions)) * len(names)
	jobs = [x for y in [[name] * repetitions for name in names] for x in y]
	jnri = list(zip(jobs, rep_idxs))
	print(jobs, rep_idxs)
	random.shuffle(jnri)
	jobs, rep_idxs = zip(*jnri)
	print(jobs, rep_idxs)
	
	# Run the three sources in parallel
	all_results = []
	try:
		all_results.extend(joblib.Parallel(
			n_jobs = -2
		)(
			joblib.delayed(
				experiment_discrete
			)(
				name,
				repetition = repetition,
				**params[name],
				**params["all"]
			)
			for name, repetition in zip(jobs, rep_idxs)
		))
	except:
		all_results = _RESULTS
	finally:
		if all_results:
			res = pd.DataFrame.from_records(
				sum(all_results, [])
			)
			res.to_csv(outpath)