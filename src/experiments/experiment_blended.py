import sys
import random

import numpy as np
import pandas as pd

from src.oom.util import *
from src.oom.util.few_step_prediction import fix_pvec, get_discretized_operators, \
	kl_divergence, mse, quantify_distribution
from src.oom.util.learning_continuous import estimate_matrices_continuous, \
	get_transfer_matrix
from src.experiments.util_experiments import *


_RESULTS = []													# Final results


def experiment_continuous(
	name: Literal["S_3", "S_5"],
	len_ci_words: dict[int, int],
	lengths: list[int],
	test_len: int = 50_000,
	qdist_length: int = 3,
	repetition: int = 0,
	qdintervals: list[tuple[float, float]] = None
):
	"""
	
	"""
	if qdintervals is None:
		raise ValueError("Intervals required for discretizing output space in the "
						 "few-step prediction assessment.")
	
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
	case = "blended"
	source_oom: ContinuousValuedOOM = make_source(name, case="blended")
	d0 = source_oom.dim
	n0 = len(source_oom.observables)
	
	_SOURCE_PARS = dict(case=case, name=name, d0=d0, n0=n0)			# Dict of stats
	
	# Perform repetition
	pbarprint(f"Computing transfer function for known memberships")
	T_inv = get_transfer_matrix(
		dict(zip(source_oom.observables, source_oom.membership_fns))
	)
	pbarprint(f"Computing transfer function for known memberships: Finished")
	
	# Perform repetition
	pbarprint(f"Repetition {repetition} started")
	
	# Generate sequence of the maximum length (all steps will be on subsequences)
	pbarprint(f"Generating {max(lengths) + test_len} observations")
	gen = source_oom.generate(max(lengths) + test_len)
	pbarprint(f"Generating {max(lengths) + test_len} observations: Finished")
	
	# Estimate entropy rate of source OOM on test data
	pbarprint(f"Computing source OOM entropy rate")
	comp_source = source_oom.compute(gen.sequence_cont[-test_len:])
	nll_source  = comp_source.nll_list[-1]
	pbarprint(f"Computing source OOM entropy rate: Finished, NLL_source = {nll_source}")
	
	# Compute distribution of few-step prediction for source OOM
	pbarprint(f"Computing source OOM few-step prediction distribution at {qdist_length}-steps")
	disc_ops_source = get_discretized_operators(source_oom, qdintervals)
	fsp_vec_source = quantify_distribution(
		steps = qdist_length,
		state = source_oom.start_state,
		operators = list(disc_ops_source.values()),
		lin_func = source_oom.lin_func
	)
	fsp_vec_source = fix_pvec(fsp_vec_source)
	pbarprint(f"Computing source OOM few-step prediction distribution at {qdist_length}-steps: Finished")
	
	_SOURCE_METRICS = dict(repetition=repetition, nll_source=nll_source,
						   steps_fsp=qdist_length, fsp_source=fsp_vec_source)
	
	for length in lengths:
		##############################################
		# UNBLENDING WITH KNOWN MEMBERSHIP FUNCTIONS #
		##############################################
		
		# Estimate large matrices with known membership functions
		cword_length = len_ci_words[n0]
		iword_length = len_ci_words[n0]
		pbarprint(f"Estimating matrices (BK) using L_c = {cword_length}, L_i = {iword_length} at length {length}")
		estimated_matrices = estimate_matrices_continuous(
			sequence             = gen.sequence_cont[:length],
			len_cwords           = cword_length,
			len_iwords           = iword_length,
			membership_functions = source_oom.membership_fns,
			observables          = source_oom.observables,
			ret_numrank          = True,
			T_inv                = T_inv
		)
		rankmax = np.linalg.matrix_rank(estimated_matrices[0][0])
		ranknum = estimated_matrices[-1]
		estimated_matrices = estimated_matrices[:-1]
		pbarprint(f"Estimating matrices (BK) using L_c = {cword_length}, L_i = {iword_length} at length {length}: "
				  f"Finished, rank(F_IJ) = {rankmax}, numrank(F_IJ) = {ranknum}")
		
		# d1_range = range(2, rankmax + 1)
		# d1_range = [x for x in range(ranknum-3, ranknum+3) if 2 <= x <= rankmax]
		d1_range = [x for x in range(2, 15) if 2 <= x <= rankmax]
		for d1 in d1_range:
			# Learn OOM using these parameters
			pbarprint(f"Learning")
			learned = ContinuousValuedOOM.from_data(
				gen.sequence_cont[:length],
				target_dimension     = d1,
				len_cwords           = cword_length,
				len_iwords           = iword_length,
				membership_functions = source_oom.membership_fns,
				observables          = source_oom.observables,
				estimated_matrices   = estimated_matrices
			)
			assert learned is not None, "OOM not learned"
			pbarprint(f"Learning: Finished")
			
			_LEARNED_PARS = dict(seqlength=length, cwlen=cword_length,
								 iwlen=iword_length, rankmax=rankmax,
								 ranknum=ranknum, d1=d1, n1=n0)
			
			# Estimate entropy rate of this learned OOM on test data
			pbarprint(f"Computing learned OOM entropy rate")
			with np.errstate(divide='ignore'):
				comp_learned = learned.compute(gen.sequence_cont[-test_len:])
			nll_learned = comp_learned.nll_list[-1]
			pbarprint(f"Computing learned OOM entropy rate: Finished, NLL_learned = {nll_learned}")
			
			# Compute distribution of few-step prediction for learned OOM
			pbarprint(f"Computing learned OOM few-step prediction distribution at {qdist_length}-steps")
			disc_ops_learned = get_discretized_operators(learned, qdintervals)
			fsp_vec_learned = quantify_distribution(
				steps = qdist_length,
				state = learned.start_state,
				operators = list(disc_ops_learned.values()),
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
			_SOURCE_PARS["case"] = _SOURCE_PARS["case"] + "_known"
			_HERE_RESULTS = dict(**_SOURCE_PARS, **_SOURCE_METRICS,
								 **_LEARNED_PARS, **_LEARNED_METRICS)
			_RESULTS.append(_HERE_RESULTS)
			
			# Print results for this learned OOM
			pbarprint.update(1)
			with np.printoptions(legacy='1.21'):
				pbarprint(f"Results {without(_HERE_RESULTS, 'fsp_source', 'fsp_learned')}")
			
			_SOURCE_PARS["case"] = _SOURCE_PARS["case"].split('_')[0]
		
		################################################
		# UNBLENDING WITH UNKNOWN MEMBERSHIP FUNCTIONS #
		################################################

		# Determine weights of source MFs over the current-length train sequence
		ser = pd.Series([obs.uid for obs in gen.sequence[:length]])
		ser = ser.value_counts(normalize = True)
		ser = ser.sort_index(key = lambda x: x.str.ljust(5, 'z'))
		weights_data = ser.values

		# Determine Gaussian Mixture Model which best fits our training data
		pbarprint(f"Estimating memberships using GMM")
		data = np.array(gen.sequence_cont[:length]).reshape(-1, 1)
		mfs_from_gmm, _GMM_BEST_METRICS = get_from_gmm(
			data               = data,
			n_components_range = list(range(2, 6 + 1)),
			ret_metrics        = True,
			true_pdfs          = source_oom.membership_fns,
			true_seqweights    = weights_data
		)
		n1 = _GMM_BEST_METRICS.pop("alphabet_size")
		pbarprint(f"Estimating memberships using GMM: Finished, best has n1 = {n1}")

		# Estimate large matrices_bl
		cword_length = len_ci_words[n1]
		iword_length = len_ci_words[n1]
		pbarprint(f"Estimating matrices (BU) using L_c = {cword_length}, L_i = {iword_length} at length {length}")
		estimated_matrices = estimate_matrices_continuous(
			sequence             = gen.sequence_cont[:length],
			len_cwords           = cword_length,
			len_iwords           = iword_length,
			membership_functions = mfs_from_gmm,
			observables          = None,
			ret_numrank          = True,
			T_inv                = None
		)
		try:
			rankmax = np.linalg.matrix_rank(estimated_matrices[0][0])
		except np.linalg.LinAlgError:
			rankmax = 0
		ranknum = estimated_matrices[-1]
		estimated_matrices = estimated_matrices[:-1]
		pbarprint(f"Estimating matrices (BU) using L_c = {cword_length}, L_i = {iword_length} at length {length}: "
				  f"Finished, rank(F_IJ) = {rankmax}, numrank(F_IJ) = {ranknum}")

		# d1_range = range(2, rankmax + 1)
		# d1_range = [x for x in range(ranknum-3, ranknum+3) if 2 <= x <= rankmax]
		d1_range = [x for x in range(2, 15) if 2 <= x <= rankmax]
		for d1 in d1_range:
			# Learn OOM using these parameters
			pbarprint(f"Learning")
			learned = ContinuousValuedOOM.from_data(
				gen.sequence_cont[:length],
				target_dimension     = d1,
				len_cwords           = cword_length,
				len_iwords           = iword_length,
				membership_functions = mfs_from_gmm,
				observables          = None,
				estimated_matrices   = estimated_matrices
			)
			assert learned is not None, "OOM not learned"
			pbarprint(f"Learning: Finished")

			_LEARNED_PARS = dict(seqlength=length, cwlen=cword_length,
								 iwlen=iword_length, rankmax=rankmax,
								 ranknum=ranknum, d1=d1, n1=n1)

			# Estimate entropy rate of this learned OOM on test data
			pbarprint(f"Computing learned OOM entropy rate")
			with np.errstate(divide='ignore'):
				comp_learned = learned.compute(gen.sequence_cont[-test_len:])
			nll_learned = comp_learned.nll_list[-1]
			pbarprint(f"Computing learned OOM entropy rate: Finished, NLL_learned = {nll_learned}")

			# Compute distribution of few-step prediction for learned OOM
			pbarprint(f"Computing learned OOM few-step prediction distribution at {qdist_length}-steps")
			disc_ops_learned = get_discretized_operators(learned, qdintervals)
			fsp_vec_learned = quantify_distribution(
				steps = qdist_length,
				state = learned.start_state,
				operators = list(disc_ops_learned.values()),
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
			_SOURCE_PARS["case"] = _SOURCE_PARS["case"] + "_gmm"
			_HERE_RESULTS = dict(**_SOURCE_PARS, **_SOURCE_METRICS,
								 **_LEARNED_PARS, **_LEARNED_METRICS,
								 **_GMM_BEST_METRICS)
			_RESULTS.append(_HERE_RESULTS)

			# Print results for this learned OOM
			pbarprint.update(1)
			with np.printoptions(legacy='1.21'):
				pbarprint(f"Results {without(_HERE_RESULTS, 'fsp_source', 'fsp_learned')}")

			_SOURCE_PARS["case"] = _SOURCE_PARS["case"].split('_')[0]
			
	return _RESULTS


if __name__ == '__main__':
	# Get output file name/path
	relative_path = "../../data/experiment_results/"
	outpath = outfile(relative_path, expcase = "blended", exptype = "allFIN")
	
	# Define the sources to be used
	names = ["S_3", "S_5"]
	params = dict(
		lengths = [int(10 ** (k/2)) for k in range(6, 12+1)],
		test_len = 50_000,
		qdist_length = 5,
		len_ci_words = {
			2: 4,
			3: 3,
			4: 2,
			5: 2,
			6: 2
		}
	)
	
	qrinf = 4
	qdintervals = [(-np.inf, -qrinf), *[(a, a+1) for a in range(-qrinf, qrinf)], (qrinf, np.inf)]
	
	repetitions = 6
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
			n_jobs = 6
		)(
			joblib.delayed(
				experiment_continuous
			)(
				name,
				repetition  = repetition,
				qdintervals = qdintervals,
				**params
			)
			for name, repetition in zip(jobs, rep_idxs)
		))
	except:
		all_results = _RESULTS
		raise
	finally:
		if all_results:
			res = pd.DataFrame.from_records(
				sum(all_results, [])
			)
			res.to_csv(outpath)