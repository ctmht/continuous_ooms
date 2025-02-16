import contextlib
import os
import time

from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib

from src.oom import *
from src.oom.util.learning_discrete import estimate_matrices_discrete_fixed


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
	"""
	Context manager to patch joblib to report into tqdm progress bar given as argument
	"""
	class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
		def __call__(self, *args, **kwargs):
			tqdm_object.update(n=self.batch_size)
			return super().__call__(*args, **kwargs)

	old_batch_callback = joblib.parallel.BatchCompletionCallBack
	joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
	try:
		yield tqdm_object
	finally:
		joblib.parallel.BatchCompletionCallBack = old_batch_callback
		tqdm_object.close()


def experiment_learning_discrete(
	params_generation,
	params_target,
	**params_source
):
	"""
	
	"""
	d0      = params_source["source_dimension"]
	density = params_source["source_density"]
	n0      = params_source["source_alphabet_size"]
	is_stat = params_source["source_stationary_state"]
	
	source = DiscreteValuedOOM.from_sparse(
		dimension                = d0,
		density                  = density,
		alphabet                 = None,
		alphabet_size            = n0,
		deterministic_functional = False,
		stationary_state         = is_stat
	)
	# genlog.info(f"Created DiscreteValuedOOM:\n{source}")
	
	_results = []
	
	for gen_pgrid in ParameterGrid(params_generation):
		gen_len = gen_pgrid["training_length"]
		
		test_len = max(50_000, min(gen_len // 4, 200_000))
		gen = source.generate(gen_len + test_len)
		
		r_target_dimension = list(range(2, d0 + 1, 2))
		
		params_target["ci_words_length"]  = [
			i for i in range(2, 8, 2)
			if max(r_target_dimension) <= n0 ** (2 * i) <= 1e+8
		]
		
		for target_pgrid in ParameterGrid(params_target):
			len_ciw = target_pgrid["ci_words_length"]
			
			estimated_matrices = estimate_matrices_discrete_fixed(
				sequence   = gen.sequence,
				len_cwords = len_ciw,
				len_iwords = len_ciw
			)
			
			for d1 in r_target_dimension:
				learned = DiscreteValuedOOM.from_data(
					gen.sequence[:-test_len],
					target_dimension   = d1,
					len_cwords         = len_ciw,
					len_iwords         = len_ciw,
					estimated_matrices = estimated_matrices
				)
				# genlog.info(f"Learned DiscreteValuedOOM:\n{learned}")
				
				with np.errstate(divide = "raise"):
					comp_source = source.compute(gen.sequence[-test_len:])
					nll_source  = comp_source.nll_list[-1]
					
					try:
						comp_learned = learned.compute(gen.sequence[-test_len:])
						nll_learned = comp_learned.nll_list[-1]
					except FloatingPointError:
						# print(learned)
						# print(*[p.T for p in learned.tv.p_vec_list[-3:]], sep = '\n')
						# print(*learned.tv.sequence[
						# 	   		learned.tv.time_step - 3
						# 			:
						# 			learned.tv.time_step
						# 	   ])
						# print(dict(params_source, **params_generation, **target_pgrid))
						# print('\n\n\n\n', flush = True)
						nll_learned = "nan"
				
				target_pgrid["target_dimension"] = d1
				
				here_results = dict(
					dict(params_source,
						 **gen_pgrid,
						 **target_pgrid),
					**dict(nll_source = nll_source,
						   nll_learned = nll_learned)
				)
				_results.append(here_results)
				print(here_results, flush = True)
	
	return _results


param_ranges = {
	"source": {
		"source_dimension"        : [2, 4, 8, 16],
		"source_density"          : [0.1, 0.5, 1.0],
		"source_alphabet_size"    : [2, 4, 6, 8],
		"source_stationary_state" : [True],
	},
	"generation": {
		"training_length"         : [100_000],
	},
	"target": {
		# "ci_words_length"		  : [3, 4, 5],
		# "target_dimension"        : [2, 4, 8, 16],
	}
}


if __name__ == '__main__':
	outpath = os.path.abspath("../../data/experiment_results/")
	
	idxmax = 0
	for file in os.listdir(outpath):
		idx = int(file.split('.')[0].split('_')[-1])
		if idx > idxmax:
			idxmax = idx
	outidx = idxmax + 1
	
	outpath = os.path.join(
		outpath,
		f"experiment_learning_discrete_results_{outidx}.csv"
	)
	
	print(f"Experiment results will be saved at '{outpath}'", flush = True)
	time.sleep(1)
	
	##################################
	
	pgrid = ParameterGrid(param_ranges["source"])
	
	# Run experiments in parallel
	try:
		with tqdm_joblib(
			tqdm(
				desc = "Experimental progress",
				total = len(pgrid),
				position = 0,
				leave = True
			)
		) as progress_bar:
			all_results = joblib.Parallel(
				n_jobs = -2
			)(
				joblib.delayed(
					experiment_learning_discrete
				)(
					param_ranges["generation"],
					param_ranges["target"],
					**source_pgrid
				)
				for source_pgrid in pgrid
			)
	finally:
		res = pd.DataFrame.from_records(
			sum(all_results, [])
		)
		res.to_csv(outpath)