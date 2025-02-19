import contextlib
import os
import time

import joblib
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

np.set_printoptions(linewidth = 1000, precision = 5, suppress = True, legacy='1.25')

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


def as_membership_function_getter(func):
	def wrapper(**hyperparameters):
		def result(n: int):
			return func(n, **hyperparameters)
		# Dynamically set the name
		params_str = ", ".join(f"{k}={v}" for k, v in hyperparameters.items())
		result.name = f"{func.__name__}, {params_str}"
		
		return result
	return wrapper

@as_membership_function_getter
def get_uniform(n: int):
	_membership_fns = []
	for _idx in range(n):
		_pdf = sp.stats.uniform(loc = _idx / n, scale = 1 / (2 * n))
		_membership_fns.append(_pdf)
	return _membership_fns

@as_membership_function_getter
def get_beta(n: int, arg_sum: int = 128):
	_membership_fns = []
	for _idx in range(1, n + 1):
		_offset = arg_sum / 2 * (n + 1 - 2*_idx) / (n + 1)
		_athis = arg_sum / 2 - _offset
		_bthis = arg_sum / 2 + _offset
		_pdf = sp.stats.beta(a = _athis, b = _bthis, loc = 0, scale = 1)
		_membership_fns.append(_pdf)
	return _membership_fns


def experiment_learning_continuous(
	params_generation,
	params_target,
	**params_source
):
	"""
	
	"""
	# Get source parameters
	d0      = params_source["source_dimension"]
	density = params_source["source_density"]
	n0      = params_source["source_alphabet_size"]
	is_stat = params_source["source_stationary_state"]
	mfn_get = params_source["source_membership_functions_getter"]
	
	# Manipulate for nicer output (see decorator 'as_membership_function_getter')
	params_source["source_membership_functions"] = mfn_get.name
	del params_source["source_membership_functions_getter"]
	
	source_disc = DiscreteValuedOOM.from_sparse(
		dimension                = d0,
		density                  = density,
		alphabet                 = None,
		alphabet_size            = n0,
		deterministic_functional = False,
		stationary_state         = is_stat
	)
	
	membership_functions = mfn_get(n0)
	source = ContinuousValuedOOM.from_discrete_valued_oom(
		dvoom                = source_disc,
		membership_functions = membership_functions
	)
	# genlog.info(f"Created DiscreteValuedOOM:\n{source}")
	
	_results = []
	
	for gen_pgrid in ParameterGrid(params_generation):
		gen_len = gen_pgrid["training_length"]
		
		test_len = max(50_000, min(gen_len // 4, 200_000))
		gen = source.generate(gen_len + test_len)
		
		comp_source = source.compute(gen.sequence_cont[-test_len:])
		nll_source  = comp_source.nll_list[-1]
		
		
		r_target_dimension = list(range(2, d0 + 1, 2))
		
		params_target["ci_words_length"]  = [
			i for i in range(2, 12, 4)
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
				learned = ContinuousValuedOOM.from_data(
					gen.sequence_cont[:-test_len],
					target_dimension     = d1,
					len_cwords           = len_ciw,
					len_iwords           = len_ciw,
					estimated_matrices   = estimated_matrices,
					membership_functions = membership_functions,
					observables          = source.observables
				)
				
				with np.errstate(divide = "raise"):
					try:
						comp_learned = learned.compute(gen.sequence_cont[-test_len:])
						nll_learned = comp_learned.nll_list[-1]
					except FloatingPointError:
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
		"source_dimension"        : [2, 8, 16],
		"source_density"          : [0.1, 0.5, 1.0],
		"source_alphabet_size"    : [2, 6],
		"source_stationary_state" : [True],
		"source_membership_functions_getter" : [
			get_beta(arg_sum = 32),
			get_beta(arg_sum = 128)
		]
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
		speclist = file.split('.')[0].split('_')
		if speclist[2] == "continuous":
			try:
				idx = int(file.split('.')[0].split('_')[-1])
				if idx > idxmax:
					idxmax = idx
			except ValueError:
				continue
	outidx = idxmax + 1
	
	outpath = os.path.join(
		outpath,
		f"experiment_learning_continuous_results_{outidx}.csv"
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
					experiment_learning_continuous
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