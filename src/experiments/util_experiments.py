from typing import Callable, Literal, Optional
import contextlib
import datetime
import os

from tqdm import tqdm
import numpy as np
import scipy as sp

from src.oom import ContinuousValuedOOM, DiscreteValuedOOM, ObservableOperatorModel
import joblib


#####################################################################################
# SAVEFILE PATH DETERMINATION
#####################################################################################
def outfile(relative_path: str, expcase: str, exptype: str):
	"""
	experiment
	_
	[[discrete/blended-known/blended-clustering]]		# experiment case
	_
	[[dataset/best-approximator/clustering/all]]		# experiment type
	_
	results
	_
	[[index]]
	"""
	outpath = os.path.abspath(relative_path)
	
	idxmax = 0
	for file in os.listdir(outpath):
		speclist = file.split('.')[0].split('_')
		try:
			if speclist[1] == expcase and speclist[2] == exptype:
					idx = int(speclist[-1])
					if idx > idxmax:
						idxmax = idx
		except:
			continue
	outidx = idxmax + 1
	
	outpath = os.path.join(
		outpath,
		f"experiment_{expcase}_{exptype}_results_{outidx}.csv"
	)
	
	print(f"Experiment results will be saved at '{outpath}'", flush = True)
	return outpath


#####################################################################################
# (DISCRETE) SOURCE OOM CREATION
#####################################################################################
def make_source(
	name: Optional[Literal["S_3", "S_5"]] = None,
	case: Optional[Literal["discrete", "blended"]] = "discrete",
	dimension: Optional[int] = None,
	alphabet_size: Optional[int] = None,
	density: Optional[float] = None,
	seed: Optional[int] = None,
) -> ObservableOperatorModel:
	"""
	
	"""
	if name is not None:
		match name:
			case "S_3":
				alphabet_size, dimension, density = 3, 10, 0.4
				seed = 95
				if case == "blended":
					mfs = get_gaussian(seed = 85)(alphabet_size)
			case "S_5":
				alphabet_size, dimension, density = 5, 10, 0.4
				seed = 79
				if case == "blended":
					mfs = get_gaussian(seed = 130)(alphabet_size)
			case _:
				raise ValueError(
					f"Source name {name} not recognized. Please provide one of "
					f"'S_3' or 'S_5'."
				)
	elif case == "blended":
		raise ValueError("Only predefined blended OOMs available. Create a discrete "
						 "OOM and then its membership functions separately for "
						 "custom blending options.")
	
	dv_oom = DiscreteValuedOOM.from_sparse(
		dimension = dimension,
		alphabet_size = alphabet_size,
		density = density,
		seed = seed
	)
	
	if case == "discrete":
		return dv_oom
	elif case == "blended":
		return ContinuousValuedOOM.from_discrete_valued_oom(dv_oom, mfs)


#####################################################################################
# TQDM FOR JOBLIB
#####################################################################################
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
	"""
	Context manager to patch joblib to report into tqdm progress bar given as argument
	"""
	class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
		def __call__(self, *args, **kwargs):
			tqdm_object.update_pdfs(n=self.batch_size)
			return super().__call__(*args, **kwargs)

	old_batch_callback = joblib.parallel.BatchCompletionCallBack
	joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
	try:
		yield tqdm_object
	finally:
		joblib.parallel.BatchCompletionCallBack = old_batch_callback
		tqdm_object.close()


class PbarPrinter:
	def __init__(self, indicator: str, **kwargs):
		self.indicator = indicator
		self.pbar = tqdm(**kwargs)
	
	def __call__(self, *args, **kwargs):
		self.pbar.clear()	  		# Clear the current progress bar
		
		# Print message without interfering with pbar
		print(f"{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} "
			  f"| {self.indicator.upper()}: ", end='')
		print(*args, **kwargs)
		
		self.pbar.refresh() 	 	# Refresh the progress bar
	
	def update(self, *args, **kwargs):
		self.pbar.update(*args, **kwargs)


def without(d, *keys):
	new_d = d.copy()
	for key in keys:
		new_d.pop(key)
	return new_d


#####################################################################################
# MEMBERSHIP FUNCTION GETTERS
#####################################################################################
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
def get_uniform(n: int, seed: int = None):
	_rng = np.random.default_rng(seed = seed)
	_rvs = sp.stats.norm(loc = 0, scale = 2).rvs    # Create RNG
	
	_membership_fns = []
	for _idx in range(n):
		_athis = _rvs(random_state = _rng)
		_bthis = _rvs(random_state = _rng) ** 2 + 1
		_pdf = sp.stats.uniform(loc = _athis - _bthis/2, scale = _bthis)
		_membership_fns.append(_pdf)
	return _membership_fns

@as_membership_function_getter
def get_uniform_control(n: int, seed: int = None):
	_membership_fns = []
	for _idx in range(n):
		_pdf = sp.stats.uniform(loc = 2*_idx, scale = 1)
		_membership_fns.append(_pdf)
	return _membership_fns

@as_membership_function_getter
def get_gaussian(n: int, seed: int = None):
	_rng = np.random.default_rng(seed = seed)
	_rvs = sp.stats.uniform(loc = -3, scale = 6).rvs    # Create RNG
	
	_membership_fns = []
	for _idx in range(n):
		_mean = _rvs(random_state = _rng)
		_var =  np.abs(_rvs(random_state = _rng)) / 3 + 0.5
		_pdf = sp.stats.norm(loc = _mean, scale = _var)
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