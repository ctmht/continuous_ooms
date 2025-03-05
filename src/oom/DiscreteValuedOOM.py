from typing import Optional, Union, override
from warnings import simplefilter

import numpy as np
import pandas as pd

from .OOM import ObservableOperatorModel
from .discrete_observable import DiscreteObservable

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class DiscreteValuedOOM(ObservableOperatorModel):
	""" n/a """
	
	@override
	def __init__(
		self,
		dim: int,
		linear_functional,
		obs_ops: dict[DiscreteObservable | str | int, np.matrix],
		start_state: np.matrix,
		ia: Optional[dict[str, Union[int, float]]] = None,
	):
		"""
		Initialize an OOM for a discrete-valued process
		"""
		super().__init__(
			dim,
			linear_functional,
			obs_ops,
			start_state,
			ia
		)
	
	
	def __repr__(
		self
	) -> str:
		"""
		
		"""
		return f"<DiscreteValuedOOM object with dimension {self.dim} "\
			   f"and alphabet size {len(self.observables)}>"
	
	
	def __str__(
		self
	) -> str:
		strrep = self.__repr__() + '\n'
		strrep += f"functional = {self.lin_func.flatten()}\n"
		strrep += f"start state = {self.start_state.flatten()}^T\n"
		strrep += f"alphabet = [" + ', '.join([o.uid for o in self.observables])
		strrep += f"]\n"
		
		for obs, op in zip(self.observables, self.operators):
			strrep += f"    {obs} operator matrix:\n{op}\n"
		
		return strrep
	
	
	@staticmethod
	def from_sparse(
		dimension: int,
		density: float,
		alphabet: Optional[list[DiscreteObservable]] = None,
		alphabet_size: Optional[int] = None,
		seed: Optional[int] = None
	) -> 'ObservableOperatorModel':
		"""
		
		"""
		from .util import random_discrete_valued_hmm
		
		return random_discrete_valued_hmm(
			dimension,
			density,
			alphabet,
			alphabet_size,
			seed
		)
	
	
	@override
	@staticmethod
	def from_data(
		obs: list[DiscreteObservable | str | int],
		target_dimension: int,
		len_cwords: Optional[int] = None,
		len_iwords: Optional[int] = None,
		max_length: Optional[int] = None,
		estimated_matrices: Optional[tuple[np.matrix]] = None
	) -> 'DiscreteValuedOOM':
		"""
		
		"""
		from .util import learn_discrete_valued_oom
		
		return learn_discrete_valued_oom(
			obs,
			target_dimension,
			len_cwords,
			len_iwords,
			max_length,
			estimated_matrices
		)