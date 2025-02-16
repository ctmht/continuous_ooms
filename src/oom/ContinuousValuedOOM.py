from typing import Optional, Union, overload, override

import numpy as np
from scipy.stats import rv_continuous

from . import DiscreteValuedOOM
from .OOM import ObservableOperatorModel
from .discrete_observable import DiscreteObservable
from .traversal import TraversalMode, TraversalState


class ContinuousValuedOOM(ObservableOperatorModel):
	""" na """
	
	def __init__(
		self,
		dim: int,
		linear_functional,
		obs_ops: dict[DiscreteObservable | str | int, np.matrix],
		start_state: np.matrix,
		membership_functions: list[rv_continuous],
		ia: Optional[dict[str, Union[int, float]]] = None,
	):
		"""
		"""
		super().__init__(
			dim,
			linear_functional,
			obs_ops,
			start_state,
			ia
		)
		self.membership_fns: list[rv_continuous] = membership_functions
	
	
	def __repr__(
		self
	) -> str:
		"""
		
		"""
		return f"<ContinuousValuedOOM object with dimension {self.dim} "\
			   f"and alphabet size {len(self.observables)}>"
	
	
	def __str__(
		self
	) -> str:
		strrep = self.__repr__() + '\n'
		strrep += f"functional = {self.lin_func.flatten()}\n"
		strrep += f"start state = {self.start_state.flatten()}^T\n"
		strrep += f"alphabet = [" + ', '.join([o.uid for o in self.observables])
		strrep += f"]\n"
		
		for op in self.operators:
			strrep += f"    {op.observable.uid} operator matrix:\n{op.mat}\n"
		
		return strrep
	
	
	@override
	def generate(
		self,
		length: int
	) -> TraversalState:
		"""
		
		"""
		traversal_obj = self.get_traversal_state(
			tvmode = TraversalMode.GENERATE,
			stop = length
		)
		traversal_obj.sequence = []
		traversal_obj.sequence_cont = []
		
		traversal_obj = self._sequence_traversal(traversal_obj)
		
		return traversal_obj
	
	
	@override
	def compute(
		self,
		sequence: list,
		length: Optional[int] = None
	) -> TraversalState:
		"""
		
		"""
		traversal_obj = self.get_traversal_state(
			tvmode = TraversalMode.COMPUTE,
			stop = min(len(sequence), length) if length else len(sequence)
		)
		traversal_obj.sequence_cont = sequence
		
		return traversal_obj
	
	
	def step_get_observation(
		self
	):
		"""
		
		"""
		match self.tv.mode:
			case TraversalMode.COMPUTE:
				obs = self.tv.sequence_cont[self.tv.time_step - 1]
			case TraversalMode.GENERATE:
				# Choose next observation randomly, then its operator
				d_obs: DiscreteObservable = np.random.choice(
					self.observables,
					p = self.tv.p_vec_list[-1]
				)
				self.tv.sequence.append(d_obs)
				
				obs = self.membership_fns[self.observables.index(d_obs)].rvs()
				self.tv.sequence_cont.append(obs)
			case _:
				raise NotImplementedError("Can only compute or generate.")
		
		return obs
	
	
	def step_get_operator(
		self,
		obs
	) -> np.matrix:
		"""
		
		"""
		weights = np.array([mf.pdf(obs) for mf in self.membership_fns])
		
		wmat = np.average(
			self.operators, weights = weights, axis = 0
		) * np.sum(weights)
		
		return np.asmatrix(wmat)
	
	
	def step_get_ll(
		self,
		obs
	) -> float:
		"""
		
		"""
		weights = np.array([mf.pdf(obs) for mf in self.membership_fns])
		p = np.sum(weights)
		ll = np.log2(p)
		
		return ll
	
	
	#################################################################################
	##### LEARNING
	#################################################################################
	@override
	@staticmethod
	def from_data(
		obs: list[DiscreteObservable | str | int],
		target_dimension: int,
		len_cwords: int,
		len_iwords: int,
		membership_functions: Optional[list[rv_continuous]],
		estimated_matrices: Optional[tuple[np.matrix]] = None,
	) -> 'ContinuousValuedOOM':
		"""
		
		"""
		from .util import learn_continuous_valued_oom
		
		if membership_functions is None:
			raise NotImplementedError("Clustering required")
		
		return learn_continuous_valued_oom(
			obs,
			target_dimension,
			len_cwords,
			len_iwords,
			membership_functions,
			estimated_matrices,
		)
	
	
	@staticmethod
	def from_discrete_valued_oom(
		dvoom: DiscreteValuedOOM,
		membership_functions: list[rv_continuous]
	) -> 'ContinuousValuedOOM':
		"""
		
		"""
		return ContinuousValuedOOM(
			dim                  = dvoom.dim,
			linear_functional    = dvoom.lin_func,
			obs_ops              = dict(zip(dvoom.observables, dvoom.operators)),
			start_state          = dvoom.start_state,
			membership_functions = membership_functions,
			ia                   = dvoom._invalidity_adj
		)