from typing import Optional, Union, overload, override

import numpy as np
from scipy.stats import rv_continuous

from .OOM import ObservableOperatorModel
from .discrete_observable import DiscreteObservable
from .traversal import TraversalMode, TraversalState, TraversalType


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
	
	
	@override
	@property
	def type(
		self
	) -> TraversalType:
		return TraversalType.CONTINUOUS
	
	
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
	
	
	# def generate(
	# 	self,
	# 	length: int
	# ):
	# 	stop = length
	# 	mode = TraversalMode.GENERATE
	# 	self.tv = self.get_traversal_state(stop, mode)
	# 	self.tv.sequence = []
	# 	self.tv.sequence_cont = []
	# 	self.tv.p_vecs_cont = []
	#
	# 	return self._sequence_traversal()
	
	@override
	def generate(
		self,
		length: int,
		tvtype: Optional[TraversalType] = None
	) -> TraversalState:
		"""
		
		"""
		traversal_obj = self.get_traversal_state(
			tvmode = TraversalMode.GENERATE,
			tvtype = self.type,
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
		length: Optional[int] = None,
		tvtype: Optional[TraversalType] = None
	) -> TraversalState:
		"""
		
		"""
		traversal_obj = self.get_traversal_state(
			tvmode = TraversalMode.COMPUTE,
			tvtype = tvtype if tvtype else self.type,
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
	
	
	def step_get_nll(
		self,
		obs
	) -> float:
		"""
		
		"""
		weights = np.array([mf.pdf(obs) for mf in self.membership_fns])
		p = np.sum(weights)
		nll = np.log2(p)
		
		return nll
	
	
	#################################################################################
	##### LEARNING
	#################################################################################
	@override
	def from_data(
		*args,
		**kwargs
	) -> 'ContinuousValuedOOM':
		pass