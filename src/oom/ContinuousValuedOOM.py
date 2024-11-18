from collections.abc import Sequence, Callable
from typing import Optional, Union, overload

import numpy as np
from scipy.stats import rv_continuous

from .observable import Observable, ObsSequence
from .operator import Operator
from .OOM import LinFunctional, ObservableOperatorModel, State


class ContinuousValuedOOM(ObservableOperatorModel):
	
	
	def __init__(
		self,
		dim: int,
		linear_functional: LinFunctional,
		operators: Sequence[Operator],
		start_state: State,
		membership_functions: list[rv_continuous],
		invalidity_adjustment: Optional[dict[str, Union[int, float]]] = None
	):
		"""
		"""
		super().__init__(
			dim,
			linear_functional,
			operators,
			start_state,
			invalidity_adjustment
		)
		self.membership_fns: list[rv_continuous] = membership_functions
	
	
	def generate(
		self,
		length: int
	):
		stop = length
		mode = self._TraversalMode.GENERATE
		self.tv = self.get_traversal_state(stop, mode)
		self.tv.sequence = []
		
		return self._sequence_traversal()
	
	
	def compute(
		self,
		sequence: Sequence,
		length_max: Optional[int] = None
	):
		stop = min(len(sequence), length_max) if length_max else len(sequence)
		mode = self._TraversalMode.COMPUTE
		self.tv = self.get_traversal_state(stop, mode)
		self.tv.sequence_cont = sequence
		self.tv.p_vecs_cont = []
		
		return self._sequence_traversal()
	
	
	def step_get_observation(
		self
	):
		match self.tv.mode:
			case self._TraversalMode.GENERATE:
				# Choose next observation randomly, then its operator
				memb_fun: rv_continuous = np.random.choice(
					self.membership_fns,
					p = self.tv.p_vecs[-1]
				)
				obs = memb_fun.rvs()
				self.tv.sequence.append(obs)
			case self._TraversalMode.COMPUTE:
				obs = self.tv.sequence[self.tv.time_step - 1]
			case _:
				raise NotImplementedError("Can only compute or generate.")
		
		return obs
	
	
	def step_get_operator(
		self,
		obs
	) -> Operator:
		mat = np.asmatrix(np.zeros(shape = (self.dim, self.dim)))
		for mf, op in zip(self.membership_fns, self.operators):
			weight = mf.pdf(obs)
			mat = mat + weight * op.mat
		
		op = Operator(
			observable = Observable("-1"),
			range_dimension = self.dim,
			matrix_rep = mat
		)
		
		return op
	
	
	def step_get_nll(
		self,
		obs
	) -> float:
		p = 0
		for mf in self.membership_fns:
			p += mf.pdf(obs)
		nll = np.log2(p)
		
		return nll