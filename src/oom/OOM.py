from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from .traversal import TraversalMode, TraversalState
from .discrete_observable import DiscreteObservable


class ObservableOperatorModel(ABC):
	"""
	Abstract base class for Observable Operator Model classes
	"""
	
	default_adjustment: dict[str, Union[int, float]] = {
		"margin": 0.005,
		"setbackMargin": 0.03,
		"setbackLength": 6
	}
	
	
	@abstractmethod
	def __init__(
		self,
		dim: int,
		linear_functional,
		obs_ops: dict[DiscreteObservable | str | int, np.matrix],
		start_state: np.matrix,
		ia: Optional[dict[str, Union[int, float]]] = None,
		*args,
		**kwargs
	):
		"""
		Assumes observables and operators are ordered identically
		"""
		self._invalidity_adj = self._set_invalidity_adjustment(ia)
		
		# Dimension of state space and operators
		self.dim: int = dim
		
		# Linear functional
		self.lin_func = linear_functional
		
		# Observables and operators
		self.observables = [
			obs if isinstance(obs, DiscreteObservable) else DiscreteObservable(obs)
			for obs in obs_ops.keys()
		]
		self.operators = list(obs_ops.values())
		
		# Set starting state of the OOM
		self.start_state = start_state
		
		self.lf_on_operators: np.matrix = np.asmatrix(
			np.vstack([self.lin_func * op for op in self.operators])
		)
		
		# Check model is correct
		self._validate()
	
	
	def _set_invalidity_adjustment(
		self,
		invalidity_adjustment: Optional[dict[str, Union[int, float]]]
	) -> dict[str, Union[int, float]]:
		"""
		Set the dictionary of parameters corresponding to the adjustment
		required to run OOMs as invalid observations generators
		"""
		if invalidity_adjustment is None:
			invalidity_adjustment = {}
		inv = {
			"margin": invalidity_adjustment.get(
				"margin", self.default_adjustment["margin"]
			),
			"setbackMargin": invalidity_adjustment.get(
				"setbackMargin", self.default_adjustment["setbackMargin"]
			),
			"setbackLength": invalidity_adjustment.get(
				"setbackLength", self.default_adjustment["setbackLength"]
			)
		}
		return inv
	
	
	def _validate(
		self
	) -> None:
		# TODO: verify components are of correct dims
		pass
	
	
	def minimize(
		self
	):
		# TODO
		pass
	
	
	def normalize(
		self,
		ones_row: bool = False
	) -> None:
		"""
		
		"""
		# TODO: division by zero errors when using large sparse operators but i am making only minimal OOMs
		
		if ones_row:
			# Linearly transform to equivalent OOM with lin_func = [ 1  1 ... 1 ]
			rho = np.asmatrix(np.diag(np.asarray(self.lin_func)[0]))
			
			self.lin_func = self.lin_func * np.linalg.inv(rho)
			self.start_state = rho * self.start_state
			for op in self.operators:
				op = rho * op * np.linalg.inv(rho)
		else:
			# Normalize linear functional
			err = self.lin_func * self.start_state
			self.lin_func /= err
		
			# Normalize operators
			mu: np.matrix = np.sum([op for op in self.operators], axis = 0)
			lin_func_err = self.lin_func * mu
			
			for op in self.operators:
				for col in range(op.shape[1]):
					op[:, col] *= (self.lin_func[0, col] / lin_func_err[0, col])
	
	
	
	#################################################################################
	##### Traversal
	#################################################################################
	
	def get_traversal_state(
		self,
		tvmode: TraversalMode,
		stop: int,
		reduced: bool = True
	):
		traversal_state = TraversalState(
			mode = tvmode,
			time_step = 0,
			time_stop = stop,
			state_list = [self.start_state],
			nll_list = [],
			p_vec_list = [],
			sequence = None,
			sequence_cont = None,
			reduced = reduced
		)
		return traversal_state
	
	
	def generate(
		self,
		length: int,
		reduced: bool = True
	) -> TraversalState:
		"""
		
		"""
		traversal_obj = self.get_traversal_state(
			tvmode = TraversalMode.GENERATE,
			stop = length,
			reduced = reduced
		)
		traversal_obj.sequence = []
		
		traversal_obj = self._sequence_traversal(traversal_obj)
		
		return traversal_obj
	
	
	def compute(
		self,
		sequence: list[DiscreteObservable],
		length: Optional[int] = None,
		reduced: bool = True
	) -> TraversalState:
		"""
		
		"""
		traversal_obj = self.get_traversal_state(
			tvmode = TraversalMode.COMPUTE,
			stop = min(len(sequence), length) if length else len(sequence),
			reduced = reduced
		)
		traversal_obj.sequence = sequence
		
		traversal_obj = self._sequence_traversal(traversal_obj)
		
		return traversal_obj
	
	
	def _sequence_traversal(
		self,
		traversal_obj: TraversalState
	):
		"""
		
		"""
		# Attribute 'tv' created just during traversal to make life easier
		self.tv: TraversalState = traversal_obj
		
		nll: float = 0
		
		while self.tv.time_step < self.tv.time_stop:
			# Get last visited state as the current state
			state = self.tv.state_list[-1]
			
			# Get the probability vector representing the distribution at this state
			p_vec, adjustflag = self.step_get_distribution(state)
			
			# If no setbacks, can now continue to increase time step and add elements
			self.tv.time_step += 1
			self.tv.p_vec_list.append(p_vec)
			
			# Getting observation and its operator
			obs = self.step_get_observation()
			op = self.step_get_operator(obs)
			
			# Apply operator to get next state
			state = op * state
			state = state / (self.lin_func * state)
			self.tv.state_list.append(state)
			
			# Get NLL using current observation
			nll_step = -self.step_get_ll(obs)
			nll = nll + (nll_step - nll) / self.tv.time_step
			self.tv.nll_list.append(nll)
			
			if self.tv.reduced and self.tv.time_step > self._invalidity_adj["setbackLength"]:
				del self.tv.state_list[0]
				del self.tv.nll_list[0]
				del self.tv.p_vec_list[0]
			
			# print(obs, state.T)
			
		# Return traversal object and remove extra class instance
		tv_result = self.tv
		del self.tv
		return tv_result
	
	
	def step_get_distribution(
		self,
		state: np.matrix
	) -> tuple[np.array, bool]:
		"""
		Acquire probability vector as the distribution over each symbol in the
		(discrete) OOM, and perform setbacks to earlier steps if the traversal is
		becoming unstable
		
		Args:
			state: the current state of the traversal, uniquely determining the
				distribution over the symbols
		"""
		# Setback parameters
		ia_margin = self._invalidity_adj["margin"]
		ia_sbmargin = self._invalidity_adj["setbackMargin"]
		ia_sblength = self._invalidity_adj["setbackLength"]
		
		# Get probability vector
		p_vec = self.lf_on_operators * state
		p_vec = np.array(p_vec).flatten()
		
		# Invalidity checks
		delta = np.sum(ia_margin - p_vec, where = p_vec <= 0)
		p_plus = np.sum(p_vec, where = p_vec > 0)
		nu_ratio = 1 - delta / p_plus
		
		p_vec[p_vec > 0] *= nu_ratio
		p_vec[p_vec <= 0] = ia_margin
		
		# Setback if unstable
		adjustflag = False
		if delta > ia_sbmargin:
			adjustflag = True
			
			if not self.tv.time_step > ia_sblength + 1:
				newstate = self.start_state
			else:
				newstate = self.start_state
				for obs in self.tv.sequence[-ia_sblength:]:
					op = self.step_get_operator(obs)
					newstate = op * newstate
					newstate = newstate / (self.lin_func * newstate)
			
			self.tv.state_list[-1] = newstate
		
		return p_vec, adjustflag
	
	
	def step_get_observation(
		self
	):
		"""
		Acquire observation at the current time step, either by sampling the symbols
		using the determined probability vector (GENERATE tvmode) or by selecting it
		from the available sequence (COMPUTE tvmode)
		"""
		match self.tv.mode:
			case TraversalMode.COMPUTE:
				# Choose operator knowing the next observation
				obs: DiscreteObservable = self.tv.sequence[self.tv.time_step - 1]
			case TraversalMode.GENERATE:
				# Choose next observation randomly
				try:
					obs: DiscreteObservable = np.random.choice(
						self.observables,
						p = self.tv.p_vec_list[-1]
					)
					self.tv.sequence.append(obs)
				except ValueError:
					# d = sum(self.tv.p_vec_list[-1]) - 1
					# if d < 1e-5:
					# 	print(d, end=' ')
					self.tv.p_vec_list[-1] /= sum(self.tv.p_vec_list[-1])
					return self.step_get_observation()
			case _:
				raise NotImplementedError("Can only compute or generate.")
		
		return obs
	
	
	def step_get_operator(
		self,
		obs
	) -> np.matrix:
		"""
		Acquire the operator corresponding to the acquired discrete_observable
		"""
		op = self.operators[self.observables.index(obs)]
		return op
	
	
	def step_get_ll(
		self,
		obs
	) -> float:
		idxoi = self.observables.index(obs)
		p = self.tv.p_vec_list[-1][idxoi]
		ll = np.log2(p)
		
		return ll
	
	
	#################################################################################
	##### Creation
	#################################################################################
	@staticmethod
	@abstractmethod
	def from_data(
		*args,
		**kwargs
	) -> 'ObservableOperatorModel':
		pass