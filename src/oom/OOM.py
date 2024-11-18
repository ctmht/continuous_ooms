import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from functools import cached_property
from typing import Optional, Union

import numpy as np

from .observable import ObsSequence, Observable
from .operator import Operator

State = np.matrix
LinFunctional = np.matrix

class ObservableOperatorModel(ABC):
	"""
	Abstract base class for Observable Operator Model classes
	"""
	class _TraversalMode(Enum):
		GENERATE = 0
		COMPUTE = 1
	
	class _TraversalState(dict):
		__getattr__ = dict.get
		__setattr__ = dict.__setitem__
		__delattr__ = dict.__delitem__
	
	default_adjustment: dict[str, Union[int, float]] = {
		"margin": 0.005,
		"setbackMargin": -0.3,
		"setbackLength": 6
	}
	
	
	@abstractmethod
	def __init__(
		self,
		dim: int,
		linear_functional: LinFunctional,
		operators: Sequence[Operator],
		start_state: State,
		invalidity_adjustment: Optional[dict[str, Union[int, float]]] = None,
		*args, **kwargs
	):
		"""
		Constructor for generic Observable Operator Model
		
		Args:
			dim: the dimension of the stochastic process to be modelled
				by this OOM
			linear_functional: the linear functional associated with the
				OOM, in the form of a 1 x dim matrix (row vector)
			operators: the observable operators corresponding 1:1 with the
				possible observations, given as a list of Operator objects
			start_state: the starting state of the OOM, in the form of
				a dim x 1 matrix (column vector)
			
		"""
		# Set state space dimension and observables
		self.dim: int = dim
		self.lin_func: LinFunctional = linear_functional
		self.operators: Sequence[Operator] = operators
		self.start_state: State = start_state
		
		self.observables: Sequence[Observable] = [op.observable for op in operators]
		self.obsuids: Sequence[str] = [obs.uid for obs in self.observables]
		
		self._invalidity_adj = self._set_invalidity_adjustment(invalidity_adjustment)
	
	
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
	
	
	def normalize(
		self
	) -> None:
		"""

		"""
		# Normalize start state
		err = self.lin_func * self.start_state
		self.start_state /= err
	
		# Normalize operators
		mu: np.matrix = np.sum([op.mat for op in self.operators], axis = 0)
		lin_func_err = self.lin_func * mu
		
		for op in self.operators:
			for col in range(op.mat.shape[1]):
				op.mat[:, col] = (op.mat[:, col]
								* self.lin_func[0, col]
								/ lin_func_err[0, col])
	
	
	def get_traversal_state(
		self,
		stop: int,
		mode: _TraversalMode
	):
		traversal_state = {
			"time_step": 0,
			"time_stop": stop,
			"state_list": [self.start_state],
			"nll_list": [],
			"p_vecs": [],
			"mode": mode,
			"sequence": None,
		}
		return self._TraversalState(traversal_state)
	
	
	def generate(
		self,
		length: int
	) -> _TraversalState:
		"""
		
		"""
		stop = length
		mode = self._TraversalMode.GENERATE
		self.tv = self.get_traversal_state(stop, mode)
		self.tv.sequence = ObsSequence([])
		
		return self._sequence_traversal()
	
	
	def compute(
		self,
		sequence: ObsSequence,
		length_max: Optional[int] = None
	) -> _TraversalState:
		"""
		
		"""
		stop = min(len(sequence), length_max) if length_max else len(sequence)
		mode = self._TraversalMode.COMPUTE
		self.tv = self.get_traversal_state(stop, mode)
		self.tv.sequence = sequence
		
		return self._sequence_traversal()
	
	
	def _sequence_traversal(
		self,
		# stop: int,
		# sequence: ObsSequence,
		# mode: _TraversalMode
	):
		"""
		stop = min(len(sequence), length_max) if length_max else len(sequence)
			for mode()
		stop = length
			for generate()
		sequence = []
			for generate()
		sequence = sequence
			for mode()
		mode = set as appropriate
		"""
		
		pn1counts = {}
		
		nll: float = 0
		
		while self.tv.time_step < self.tv.time_stop:
			state = self.tv.state_list[-1]
			p_vec, adjustflag = self.step_get_distribution(state)
			if adjustflag:
				continue
			
			self.tv.time_step += 1
			
			self.tv.p_vecs.append(p_vec)
			
			###### DEBUG
			if abs(np.sum(p_vec) - 1) > 1e-13:
				print(f"---------------IA NEEDED 2--------------{np.sum(p_vec)}")
			if np.sum(p_vec) - 1 != 0:
				if np.sum(p_vec) - 1 not in pn1counts:
					pn1counts[np.sum(p_vec) - 1] = 0
				pn1counts[np.sum(p_vec) - 1] += 1
			
			# Adding
			obs = self.step_get_observation()
			op = self.step_get_operator(obs)
			
			# Apply operator to get next state
			state = op(state)
			state = state / (self.lin_func * state)
			self.tv.state_list.append(state)
			
			nll_step = self.step_get_nll(obs)
			
			# Get NLL using current observation
			nll = nll - (nll + nll_step) / self.tv.time_step
			self.tv.nll_list.append(nll)
		
		return self.tv
	
	
	def step_get_distribution(
		self,
		state: State
	) -> tuple[np.array, bool]:
		ia_margin = self._invalidity_adj["margin"]
		ia_sbmargin = self._invalidity_adj["setbackMargin"]
		ia_sblength = self._invalidity_adj["setbackLength"]
		
		# Get probability vector
		p_vec = self.lf_on_operators * state
		p_vec = np.array(p_vec).flatten()
		
		if np.linalg.norm(p_vec) > np.sqrt(len(self.observables)):
			msg = f"probability explosion, observations distributed in a vector "\
				  f"with norm {np.linalg.norm(p_vec)}. Is your model dimension "\
				  f"much larger than the process it models? (current OOM dimension "\
				  f"= {self.dim})"
			raise ValueError(msg)
		
		# Invalidity checks
		delta = np.sum(ia_margin - p_vec, where = p_vec <= 0)
		p_plus = np.sum(p_vec, where = p_vec > 0)
		nu_ratio = 1 - delta / p_plus
		p_out1 = abs(np.sum(p_vec) - 1)
		
		# Setback if unstable
		adjustflag = False
		if delta < ia_sbmargin or p_out1 > 1e-13:
			adjustflag = True
			
			# Reset by setbackLength and discard what comes after
			setback = min(self.tv.time_step, ia_sblength)
			self.tv.time_step -= setback
			
			for tlkey in self.tv:
				if isinstance(self.tv[tlkey], list):
					self.tv[tlkey] = self.tv[tlkey][:-setback]
			
			match self.tv.mode:
				case self._TraversalMode.GENERATE:
					self.tv.sequence = self.tv.sequence[:-setback]
			
			# DEBUG
			# print("    AFTER")
			# for key, value in self.tv.items():
			# 	print(
			# 		key,
			# 		len(value) if isinstance(value, list)
			# 				   or isinstance(value, ObsSequence)
			# 				   else value)
			# for p in self.tv.p_vecs[-5:]:
			# 	print(p.flatten())
			# print()
			# time.sleep(5)
		
		if not adjustflag:
			# Set negatives to "margin" and adjust valid probabilities
			p_vec[p_vec > 0] *= nu_ratio
			p_vec[p_vec <= 0] = ia_margin
		elif self.tv.mode == self._TraversalMode.COMPUTE:
			raise TimeoutError("Fail")
		
		return p_vec, adjustflag
	
	
	def step_get_observation(
		self
	):
		match self.tv.mode:
			case self._TraversalMode.GENERATE:
				# Choose next observation randomly
				obs: Observable = np.random.choice(
					self.observables,
					p = self.tv.p_vecs[-1]
				)
				self.tv.sequence.append(obs)
			case self._TraversalMode.COMPUTE:
				# Choose operator knowing the next observation
				obs: Observable = self.tv.sequence[self.tv.time_step - 1] # time goes 1 -> N
			case _:
				raise NotImplementedError("Can only compute or generate.")
		
		return obs
	
	
	def step_get_operator(
		self,
		obs
	) -> Operator:
		op = self.operators[self.obsuids.index(obs.uid)]
		return op
	
	
	def step_get_nll(
		self,
		obs
	) -> float:
		idxoi = self.observables.index(obs) # TODO not ok for cont
		p = self.tv.p_vecs[-1][idxoi]
		nll = np.log2(p)
		
		return nll
	
	#################################################################################
	##### Instance properties
	#################################################################################
	@cached_property
	def lf_on_operators(
		self
	) -> np.matrix:
		"""
		
		"""
		return np.asmatrix(
			np.vstack([self.lin_func * op.mat for op in self.operators])
		)