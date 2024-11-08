from functools import cached_property
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union
from enum import Enum

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
	
	default_adjustment: dict[str, Union[int, float]] = {
		"margin": 0.005,
		"setbackMargin": -0.3,
		"setbackLength": 2
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
		self.obsnames: Sequence[str] = [obs.uid for obs in self.observables]
		
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
		mu = sum([op.mat for op in self.operators])
		lin_func_err = self.lin_func * mu
		
		for op in self.operators:
			for col in range(op.mat.shape[1]):
				op.mat[:, col] = (op.mat[:, col]
								* self.lin_func[0, col]
								/ lin_func_err[0, col])
	
	
	# def generate(
	# 	self,
	# 	length: int = 100
	# ) -> tuple:
	# 	"""
	# 	Use the OOM to generate a observations of a given length from the
	# 	discrete-valued, stationary, ergodic process it models
	# 	"""
	# 	# Generated observations
	# 	state: State = self.start_state
	# 	statelist: list[State] = [state]
	# 	sequence: list[Observable] = []
	# 	nlls: list[np.array] = []
	# 	nll: float = 0
	#
	# 	for time_step in range(length):
	# 		p_vec = self.lf_on_operators * state
	#
	# 		# Invalidity adjustment
	# 		p_vec = np.array(p_vec).flatten()
	# 		delta = np.sum(
	# 			self._invalidity_adj["margin"] - p_vec, where = p_vec < 0
	# 		)
	# 		p_plus = np.sum(
	# 			p_vec, where = p_vec > 0
	# 		)
	# 		nu_ratio = 1 - delta / p_plus
	# 		# print(time_step, p_vec, delta)
	# 		if delta < self._invalidity_adj["setbackMargin"]:
	# 			# Reset by setbackLength and discard what comes after
	# 			time_step -= self._invalidity_adj["setbackLength"]
	# 			state = statelist[-self._invalidity_adj["setbackLength"]]
	# 			statelist = statelist[:-self._invalidity_adj["setbackLength"]]
	# 			sequence = sequence[:-self._invalidity_adj["setbackLength"]]
	#
	# 		# Set negatives to "margin" and adjust valid probabilities
	# 		p_vec[p_vec > 0] *= nu_ratio
	# 		p_vec[p_vec <= 0] = self._invalidity_adj["margin"]
	#
	# 		# Choose next obs randomly
	# 		op: Operator = np.random.choice(self.operators, p = p_vec)
	# 		observation = op.observable
	#
	# 		# Apply operator to get next state
	# 		state = op(state)
	# 		state = state / (self.lin_func * state)
	#
	# 		# Save state and obs
	# 		statelist.append(state)
	# 		sequence.append(observation)
	#
	# 		nll_step = - p_vec.dot(np.log2(p_vec))
	# 		nll = nll + (nll_step - nll)/max(1, time_step)
	# 		nlls.append(nll)
	#
	# 	return statelist, nlls, ObsSequence(sequence)
	#
	#
	# def compute(
	# 	self,
	# 	sequence: ObsSequence,
	# 	length_max: Optional[int] = None
	# ) -> tuple[Sequence[State], Sequence[float]]:
	# 	"""
	# 	length_max is settable for mostly debug purposes
	# 	"""
	# 	# Generated observations
	# 	state = self.start_state
	# 	statelist: list[State] = [state]
	# 	nlls: list[np.array] = []
	# 	nll: float = 0
	#
	# 	stop = min(len(sequence), length_max) if length_max else len(sequence)
	# 	for time_step in range(stop):
	# 		p_vec = self.lf_on_operators * state
	#
	# 		# Invalidity adjustment
	# 		p_vec = np.array(p_vec).flatten()
	# 		delta = np.sum(
	# 			self._invalidity_adj["margin"] - p_vec, where = p_vec < 0
	# 		)
	# 		p_plus = np.sum(
	# 			p_vec, where = p_vec > 0
	# 		)
	# 		nu_ratio = 1 - delta / p_plus
	# 		if delta < self._invalidity_adj["setbackMargin"]:
	# 			# Reset by setbackLength and discard what comes after
	# 			time_step -= self._invalidity_adj["setbackLength"]
	# 			state = statelist[-self._invalidity_adj["setbackLength"]]
	# 			statelist = statelist[:-self._invalidity_adj["setbackLength"]]
	#
	# 		# Set negatives to "margin" and adjust valid probabilities
	# 		p_vec[p_vec > 0] *= nu_ratio
	# 		p_vec[p_vec <= 0] = self._invalidity_adj["margin"]
	#
	# 		# Choose operator knowing the next obs
	# 		obs_now = sequence[time_step]
	# 		op = self.operators[self.obsnames.index(obs_now)]
	#
	# 		# Apply operator to get next state
	# 		state = op(state)
	# 		state = state / (self.lin_func * state)
	#
	# 		# Save state and obs
	# 		statelist.append(state)
	#
	# 		nll_step = - p_vec.dot(np.log2(p_vec))
	# 		nll = nll + (nll_step - nll)/max(1, time_step)
	# 		nlls.append(nll)
	#
	# 	return statelist, nlls
	
	
	def generate(
		self,
		length: int
	) -> tuple:
		"""
		
		"""
		return self._sequence_traversal(
			stop = length,
			sequence = ObsSequence([]),
			mode = self._TraversalMode.GENERATE
		)
	
	
	def compute(
		self,
		sequence: ObsSequence,
		length_max: Optional[int] = None
	) -> tuple:
		"""
		
		"""
		return self._sequence_traversal(
			stop = min(len(sequence), length_max) if length_max else len(sequence),
			sequence = sequence,
			mode = self._TraversalMode.COMPUTE
		)
	
	
	def _sequence_traversal(
		self,
		stop: int,
		sequence: ObsSequence,
		mode: _TraversalMode
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
		# Traversal variables
		state = self.start_state
		statelist: list[State] = [state]
		nlllist: list[np.array] = []
		nll: float = 0
		
		ia_margin = self._invalidity_adj["margin"]
		ia_sbmargin = self._invalidity_adj["setbackMargin"]
		ia_sblength = self._invalidity_adj["setbackLength"]
		
		for time_step in range(stop):
			p_vec = self.lf_on_operators * state
			
			# Invalidity adjustment
			p_vec = np.array(p_vec).flatten()
			delta = np.sum(
				ia_margin - p_vec, where = p_vec < 0
			)
			p_plus = np.sum(
				p_vec, where = p_vec > 0
			)
			nu_ratio = 1 - delta / p_plus
			if delta < ia_sbmargin:
				# Reset by setbackLength and discard what comes after
				time_step -= ia_sblength
				state = statelist[-ia_sblength]
				statelist = statelist[:-ia_sblength]
				
				match mode:
					case self._TraversalMode.GENERATE:
						sequence = sequence[:-ia_sblength]
	
			# Set negatives to "margin" and adjust valid probabilities
			p_vec[p_vec > 0] *= nu_ratio
			p_vec[p_vec <= 0] = ia_margin
			
			match mode:
				case self._TraversalMode.COMPUTE:
					# Choose operator knowing the next observation
					obsnow_name: str = sequence[time_step]
					op: Operator = self.operators[self.obsnames.index(obsnow_name)]
				case self._TraversalMode.GENERATE:
					# Choose next observation randomly, then its operator
					op: Operator = np.random.choice(self.operators, p = p_vec)
					obsnow: Observable = op.observable
					sequence.append(obsnow)
					obsnow_name = obsnow.uid
				case _:
					raise NotImplementedError("Can only compute or generate.")
			
			# Apply operator to get next state
			state = op(state)
			state = state / (self.lin_func * state)
			statelist.append(state)
			
			# Get NLL using current observation
			idxoi = self.obsnames.index(obsnow_name)
			nll_step = - p_vec[idxoi] * np.log2(p_vec[idxoi])
			nll = nll + (nll_step - nll)/max(1, time_step)
			nlllist.append(nll)
		
		rpack = (statelist, nlllist)
		match mode:
			case self._TraversalMode.GENERATE:
				rpack += (sequence,)
				
		return rpack
	
	
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