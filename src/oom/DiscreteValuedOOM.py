from collections.abc import Sequence, Callable
from typing import Optional, Union, Self

import numpy as np

from .observable import Observable, ObsSequence
from .operator import Operator
from .util import get_matrix_estimates, get_CQ_by_svd
from .OOM import ObservableOperatorModel, State, LinFunctional
from .ContinuousValuedOOM import ContinuousValuedOOM


class DiscreteValuedOOM(ObservableOperatorModel):
	default_adjustment: dict[str, Union[int, float]] = {
		"margin": 0.005,
		"setbackMargin": -0.3,
		"setbackLength": 2
	}
	
	
	def __init__(
		self,
		dim: int,
		linear_functional: LinFunctional,
		operators: Sequence[Operator],
		start_state: State,
		invalidity_adjustment: Optional[dict[str, Union[int, float]]] = None
	):
		"""
		Initialize an OOM for a discrete-valued process
		
		Args:
			dim: the dimension of the stochastic process to be modelled
				by this OOM
			linear_functional: the linear functional associated with the
				OOM, in the form of a 1 x dim matrix (row vector)
			operators: the observable operators corresponding 1:1 with the
				possible observations, given as a list of Operator objects
			start_state: the starting state of the OOM, in the form of
				a dim x 1 matrix (column vector)
			invalidity_adjustment: dictionary containing the margin,
				setbackMargin, and setbackLength parameters for running OOMs
				as invalid sequence generators (stepping back in the generation
				process whenever negative probabilities become an issue)
		"""
		super().__init__(dim, linear_functional, operators, start_state)
		self._invalidity_adj = self._set_invalidity_adjustment(invalidity_adjustment)
	
	
	def _set_invalidity_adjustment(
		self,
		invalidity_adjustment: Optional[dict[str, Union[int, float]]]
	) -> dict[str, Union[int, float]]:
		"""
		Set the dictionary of parameters corresponding to the adjustment
		required to run OOMs as invalid sequence generators
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
				"setbackLength", self.default_adjustment["setbackMargin"]
			)
		}
		return inv
	
	
	def validate(
		self
	) -> bool:
		pass
	
	
	def generate(
		self,
		length: int = 100
	) -> tuple[Sequence[State], Sequence[Observable]]:
		"""
		Use the OOM to generate a sequence of a given length from the
		discrete-valued, stationary, ergodic process it models
		
		Args:
			length: the desired generation length
		Returns:
			(statelist, observations): tuple of the visited states 'statelist'
				and corresponding observations 'observations'; every item
				statelist[index] corresponds to observations[index - 1], given
				the starting state statelist[0] corresponds to the "empty
				obs"
		"""
		# Precomputed operator matrix
		functional_on_operators = np.asmatrix(
			np.vstack([self.lin_func * op.mat for op in self.operators])
		)
		
		# Generated sequence
		state = self.start_state
		statelist: list[State] = [state]
		observations: list[Observable] = []
		
		for time_step in range(length):
			p_vec = functional_on_operators * state
			p_vec = p_vec #/ np.sum(p_vec) # TODO: forced normalization???
			
			# Invalidity adjustment
			p_vec = np.array(p_vec).flatten()
			delta = np.sum(
				self._invalidity_adj["margin"] - p_vec, where = p_vec < 0
			)
			if delta < self._invalidity_adj["setbackMargin"]:
				# Reset by setbackLength and discard what comes after
				time_step -= self._invalidity_adj["setbackLength"]
				state = statelist[-self._invalidity_adj["setbackLength"]]
				statelist = statelist[:-self._invalidity_adj["setbackLength"]]
				observations = observations[:-self._invalidity_adj["setbackLength"]]
			
			# Choose next obs randomly
			op: Operator = np.random.choice(self.operators, p = p_vec)
			observation = op.observable
			
			# Apply operator to get next state
			state = op(state)
			state = state / (self.lin_func * state)
			
			# Save state and obs
			statelist.append(state)
			observations.append(observation)
   
		return statelist, observations
	
	
	def blend(
		self,
		membership_functions: Optional[Sequence[Callable[[np.array], np.array]]]
	) -> 'ContinuousValuedOOM':
		# Blend using membership_functions if given
		# Learn membership_functions otherwise
		
		# unblend
		pass # return ContinuousValuedOOM() #
	
	
	@staticmethod
	def from_data(
		obs: ObsSequence,
		target_dimension: int,
		memory_limit_mb: float = 50
	) -> 'DiscreteValuedOOM':
		"""
		
		"""
		# Get characteristic and indicative words within given memory limit
		chr_w, ind_w = obs.estimate_char_ind(memory_limit_mb = memory_limit_mb)
		print(f"{len(chr_w)} char. words, {len(ind_w)} ind. words")
		
		# Estimate large matrices and get characterizer and inductor matrices spectrally
		F_0J, F_I0, F_IJ, F_IzJ = get_matrix_estimates(obs, chr_w, ind_w)
		C, Q = get_CQ_by_svd(F_IJ, target_dimension)
		
		# Inverse matrix computation
		V = C * F_IJ * Q
		V_inv = np.linalg.inv(V)
	
		# Get linear functional (learning equation #1)
		# It holds that type(sigma) == LinFunctional == np.matrix
		sigma = F_0J * Q * V_inv
	
		# Get observable operators (learning equation #2)
		tau_z = []
		for F_IkJ, observable in zip(F_IzJ, obs.alphabet):
			operator_matrix = C * F_IkJ * Q * V_inv
			operator = Operator(observable,
								range_dimension = target_dimension,
								matrix_rep = operator_matrix)
			tau_z.append(operator)
		
		# Get state vector (learning equation #3)
		# It holds that type(sigma) == State == np.matrix
		omega = C * F_I0
		
		return DiscreteValuedOOM(dim = target_dimension,
								 linear_functional = sigma,
								 operators = tau_z,
								 start_state = omega)