from typing import Optional, Union, override

import numpy as np
from scipy.stats import rv_continuous

from . import DiscreteValuedOOM
from .oom import ObservableOperatorModel
from .discrete_observable import DiscreteObservable
from .traversal import TraversalMode, TraversalState
from .util import _MfLookup


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
		
		# Replaces mf.pdf(sequence_item) sequential computation with
		# accessing precomputed PDF values by mf_lookup[observable, index(sequence_item)]
		self._mf_lookup = _MfLookup(
			dict(zip(self.observables, self.membership_fns)),
			steps_pdfs = 10000
		)
		
	
	
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
		
		for obs, op in zip(self.observables, self.operators):
			strrep += f"    {obs} operator matrix:\n{op}\n"
		
		return strrep
	
	
	@override
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
		traversal_obj.sequence_cont = []
		
		self._mf_lookup.update_rvs(10000)
		
		traversal_obj = self._sequence_traversal(traversal_obj)
		
		return traversal_obj
	
	
	@override
	def compute(
		self,
		sequence: list,
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
		traversal_obj.sequence_cont = sequence
		
		traversal_obj = self._sequence_traversal(traversal_obj)
		
		return traversal_obj
	
	
	@override
	def setback_state(
		self
	) -> np.matrix:
		"""
		
		"""
		self.tv.n_setbacks += 1
		
		ia_sblength = self._invalidity_adj["setbackLength"]
		
		newstate = self.start_state
		if self.tv.time_step > ia_sblength + 1:
			for sblength in range(ia_sblength, 0, -1):
				_ZERO_STATE_ERROR = False
				for obs in self.tv.sequence[-sblength:]:
					op = self.step_get_operator(obs)
					if np.all(op * newstate == 0):
						_ZERO_STATE_ERROR = True
						break
					newstate = op * newstate
					newstate = newstate / (self.lin_func * newstate)
				if _ZERO_STATE_ERROR: continue
				else: break
		return newstate
	
	
	def step_get_distribution(
		self,
		state: np.matrix
	) -> np.array:
		"""
		
		"""
		# Get probability vector
		p_vec = self.lf_on_operators * state
		p_vec = np.array(p_vec).flatten()
		
		return p_vec
	
	
	def step_get_observation(
		self,
		p_vec: np.array
	):
		"""
		
		"""
		match self.tv.mode:
			case TraversalMode.COMPUTE:
				obs = self.tv.sequence_cont[self.tv.time_step - 1]
			case TraversalMode.GENERATE:
				# Choose next observation randomly, then its operator
				# d_obs = self.observables[
				# 	np.argmax(np.cumsum(self.tv.p_vec_list[-1]) > np.random.rand())
				# ]
				d_obs = np.random.choice(self.observables, p = p_vec)
				self.tv.sequence.append(d_obs)
				
				# Use precomputed RVs values
				if not self._mf_lookup.holds_rvs(d_obs):
					self._mf_lookup.update_rvs(10000, d_obs)
				
				obs = self._mf_lookup('rvs', d_obs)
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
		match self.tv.mode:
			case TraversalMode.GENERATE:
				# Use precomputed PDF values
				membership_weights = self._mf_lookup('pdfs')
			case TraversalMode.COMPUTE:
				# Use precomputed PDF values
				if not self._mf_lookup.holds_pdfs(
					self.tv.time_step,
					self.tv.time_step + 1
				):
					self._mf_lookup.update_pdfs(
						self.tv.sequence_cont,
						start = self.tv.time_step
					)
				
				self._membership_weights = np.array([
					self._mf_lookup[dobs, self.tv.time_step - 1]
					for dobs in self.observables
				])
				membership_weights = self._membership_weights
			case _:
				raise NotImplementedError("Can only compute or generate.")
		
		wmat = np.tensordot(
			self.operators, membership_weights, axes=([0], [0])
		) * np.sum(membership_weights)
		
		return np.asmatrix(wmat)
	
	
	def step_get_ll(
		self,
		obs
	) -> float:
		"""
		
		"""
		match self.tv.mode:
			case TraversalMode.GENERATE:
				membership_weights = np.array(self._mf_lookup('pdfs'))
			case TraversalMode.COMPUTE:
				membership_weights = self._membership_weights
			case _:
				raise NotImplementedError("Can only compute or generate.")
		
		p = np.dot(self.tv.p_vec_list[-1], membership_weights)
		ll = np.log2(p)
		
		return ll
	
	
	#################################################################################
	##### LEARNING
	#################################################################################
	@staticmethod
	@override
	def from_data(
		obs: list[DiscreteObservable | str | int],
		target_dimension: int,
		len_cwords: int,
		len_iwords: int,
		membership_functions: list[rv_continuous],
		observables: Optional[list[DiscreteObservable]] = None,
		estimated_matrices: Optional[tuple[np.matrix]] = None,
	) -> 'ContinuousValuedOOM':
		"""
		
		"""
		from .util import learn_continuous_valued_oom
		
		return learn_continuous_valued_oom(
			obs,
			target_dimension,
			len_cwords,
			len_iwords,
			membership_functions,
			observables,
			estimated_matrices
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