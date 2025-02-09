from dataclasses import dataclass

import numpy as np

from .traversal_mode import TraversalMode, TraversalType
from ..discrete_observable import DiscreteObservable


@dataclass
class TraversalState:
	"""
	Simple dataclass holding information about sequence traversals within an OOM
	"""
	mode: TraversalMode					# flag to indicate GENERATE or COMPUTE modes
	
	type: TraversalType					# flag to indicate DISCRETE or CONTINUOUS
	
	time_step: int						# current time step of the traversal
	
	time_stop: int						# maximum time step of the traversal
	
	state_list: list[np.matrix]			# list of internal states reached by the OOM
										# throughout the traversal
	
	nll_list: list[float]				# list of entropy rate approximations for the
										# traversal sequence at every time step
										# (recorded for convergence analysis)
	
	p_vec_list: list[np.matrix]			# list of probability vectors of the discrete
										# OOM symbols at every time step
	
	sequence: list[DiscreteObservable]			# the proper sequence of discrete symbols
										# either used or generated in the traversal
	
	sequence_cont: list					# the sequence of values either used or
										# generated in the traversal of an OOM in
										# the CONTINUOUS TraversalType (blended OOM)