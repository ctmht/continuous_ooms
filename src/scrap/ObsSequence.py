from collections.abc import Sequence
from collections import Counter
from typing import Union, Tuple, Self
from functools import cached_property
import re

import pandas as pd

from src.oom.discrete_observable.discrete_observable import DiscreteObservable


class ObsSequence:
	"""
	List of observables forming an observed observations
	"""
	
	#################################################################################
	##### Instance creation
	#################################################################################
	def __init__(
		self,
		observations: Union[str, Sequence[DiscreteObservable], Sequence[str]]
	):
		"""
		
		"""
		datafun = self._get_list
		self._data: list[DiscreteObservable] = datafun(observations)
	
	
	def _get_list(self, observations) -> list[DiscreteObservable]:
		if isinstance(observations, ObsSequence):			# discrete_observable sequence
			return observations._data
		
		if isinstance(observations, str):					# joined string
			return [DiscreteObservable(name) for name in observations.split('O')]
		
		if not isinstance(observations, list):				# not a list
			raise TypeError("'observations' is neither a string or a sequence.")
		
		if len(observations) == 0:							# empty list
			return []
		if isinstance(observations[0], DiscreteObservable):			# list of Observables
			return observations
		if isinstance(observations[0], str):				# list of strings
			for idx, strentry in enumerate(observations):
				if strentry[0] == 'O':
					strentry = strentry[1:]
				observations[idx] = DiscreteObservable(strentry)
		
		raise TypeError("'observations' is neither a sequence of strings "
						"nor a sequence of observables.")
	
	
	def _get_str(
		self,
		observations: Union[str, Sequence[DiscreteObservable], Sequence[str]]
	) -> str:
		"""
		
		"""
		if isinstance(observations, str):
			return observations
		if not isinstance(observations, Sequence):
			raise TypeError("observations is not a string or a sequence.")
		
		if isinstance(observations[0], DiscreteObservable):
			return "".join([obs.uid for obs in observations])
		if isinstance(observations[0], str):
			return "".join(observations)
		raise TypeError("observations is not a sequence of strings or observables.")
	
	
	#################################################################################
	##### Instance properties
	#################################################################################
	@cached_property
	def alphabet(
		self
	) -> list[DiscreteObservable]:
		"""
		The alphabet is the complete set of uniquely-identified observables
		encountered in the observation observations.
		
		It is expected that the observation sequences on which this property is
		accessed are long enough to represent the true alphabet of the process, such
		that adding sequences does not affect it.
		"""
		return sorted(
			Counter(self._data).keys(),
			key = lambda obs: obs.uid,
			reverse = False
		)
	
	@property
	def uids(
		self
	):
		return [obs.uid for obs in self._data]
	
	
	def __len__(
		self
	) -> int:
		"""
		The length of the observation observations is the number of observables in its
		string.
		"""
		return len(self._data)
	
	
	#################################################################################
	##### Python methods
	#################################################################################
	def __repr__(
		self
	) -> str:
		"""
		Representation of the observation observations is its string.
		"""
		return self._data.__repr__()
	
	
	def __add__(
		self,
		other: Union[Self, DiscreteObservable]
	):
		"""
		Add two discrete_observable sequences to get a new observation observations, while
		leaving the original two sequences unchanged.
		"""
		if isinstance(other, ObsSequence):
			return ObsSequence(self._data + other._data)
		elif isinstance(other, DiscreteObservable):
			return ObsSequence(self._data + [other.uid])
	
	
	def __iadd__(
		self,
		other: Union[Self, DiscreteObservable]
	):
		"""
		Append the given discrete_observable observations to the current object.
		"""
		if isinstance(other, ObsSequence):
			self._data += other._data
		elif isinstance(other, DiscreteObservable):
			self.append(other)
			
	
	def append(
		self,
		other: DiscreteObservable
	):
		self._data.append(other)
	
	
	def __getitem__(
		self,
		obsseq_slice
	) -> Union[DiscreteObservable, 'ObsSequence']:
		"""
		Access
		"""
		if isinstance(obsseq_slice, int):
			return self._data[obsseq_slice]
		else:
			return ObsSequence(self._data[obsseq_slice])


#############################


def _search_memlim(
	n_obs: int,
	max_mb: float = 50
) -> Tuple[int, int]:
	"""
	
	"""
	# Start search at square matrix dimensions
	clen, ilen = 0, 0
	while True:
		clen += 1
		ilen += 1
		mem = _get_mb_usage(n_obs, clen, ilen)
		if mem > max_mb: clen -= 1; ilen -= 1; break
	
	# Attempt to increase only characteristic words
	while True:
		clen += 1
		mem = _get_mb_usage(n_obs, clen, ilen)
		if mem > max_mb: clen -= 1; break
	
	# Return results maximally fitting under max_mb
	return clen, ilen


def _get_mb_usage(
	n_obs: int,
	l_chr: int,
	l_ind: int
) -> float:
	"""
	
	"""
	# Compute the number of characteristic and indicative words of the given lengths
	n_chr = (n_obs ** (l_chr + 1) - 1) / (n_obs - 1)
	n_ind = (n_obs ** (l_ind + 1) - 1) / (n_obs - 1)
	
	def mat_memsize_bytes(n_rows, n_cols):
		# 8 bytes per entry, 128 bytes of extra machinery
		return 8 * n_rows * n_cols + 128
	
	# F_IJ and F_IzJ (for z in observations) => shape (n_words + 1) x (n_words + 1)
	mem_bigmats = (1 + n_obs) + mat_memsize_bytes(n_ind, n_chr)

	# F_0J and F_I0 => shape 1 x n_words, n_words x 1
	mem_rcvecs = mat_memsize_bytes(n_ind, 1) + mat_memsize_bytes(1, n_chr)

	# 1 MB = 10**6 bytes
	conv_fac = 10 ** 6
	
	return (mem_bigmats + mem_rcvecs) / conv_fac