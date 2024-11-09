from collections.abc import Sequence
from collections import Counter
from typing import Union, Tuple, Self
from functools import cached_property
import re

import pandas as pd

from .Observable import Observable


class ObsSequence:
	"""
	List of observables forming an observed observations
	"""
	
	#################################################################################
	##### Instance creation
	#################################################################################
	def __init__(
		self,
		observations: Union[str, Sequence[Observable], Sequence[str]]
	):
		"""
		
		"""
		datafun = self._get_list
		self._data: list[Observable] = datafun(observations)
	
	
	def _get_list(self, observations) -> list[Observable]:
		if isinstance(observations, str):
			return [Observable(name) for name in observations.split('O')]
		if not isinstance(observations, list):
			raise TypeError("observations is not a string or a sequence.")
		
		if len(observations) == 0:
			return []
		if isinstance(observations[0], Observable):
			return observations
		if isinstance(observations[0], str):
			for idx, strentry in enumerate(observations):
				if strentry[0] == 'O':
					strentry = strentry[1:]
				observations[idx] = Observable(strentry)
		raise TypeError("observations is not a sequence of strings or observables.")
	
	
	def _get_str(
		self,
		observations: Union[str, Sequence[Observable], Sequence[str]]
	) -> str:
		"""
		
		"""
		if isinstance(observations, str):
			return observations
		if not isinstance(observations, Sequence):
			raise TypeError("observations is not a string or a sequence.")
		
		if isinstance(observations[0], Observable):
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
	) -> list[Observable]:
		"""
		The alphabet is the complete set of uniquely-identified observables
		encountered in the observation observations.
		
		It is expected that the observation sequences on which this property is
		accessed are long enough to represent the true alphabet of the process, such
		that adding sequences does not affect it.
		"""
		return sorted(Counter([obs for obs in self._data]).keys())
	
	
	def __len__(
		self
	) -> int:
		"""
		The length of the observation observations is the number of observables in its
		string.
		"""
		return len(self._data)
	
	
	#################################################################################
	##### Subsequences and getting characteristic / indicative words
	#################################################################################
	# def count_sub(
	# 	self,
	# 	other: Union[str, Sequence[Observable], Sequence[str], Self]
	# ) -> int:
	# 	"""
	# 	Counts the number of times a given subsequence appears in this observation
	# 	sequence.
	# 	"""
	# 	otherstr = other._str if isinstance(other, ObsSequence)\
	# 						  else self._get_str(other)
	# 	pattern = "(?=(" + otherstr + "))"
	# 	count = len(re.findall(pattern, self._str))
	# 	return count
	#
	#
	# def estimate_char_ind(
	# 	self,
	# 	memory_limit_mb: float = 50
	# ) -> Tuple[Sequence[Self], Sequence[Self]]:
	# 	"""
	# 	Get the complete sets of characteristic and indicative words of any lengths
	# 	under a given threshold, constrained by the memory usage of the spectral
	# 	learning algorithm they are then used on.
	# 	"""
	# 	clen, ilen = _search_memlim(len(self.alphabet), max_mb = memory_limit_mb)
	#
	# 	cwords = self._construct_words(maxlen = clen)
	# 	iwords = cwords[cwords <= ilen]
	#
	# 	return cwords.index.values, iwords.index.values
	#
	#
	# def _construct_words(
	# 	self,
	# 	maxlen: int
	# ) -> pd.Series:
	# 	"""
	# 	Construct all sets of words of length less than a specified maximum
	# 	"""
	# 	words = ['']
	#
	# 	for wlen in range(1, maxlen + 1):
	# 		# Save reference for which words already exist
	# 		cur_nwords = len(words)
	#
	# 		# Generate all words of length wlen by extending the word list
	# 		for idx, word in enumerate(words):
	# 			# Iterate through words that existed at the start of first loop
	# 			if idx >= cur_nwords:
	# 				break
	#
	# 			for obs in self.alphabet:
	# 				# Get new word
	# 				new_word = word + obs.uid
	#
	# 				if new_word in words:
	# 					continue
	# 				words.append(new_word)
	#
	# 		# Remove empty word
	# 		if '' in words:
	# 			words.remove('')
	#
	# 	words = [ObsSequence(word) for word in words]
	# 	words_srs = pd.Series(words).apply(len)
	# 	words_srs.index = words
	# 	return words_srs
	
	
	#################################################################################
	##### Python methods
	#################################################################################
	def __repr__(
		self
	) -> str:
		"""
		Representation of the observation observations is its string.
		"""
		return self._data
	
	
	def __add__(
		self,
		other: Union[Self, Observable]
	):
		"""
		Add two observable sequences to get a new observation observations, while
		leaving the original two sequences unchanged.
		"""
		if isinstance(other, ObsSequence):
			return ObsSequence(self._data + other._data)
		elif isinstance(other, Observable):
			return ObsSequence(self._data + [other.uid])
	
	
	def __iadd__(
		self,
		other: Union[Self, Observable]
	):
		"""
		Append the given observable observations to the current object.
		"""
		if isinstance(other, ObsSequence):
			self._data += other._data
		elif isinstance(other, Observable):
			self.append(other)
			
	
	def append(
		self,
		other: Observable
	):
		self._data.append(other.uid)
	
	
	def __getitem__(
		self,
		obsseq_slice
	) -> Union[Observable, Self]:
		"""
		Access
		"""
		return self._data[obsseq_slice]


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