from collections.abc import Sequence
from typing import Union, Tuple, Self
from collections import Counter
import re

import pandas as pd

from .Observable import Observable


class ObsSequence:
	"""
	List of observables forming an observed sequence
	"""

	def __init__(
		self,
		sequence: Union[Sequence[Observable], Sequence[str], str]
	):
		"""
		
		"""
		self._str: str = self._get_seq(sequence)
		self.alphabet = [Observable(oid) for oid in sorted(Counter(self._str.split('O')[1 :]).keys())]
	
	
	def _get_seq(
		self,
		sequence: Union[Sequence[Observable], str]
	) -> str:
		"""
		
		"""
		if type(sequence) == str:
			return sequence
		else:
			return "".join([obs.name() for obs in sequence])
	
	
	def __repr__(
		self
	) -> str:
		"""
		
		"""
		return self._str
	
	
	def to_string(
		self
	) -> str:
		"""
		
		"""
		return self._str
	
	
	def __str__(
		self
	) -> str:
		"""
		
		"""
		return self._str
	
	
	def __len__(
		self
	) -> int:
		"""
		
		"""
		return len(self._str.split('O')[1 : ])
	
	
	def __add__(
		self,
		other: Union[Self, Observable]
	):
		"""
		
		"""
		if type(other) == ObsSequence:
			return ObsSequence(self._str + other._str)
		elif type(other) == Observable:
			return ObsSequence(self._str + other.name())
	
	
	def __iadd__(
		self,
		other: Union[Self, Observable]
	):
		"""
		
		"""
		if type(other) == ObsSequence:
			self._str += other._str
		elif type(other) == Observable:
			self._str += other
	
	
	def __getitem__(
		self,
		item
	) -> Union[Observable, Self]:
		"""
		
		"""
		return re.findall('(O[^(O)]+)', self._str)[item]
	
	
	def count_sub(
		self,
		other: Self
	) -> int:
		"""
		
		"""
		other = other._str
		pattern = "(?=(" + other + "))"
		count = len(re.findall(pattern, self._str))
		return count
	
	
	def estimate_char_ind(
		self,
		memory_limit_mb: float = 50
	) -> Tuple[Sequence[Self], Sequence[Self]]:
		"""
		
		"""
		clen, ilen = _search_memlim(len(self.alphabet), max_mb = memory_limit_mb)
		
		cwords = self._construct_words(maxlen = clen)
		iwords = cwords[cwords <= ilen]
		
		return cwords.index.values, iwords.index.values
	
	
	def _construct_words(
		self,
		maxlen: int
	) -> pd.Series:
		"""
		
		"""
		words = ['']
		
		for wlen in range(1, maxlen + 1):
			# Save reference for which words already exist
			cur_nwords = len(words)
	
			# Generate all words of length wlen by extending the word list
			for idx, word in enumerate(words):
				# Iterate through words that existed at the start of first loop
				if idx >= cur_nwords: break
				
				for obs in self.alphabet:
					# Get new word
					new_word = word + obs.name()
					if new_word in words: continue
					
					words.append(new_word)
				
			# Remove empty word
			if '' in words: words.remove('')
		
		words = [ObsSequence(word) for word in words]
		words_srs = pd.Series(words).apply(len)
		words_srs.index = words
		return words_srs


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