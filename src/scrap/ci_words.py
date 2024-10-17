from collections import Counter

import pandas as pd


def get_dim(n_obs, l_chr, l_ind):
	return int((n_obs ** (l_chr + 1) - 1) *\
			   (n_obs ** (l_ind + 1) - 1) /\
			   ((n_obs - 1) ** 2))


def get_memory_usage_MB_new(
	n_obs,
	l_chr,
	l_ind
):
	n_chr = (n_obs ** (l_chr + 1) - 1) / (n_obs - 1)
	n_ind = (n_obs ** (l_ind + 1) - 1) / (n_obs - 1)
	
	def mat_memsize_bytes(n_rows, n_cols):
		# 8 bytes per entry
		# 128 bytes of machinery
		return 8 * n_rows * n_cols + 128
	
	# F_IJ and F_IzJ (for z in observations) => shape (n_words + 1) x (n_words + 1)
	mem_bigmats = (1 + n_obs) + mat_memsize_bytes(n_ind, n_chr)

	# F_0J and F_I0 => shape 1 x n_words, n_words x 1
	mem_rcvecs = mat_memsize_bytes(n_ind, 1) + mat_memsize_bytes(1, n_chr)

	# 1 MB = 10**6 bytes
	conv_fac = 10 ** 6
	
	return (mem_bigmats + mem_rcvecs) / conv_fac


def search_memlim(n_obs, max_mb: float = 50):
	mem = 0
	clen, ilen = 0, 0
	while True:
		clen += 1
		ilen += 1
		dim = get_dim(n_obs, clen, ilen)
		mem = get_memory_usage_MB_new(n_obs, clen, ilen)
		
		if mem > max_mb:
			clen -= 1
			ilen -= 1
			break
		else:
			print(f"{clen=} {ilen=} {mem=}")
	
	while True:
		clen += 1
		dim = get_dim(n_obs, clen, ilen)
		mem = get_memory_usage_MB_new(n_obs, clen, ilen)
		
		if mem > max_mb:
			clen -= 1
			break
		else:
			print(f"{clen=} {ilen=} {mem=}")
	return clen, ilen


def construct_words(
	observation_sequence,
	maxlen,
	possible_observations = None
):
	if possible_observations is None:
		# Get alphabet by unique observations
		possible_observations = Counter(observation_sequence).keys()
	
	words = ['']
	
	for wlen in range(1, maxlen + 1):
		# Save reference for which words already exist
		cur_nwords = len(words)

		# Generate all words of length wlen by extending the word list
		for idx, word in enumerate(words):
			# Iterate through words that existed at the start of first loop
			if idx >= cur_nwords:
				break
			
			for obs in possible_observations:
				# Get new word
				new_word = word + obs
				if new_word in words:
					continue

				# Keep # if relevant
				# if prop > 0:
				words.append(new_word)

		# Remove empty word
		try:
			words.remove('')
		except ValueError:
			pass

	words_srs = pd.Series(words).apply(len)
	words_srs.index = words
	return words_srs


def get_char_ind_words(observation_sequence, max_mb = 50):
	possible_observations = Counter(observation_sequence).keys()
	
	# clen >= ilen by construction
	clen, ilen = search_memlim(len(possible_observations), max_mb = max_mb)
	
	cwords = construct_words(observation_sequence, maxlen = clen)
	iwords = cwords[cwords <= ilen]
	
	return cwords, iwords