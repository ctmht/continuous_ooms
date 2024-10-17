from typing import Sequence

import numpy as np


from src.oom.observable import ObsSequence


# def count_appearences(
# 	obs,
# 	*subwords
# ) -> int:
# 	word = "".join(subwords)
# 	pattern = "(?=(" + word + "))"
# 	count = len(re.findall(pattern, obs))
# 	return count
#
#
# def f_estimate(
# 	obs,
# 	*subwords
# ):
# 	word = "".join(subwords)
# 	count = count_appearences(obs, word)
# 	return count / (len(obs) - len(word) + 1)


# def word_generator(
# 	possible_observations,
# 	maxlen
# ) -> Iterator[str]:
# 	words = ['']
#
# 	for wlen in range(1, maxlen + 1):
# 		# Save reference for which words already exist
# 		cur_nwords = len(words)
#
# 		for idx, word in enumerate(words):
# 			# Iterate through words that existed at the start of first loop
# 			if idx >= cur_nwords:
# 				break
#
# 			for obs in possible_observations:
# 				# Append another obs
# 				new_word = word + obs
# 				if new_word in words:
# 					continue
#
# 				yield new_word










# def learn_OOM(observation_sequence, target_dimension):
# 	possible_observations = Counter(observation_sequence).keys()
#
# 	chr_w, ind_w = get_char_ind_words(observation_sequence)
#
# 	(sigma, tau, omega), F_IJ = estimate_OOM(
# 		observation_sequence, possible_observations,
# 		chr_w, ind_w, target_dimension
# 	)
#
# 	return sigma, tau, omega
#
# def learn_spectral(
# 	observation_sequence: Sequence[Observable],
# 	target_dimension: int
# ) -> DiscreteValuedOOM:
# 	sigma, tau, omega = learn_OOM(observation_sequence, target_dimension)
#
# 	return DiscreteValuedOOM(target_dimension, sigma, tau, omega)