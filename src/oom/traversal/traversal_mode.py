from enum import Enum


class TraversalMode(Enum):
	"""
	Simple enum for tracking whether OOM is in generate/compute tvmode
	"""
	GENERATE = 0
	COMPUTE = 1