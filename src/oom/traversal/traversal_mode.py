from enum import Enum


class TraversalMode(Enum):
	"""
	Simple enum for tracking whether OOM is in generate/compute tvmode
	"""
	GENERATE = 0
	COMPUTE = 1


class TraversalType(Enum):
	"""
	Simple enum for tracking whether OOM is used for discrete/continuous values
	"""
	DISCRETE = 0
	CONTINUOUS = 1