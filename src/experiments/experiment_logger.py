import logging
datalog = logging.getLogger("data_logger_ooms")
genlog = logging.getLogger("general_logger_ooms")
logging.basicConfig(
	format   = '\t\t%(levelname)s | \t%(funcName)30s():\t%(message)s\n',
	level    = logging.DEBUG,
	filename = "LOG_experiment_learning_discrete.txt",
	filemode = 'w+'
)

import numpy as np
np.set_printoptions(linewidth = 1000, precision = 5, suppress = True, legacy='1.25')