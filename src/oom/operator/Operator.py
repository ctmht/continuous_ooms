from typing import Optional

import numpy as np

from ..observable import Observable


class Operator:
    """
    TODO:
    - inherit from numpy matrix??
    - assertions
    - validation
    - random matrix adjustment to validity (OPEN PROBLEM)
    """

    def __init__(
        self,
        observable: Observable,
        range_dimension: int,
        matrix_rep: Optional[np.array]
    ):
        self._observable: Observable = observable

        if matrix_rep is not None:
            assert matrix_rep.shape[-1] == range_dimension, \
                "Matrix 'matrix_rep' dimensions must match data dimension" \
                f"'range_dimension', instead have {matrix_rep.shape[-1]} " \
                f"and {range_dimension}."
            self._matrix_rep: np.array = matrix_rep
        else:
            self._matrix_rep = None

        self.validate()

    #

    def __call__(
        self,
        state: np.array
    ) -> np.array:
        # Assert dimension match
        return self._matrix_rep * state
    #

    def validate(self) -> bool:
        # Check sums = 1
        pass
    #