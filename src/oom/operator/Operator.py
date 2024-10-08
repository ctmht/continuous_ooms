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
        self.observable: Observable = observable
        self.mat: np.array = matrix_rep

        self.validate()

    #

    def __call__(
        self,
        state: np.array
    ) -> np.array:
        # Assert dimension match
        return self.mat * state
    #

    def validate(self) -> bool:
        # Check sums = 1
        pass
    #