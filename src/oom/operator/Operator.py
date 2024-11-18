from typing import Optional

import numpy as np

from ..observable import Observable


class Operator:
    """
    TODO:
    - __call__ and multiplication by State matrix identical
    """

    def __init__(
        self,
        observable: Observable,
        range_dimension: int,
        matrix_rep: np.matrix
    ):
        self.observable: Observable = observable
        self.mat: np.matrix = matrix_rep

        self.validate()
    

    def __call__(
        self,
        state: np.array
    ) -> np.array:
        # Assert dimension match
        return self.mat * state
    

    def validate(self) -> bool:
        # Check sums = 1
        pass
    