from collections.abc import Sequence, Callable
from typing import Optional

import numpy as np

from .observable import Observable
from .operator import Operator
from .OOM import ObservableOperatorModel


class ContinuousValuedOOM(ObservableOperatorModel):
    
    def __init__(
        self,
        data_dim: int,
        observables: Sequence[Observable],
        operators: Optional[Sequence[Operator]],
        membership_functions: Optional[Sequence[Callable[[np.array], np.array]]]
    ):
        super().__init__(data_dim, observables, operators)
        self._membership_fn: Sequence[Callable[[np.array], np.array]] = membership_functions
    
    
    def validate(
        self
    ) -> bool:
        pass
    
    
    def generate(
        self
    ) -> Sequence[np.array]:
        pass
    
    
    @staticmethod
    def learn(
        input_sequence: Sequence[np.array]
    ) -> type["ContinuousValuedOOM"]:
        pass
    
    
    def unblend(
        self,
        membership_functions: Optional[Sequence[Callable[[np.array], np.array]]]
    ) -> None:
        # Override if given new membership functions
        if membership_functions is not None:
            self._membership_fn = membership_functions
        
        # unblend
        return None #