from collections.abc import Sequence, Callable
from typing import Optional

import numpy as np

from .observable import Observable
from .operator import Operator
from .OOM import ObservableOperatorModel
from .ContinuousValuedOOM import ContinuousValuedOOM


class DiscreteValuedOOM(ObservableOperatorModel):
    
    def __init__(
        self,
        data_dim: int,
        observables: Sequence[Observable],
        operators: Optional[Sequence[Operator]]
    ):
        super().__init__(data_dim, observables, operators)
    
    
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
    ) -> type["DiscreteValuedOOM"]:
        pass
    
    
    def blend(
        self,
        membership_functions: Optional[Sequence[Callable[[np.array], np.array]]]
    ) -> ContinuousValuedOOM:
        # Blend using membership_functions if given
        # Learn membership_functions otherwise
        
        # unblend
        return ContinuousValuedOOM() #