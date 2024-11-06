from collections.abc import Sequence, Callable
from typing import Optional, Union

import numpy as np

from .observable import Observable
from .operator import Operator
from .OOM import LinFunctional, ObservableOperatorModel, State


class ContinuousValuedOOM(ObservableOperatorModel):
    
    
    def __init__(
		self,
		dim: int,
		linear_functional: LinFunctional,
		operators: Sequence[Operator],
		start_state: State,
        membership_functions: Optional[Sequence[Callable[[np.array], np.array]]],
		invalidity_adjustment: Optional[dict[str, Union[int, float]]] = None
	):
        super().__init__(dim, linear_functional, operators, start_state)
        self._membership_fn = membership_functions
    
    
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