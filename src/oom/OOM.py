from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from .observable import Observable
from .operator import Operator


State = np.matrix
LinFunctional = np.matrix

class ObservableOperatorModel(ABC):
    """
    Abstract base class for Observable Operator Model classes
    """
    """
    TODO:
    - assertions
    - validation
    - ensuring observables and operators are in 1:1 order
    - generate() and learn() for state sequences
    """
    
    @abstractmethod
    def __init__(
        self,
        dim: int,
        linear_functional: LinFunctional,
        operators: Sequence[Operator],
        start_state: State
    ):
        """
        Constructor for generic Observable Operator Model
        
        Args:
            dim: the dimension of the stochastic process to be modelled
                by this OOM
            linear_functional: the linear functional associated with the
                OOM, in the form of a 1 x dim matrix (row vector)
            operators: the observable operators corresponding 1:1 with the
                possible observations, given as a list of Operator objects
            start_state: the starting state of the OOM, in the form of
                a dim x 1 matrix (column vector)
        """
        # Set state space dimension and observables
        self.dim: int = dim
        self.lin_func: LinFunctional = linear_functional
        self.operators: Sequence[Operator] = operators
        self.start_state: State = start_state
        
        self.observables: Sequence[Observable] = [op.observable for op in operators]
        
        self.validate()
    
    
    @abstractmethod
    def validate(
        self
    ) -> bool:
        pass
    
    
    @abstractmethod
    def generate(
        self,
        invalidity_adjustment: Optional[Union[dict[str, float], bool]]
    ) -> Sequence[np.array]:
        pass