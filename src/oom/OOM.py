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
    - ensuring _observables and _operators are in 1:1 order
    - generate() and learn() for state sequences
    """
    
    @abstractmethod
    def __init__(
        self,
        data_dim: int,
        linear_functional: LinFunctional,
        operators: Sequence[Operator],
        start_state: State
    ):
        """
        Constructor for generic Observable Operator Model
        
        Args:
            data_dim: the dimension of the stochastic process to be modelled
                    by this OOM
            observables: the possible observations of the OOM, given as a
                    sequence of Observable objects
            operators: optionally, the observable operators corresponding
                    1:1 with the given observables, given as a list of
                    Operator objects
        """
        # Set state space dimension and observables
        self._state_space_dim: int = data_dim
        self._lin_functional: LinFunctional = linear_functional
        self._operators: Sequence[Operator] = operators
        self._start_state: State = start_state
        
        self._observables: Sequence[Observable] = [op.observable for op in operators]
        
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
    
    
    @staticmethod
    @abstractmethod
    def learn(
        input_sequence: Sequence[np.array]
    ) -> type["ObservableOperatorModel"]:
        pass