from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

import numpy as np

from .observable import Observable
from .operator import Operator


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
        observables: Sequence[Observable],
        operators: Optional[Sequence[Operator]]
    ):
        # Set state space dimension and observables
        self._state_space_dim: int = data_dim
        self._observables: Sequence[Observable] = observables

        # Set operators
        if operators is not None:
            self._operators: Sequence[Operator] = operators
        else:
            # Generate new operators
            newopl: list[Operator] = []
            for observable in self._observables:
                newopl.append(Operator(observable, data_dim))
            self._operators: Sequence[Operator] = newopl

        self.validate()
    
    
    @abstractmethod
    def validate(
        self
    ) -> bool:
        pass
    
    
    @abstractmethod
    def generate(
        self
    ) -> Sequence[np.array]:
        pass
    
    
    @staticmethod
    @abstractmethod
    def learn(
        input_sequence: Sequence[np.array]
    ) -> type["ObservableOperatorModel"]:
        pass