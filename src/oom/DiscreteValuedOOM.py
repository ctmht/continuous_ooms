from collections.abc import Sequence, Callable
from typing import Optional, Union

import numpy as np

from .observable import Observable
from .operator import Operator
from .OOM import ObservableOperatorModel, State, LinFunctional
from .ContinuousValuedOOM import ContinuousValuedOOM


class DiscreteValuedOOM(ObservableOperatorModel):
    default_adjustment: dict[str, Union[int, float]] = {
        "margin": 0.005,
        "setbackMargin": -0.3,
        "setbackLength": 2
    }
    
    
    def __init__(
        self,
        data_dim: int,
        linear_functional: LinFunctional,
        operators: Sequence[Operator],
        start_state: State
    ):
        super().__init__(data_dim, linear_functional, operators, start_state)
        self._invalidity_adj = self.default_adjustment
    
    
    def validate(
        self
    ) -> bool:
        pass
    
    
    def generate(
        self,
        length: int = 100
    ) -> tuple[Sequence[State], Sequence[Observable]]:
        # Precomputed matrix
        Sigma = np.asmatrix(np.vstack([self._lin_functional * op.mat for op in self._operators]))
        
        # Generated sequence
        state = self._start_state
        statelist: list[State] = [state]
        observations: list[Observable] = []
        
        for t in range(length):
            p = Sigma * state
            
            # Invalidity adjustment
            p = np.array(p).flatten()
            delta = np.sum(self._invalidity_adj["margin"] - p, where = p < 0)
            if delta < self._invalidity_adj["setbackMargin"]:
                # Reset by setbackLength
                t -= self._invalidity_adj["setbackLength"]
                w = statelist[-self._invalidity_adj["setbackLength"]]
                statelist = statelist[:-self._invalidity_adj["setbackLength"]]
                observations = observations[:-self._invalidity_adj["setbackLength"]]
            
            # Update state
            op: Operator = np.random.choice(self._operators, p = p)
            observation = op.observable
            state = op.mat * state
            state = state / (self._lin_functional * state)
            
            # Save state
            statelist.append(state)
            observations.append(observation)
        return statelist, observations
    
    
    def _set_invalidity_adjustment(
        self,
        invalidity_adjustment: Optional[dict[str, float]]
    ) -> None:
        if invalidity_adjustment is not None:
            self._invalidity_adj["margin"] = invalidity_adjustment.get("margin")
            self._invalidity_adj["setbackMargin"] = invalidity_adjustment.get("setbackMargin")
            self._invalidity_adj["setbackLength"] = invalidity_adjustment.get("setbackLength")
    
    
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