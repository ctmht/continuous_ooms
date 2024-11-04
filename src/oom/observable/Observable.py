from functools import cached_property
from typing import Any


class Observable:
    """
    
    """
    
    #################################################################################
	##### Instance creation
	#################################################################################
    def __init__(
        self,
        oid: Any
    ):
        """
        
        """
        self.oid: str = str(oid)
    
    
    #################################################################################
    ##### Instance properties
    #################################################################################
    @cached_property
    def name(
        self
    ) -> str:
        """
        Property representing an observable instance's name based on its observation
        ID. For the purposes of regex and admitting arbitrarily large discrete
        alphabet sizes, the observations have the name 'O' + oid.
        """
        return 'O' + self.oid
    
    
    #################################################################################
    ##### Python methods
    #################################################################################
    def __repr__(
        self
    ) -> str:
        """
        The representation of an observable instance is its own name.
        """
        return self.name