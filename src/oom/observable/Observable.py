from typing import Optional


class Observable:
    """
    TODO:
    - nothing really for now
    """
    
    def __init__(
        self,
        oid: str
    ):
        """
        
        """
        self.oid: str = oid
    
    
    def name(
        self
    ) -> str:
        """
        
        """
        return 'O' + self.oid
    
    
    def __repr__(
        self
    ) -> str:
        """
        
        """
        return self.name()