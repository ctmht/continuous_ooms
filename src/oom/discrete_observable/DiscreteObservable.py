class DiscreteObservable():
    """
    
    """
    
    _instances: dict[str, 'DiscreteObservable'] = {}
    
    def __new__(cls, name):
        if name in DiscreteObservable._instances:
            return DiscreteObservable._instances[name]
        else:
            return super(DiscreteObservable, cls).__new__(cls)
    
    def __init__(self, name):
        if hasattr(self, 'uid'):
            return
        self.uid = 'O' + name
        DiscreteObservable._instances[name] = self
    
    def __repr__(self):
        return self.uid