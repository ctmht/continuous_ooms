from functools import cached_property
from typing import Any


class Observable():
    _instances: dict[str, 'Observable'] = {}

    def __new__(cls, name):
        if name in Observable._instances:
            return Observable._instances[name]
        else:
            return super(Observable, cls).__new__(cls)
    
    def __init__(self, name):
        if hasattr(self, 'uid'):
            return
        self.uid = 'O' + name
        Observable._instances[name] = self

    def __repr__(self):
        return self.uid