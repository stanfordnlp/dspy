from abc import ABC, abstractmethod


class BaseTracker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def call(self, **kwargs):
        pass
