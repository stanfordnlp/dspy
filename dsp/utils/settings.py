from contextlib import contextmanager
from dsp.utils.utils import dotdict


class Settings(object):
    _instance = None

    def __new__(cls):
        """
        Singleton Pattern. See https://python-patterns.guide/gang-of-four/singleton/
        """
        
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.stack = []

            config = dotdict()
            cls._instance.__append(config)

        return cls._instance

    @property
    def config(self):
        return self.stack[-1]

    def __getattr__(self, name):
        if hasattr(self.config, name):
            return getattr(self.config, name)
        
        if name in self.config:
            return self.config[name]

        super().__getattr__(name)

    def __append(self, config):
        self.stack.append(config)

    def __pop(self):
        self.stack.pop()
    
    def configure(self, inherit_config=True, **kwargs):
        if inherit_config:
            config = {**self.config, **kwargs}

        self.__append(config)

    @contextmanager
    def context(self, inherit_config=True, **kwargs):
        self.configure(inherit_config=inherit_config, **kwargs)

        try:
            yield
        finally:
            self.__pop()

    def __repr__(self) -> str:
        return repr(self.config)

settings = Settings()
