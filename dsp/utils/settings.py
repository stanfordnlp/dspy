from contextlib import contextmanager
from dsp.utils.utils import dotdict


class Settings(object):
    """DSP configuration settings."""

    _instance = None
    branch_idx: int = 0

    def __new__(cls):
        """
        Singleton Pattern. See https://python-patterns.guide/gang-of-four/singleton/
        """

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.stack = []

            #  TODO: remove first-class support for re-ranker and potentially combine with RM to form a pipeline of sorts
            #  eg: RetrieveThenRerankPipeline(RetrievalModel, Reranker)
            #  downstream operations like dsp.retrieve would use configs from the defined pipeline.
            config = dotdict(
                lm=None,
                rm=None,
                reranker=None,
                compiled_lm=None,
                force_reuse_cached_compilation=False,
                compiling=False,
            )
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

    def configure(self, inherit_config: bool = True, **kwargs):
        """Set configuration settings.

        Args:
            inherit_config (bool, optional): Set configurations for the given, and use existing configurations for the rest. Defaults to True.
        """
        if inherit_config:
            config = {**self.config, **kwargs}
        else:
            config = {**kwargs}

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
