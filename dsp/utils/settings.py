import threading
from contextlib import contextmanager

from dsp.utils.utils import dotdict


class Settings:
    """DSP configuration settings."""

    _instance = None

    def __new__(cls):
        """
        Singleton Pattern. See https://python-patterns.guide/gang-of-four/singleton/
        """

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.lock = threading.Lock()
            cls._instance.main_tid = threading.get_ident()
            cls._instance.main_stack = []
            cls._instance.stack_by_thread = {}
            cls._instance.stack_by_thread[threading.get_ident()] = cls._instance.main_stack

            #  TODO: remove first-class support for re-ranker and potentially combine with RM to form a pipeline of sorts
            #  eg: RetrieveThenRerankPipeline(RetrievalModel, Reranker)
            #  downstream operations like dsp.retrieve would use configs from the defined pipeline.
            config = dotdict(
                lm=None,
                adapter=None,
                rm=None,
                branch_idx=0,
                reranker=None,
                compiled_lm=None,
                force_reuse_cached_compilation=False,
                compiling=False,  # TODO: can probably be removed
                skip_logprobs=False,
                trace=[],
                release=0,
                bypass_assert=False,
                bypass_suggest=False,
                assert_failures=0,
                suggest_failures=0,
                langchain_history=[],
                experimental=False,
                backoff_time = 10
            )
            cls._instance.__append(config)

        return cls._instance

    @property
    def config(self):
        thread_id = threading.get_ident()
        if thread_id not in self.stack_by_thread:
            self.stack_by_thread[thread_id] = [self.main_stack[-1].copy()]
        return self.stack_by_thread[thread_id][-1]

    def __getattr__(self, name):
        if hasattr(self.config, name):
            return getattr(self.config, name)

        if name in self.config:
            return self.config[name]

        super().__getattr__(name)

    def __append(self, config):
        thread_id = threading.get_ident()
        if thread_id not in self.stack_by_thread:
            self.stack_by_thread[thread_id] = [self.main_stack[-1].copy()]
        self.stack_by_thread[thread_id].append(config)

    def __pop(self):
        thread_id = threading.get_ident()
        if thread_id in self.stack_by_thread:
            self.stack_by_thread[thread_id].pop()

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
