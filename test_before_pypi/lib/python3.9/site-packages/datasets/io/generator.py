from typing import Callable, Optional

from .. import Features, NamedSplit, Split
from ..packaged_modules.generator.generator import Generator
from .abc import AbstractDatasetInputStream


class GeneratorDatasetInputStream(AbstractDatasetInputStream):
    def __init__(
        self,
        generator: Callable,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        streaming: bool = False,
        gen_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        split: NamedSplit = Split.TRAIN,
        **kwargs,
    ):
        super().__init__(
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            streaming=streaming,
            num_proc=num_proc,
            **kwargs,
        )
        self.builder = Generator(
            cache_dir=cache_dir,
            features=features,
            generator=generator,
            gen_kwargs=gen_kwargs,
            split=split,
            **kwargs,
        )

    def read(self):
        # Build iterable dataset
        if self.streaming:
            dataset = self.builder.as_streaming_dataset(split=self.builder.config.split)
        # Build regular (map-style) dataset
        else:
            download_config = None
            download_mode = None
            verification_mode = None
            base_path = None

            self.builder.download_and_prepare(
                download_config=download_config,
                download_mode=download_mode,
                verification_mode=verification_mode,
                base_path=base_path,
                num_proc=self.num_proc,
            )
            dataset = self.builder.as_dataset(
                split=self.builder.config.split, verification_mode=verification_mode, in_memory=self.keep_in_memory
            )
        return dataset
