import platform
from argparse import ArgumentParser

import fsspec
import huggingface_hub
import pandas
import pyarrow

from datasets import __version__ as version
from datasets.commands import BaseDatasetsCLICommand


def info_command_factory(_):
    return EnvironmentCommand()


class EnvironmentCommand(BaseDatasetsCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("env", help="Print relevant system environment info.")
        download_parser.set_defaults(func=info_command_factory)

    def run(self):
        info = {
            "`datasets` version": version,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
            "`huggingface_hub` version": huggingface_hub.__version__,
            "PyArrow version": pyarrow.__version__,
            "Pandas version": pandas.__version__,
            "`fsspec` version": fsspec.__version__,
        }

        print("\nCopy-and-paste the text below in your GitHub issue.\n")
        print(self.format_dict(info))

        return info

    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
