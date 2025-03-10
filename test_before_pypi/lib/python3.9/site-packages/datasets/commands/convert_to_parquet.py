from argparse import ArgumentParser
from typing import Optional

from datasets.commands import BaseDatasetsCLICommand
from datasets.hub import convert_to_parquet


def _command_factory(args):
    return ConvertToParquetCommand(
        args.dataset_id,
        args.token,
        args.revision,
        args.trust_remote_code,
    )


class ConvertToParquetCommand(BaseDatasetsCLICommand):
    @staticmethod
    def register_subcommand(parser):
        parser: ArgumentParser = parser.add_parser("convert_to_parquet", help="Convert dataset to Parquet")
        parser.add_argument(
            "dataset_id", help="source dataset ID, e.g. USERNAME/DATASET_NAME or ORGANIZATION/DATASET_NAME"
        )
        parser.add_argument("--token", help="access token to the Hugging Face Hub (defaults to logged-in user's one)")
        parser.add_argument("--revision", help="source revision")
        parser.add_argument(
            "--trust_remote_code", action="store_true", help="whether to trust the code execution of the load script"
        )
        parser.set_defaults(func=_command_factory)

    def __init__(
        self,
        dataset_id: str,
        token: Optional[str],
        revision: Optional[str],
        trust_remote_code: bool,
    ):
        self._dataset_id = dataset_id
        self._token = token
        self._revision = revision
        self._trust_remote_code = trust_remote_code

    def run(self) -> None:
        _ = convert_to_parquet(
            self._dataset_id, revision=self._revision, token=self._token, trust_remote_code=self._trust_remote_code
        )
