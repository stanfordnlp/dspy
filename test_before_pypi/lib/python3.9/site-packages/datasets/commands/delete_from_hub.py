from argparse import ArgumentParser
from typing import Optional

from datasets.commands import BaseDatasetsCLICommand
from datasets.hub import delete_from_hub


def _command_factory(args):
    return DeleteFromHubCommand(
        args.dataset_id,
        args.config_name,
        args.token,
        args.revision,
    )


class DeleteFromHubCommand(BaseDatasetsCLICommand):
    @staticmethod
    def register_subcommand(parser):
        parser: ArgumentParser = parser.add_parser("delete_from_hub", help="Delete dataset config from the Hub")
        parser.add_argument(
            "dataset_id", help="source dataset ID, e.g. USERNAME/DATASET_NAME or ORGANIZATION/DATASET_NAME"
        )
        parser.add_argument("config_name", help="config name to delete")
        parser.add_argument("--token", help="access token to the Hugging Face Hub")
        parser.add_argument("--revision", help="source revision")
        parser.set_defaults(func=_command_factory)

    def __init__(
        self,
        dataset_id: str,
        config_name: str,
        token: Optional[str],
        revision: Optional[str],
    ):
        self._dataset_id = dataset_id
        self._config_name = config_name
        self._token = token
        self._revision = revision

    def run(self) -> None:
        _ = delete_from_hub(self._dataset_id, self._config_name, revision=self._revision, token=self._token)
