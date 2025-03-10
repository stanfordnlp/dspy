# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains commands to authenticate to the Hugging Face Hub and interact with your repositories.

Usage:
    # login and save token locally.
    huggingface-cli login --token=hf_*** --add-to-git-credential

    # switch between tokens
    huggingface-cli auth switch

    # list all tokens
    huggingface-cli auth list

    # logout from a specific token, if no token-name is provided, all tokens will be deleted from your machine.
    huggingface-cli logout --token-name=your_token_name

    # find out which huggingface.co account you are logged in as
    huggingface-cli whoami

    # create a new dataset repo on the Hub
    huggingface-cli repo create mydataset --type=dataset

"""

import subprocess
from argparse import _SubParsersAction
from typing import List, Optional

from requests.exceptions import HTTPError

from huggingface_hub.commands import BaseHuggingfaceCLICommand
from huggingface_hub.constants import ENDPOINT, REPO_TYPES, REPO_TYPES_URL_PREFIXES, SPACES_SDK_TYPES
from huggingface_hub.hf_api import HfApi

from .._login import (  # noqa: F401 # for backward compatibility  # noqa: F401 # for backward compatibility
    NOTEBOOK_LOGIN_PASSWORD_HTML,
    NOTEBOOK_LOGIN_TOKEN_HTML_END,
    NOTEBOOK_LOGIN_TOKEN_HTML_START,
    auth_list,
    auth_switch,
    login,
    logout,
    notebook_login,
)
from ..utils import get_stored_tokens, get_token, logging
from ._cli_utils import ANSI


logger = logging.get_logger(__name__)

try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice

    _inquirer_py_available = True
except ImportError:
    _inquirer_py_available = False


class UserCommands(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        login_parser = parser.add_parser("login", help="Log in using a token from huggingface.co/settings/tokens")
        login_parser.add_argument(
            "--token",
            type=str,
            help="Token generated from https://huggingface.co/settings/tokens",
        )
        login_parser.add_argument(
            "--add-to-git-credential",
            action="store_true",
            help="Optional: Save token to git credential helper.",
        )
        login_parser.set_defaults(func=lambda args: LoginCommand(args))
        whoami_parser = parser.add_parser("whoami", help="Find out which huggingface.co account you are logged in as.")
        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))

        logout_parser = parser.add_parser("logout", help="Log out")
        logout_parser.add_argument(
            "--token-name",
            type=str,
            help="Optional: Name of the access token to log out from.",
        )
        logout_parser.set_defaults(func=lambda args: LogoutCommand(args))

        auth_parser = parser.add_parser("auth", help="Other authentication related commands")
        auth_subparsers = auth_parser.add_subparsers(help="Authentication subcommands")
        auth_switch_parser = auth_subparsers.add_parser("switch", help="Switch between access tokens")
        auth_switch_parser.add_argument(
            "--token-name",
            type=str,
            help="Optional: Name of the access token to switch to.",
        )
        auth_switch_parser.add_argument(
            "--add-to-git-credential",
            action="store_true",
            help="Optional: Save token to git credential helper.",
        )
        auth_switch_parser.set_defaults(func=lambda args: AuthSwitchCommand(args))
        auth_list_parser = auth_subparsers.add_parser("list", help="List all stored access tokens")
        auth_list_parser.set_defaults(func=lambda args: AuthListCommand(args))
        # new system: git-based repo system
        repo_parser = parser.add_parser("repo", help="{create} Commands to interact with your huggingface.co repos.")
        repo_subparsers = repo_parser.add_subparsers(help="huggingface.co repos related commands")
        repo_create_parser = repo_subparsers.add_parser("create", help="Create a new repo on huggingface.co")
        repo_create_parser.add_argument(
            "name",
            type=str,
            help="Name for your repo. Will be namespaced under your username to build the repo id.",
        )
        repo_create_parser.add_argument(
            "--type",
            type=str,
            help='Optional: repo_type: set to "dataset" or "space" if creating a dataset or space, default is model.',
        )
        repo_create_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        repo_create_parser.add_argument(
            "--space_sdk",
            type=str,
            help='Optional: Hugging Face Spaces SDK type. Required when --type is set to "space".',
            choices=SPACES_SDK_TYPES,
        )
        repo_create_parser.add_argument(
            "-y",
            "--yes",
            action="store_true",
            help="Optional: answer Yes to the prompt",
        )
        repo_create_parser.set_defaults(func=lambda args: RepoCreateCommand(args))


class BaseUserCommand:
    def __init__(self, args):
        self.args = args
        self._api = HfApi()


class LoginCommand(BaseUserCommand):
    def run(self):
        logging.set_verbosity_info()
        login(
            token=self.args.token,
            add_to_git_credential=self.args.add_to_git_credential,
        )


class LogoutCommand(BaseUserCommand):
    def run(self):
        logging.set_verbosity_info()
        logout(token_name=self.args.token_name)


class AuthSwitchCommand(BaseUserCommand):
    def run(self):
        logging.set_verbosity_info()
        token_name = self.args.token_name
        if token_name is None:
            token_name = self._select_token_name()

        if token_name is None:
            print("No token name provided. Aborting.")
            exit()
        auth_switch(token_name, add_to_git_credential=self.args.add_to_git_credential)

    def _select_token_name(self) -> Optional[str]:
        token_names = list(get_stored_tokens().keys())

        if not token_names:
            logger.error("No stored tokens found. Please login first.")
            return None

        if _inquirer_py_available:
            return self._select_token_name_tui(token_names)
        # if inquirer is not available, use a simpler terminal UI
        print("Available stored tokens:")
        for i, token_name in enumerate(token_names, 1):
            print(f"{i}. {token_name}")
        while True:
            try:
                choice = input("Enter the number of the token to switch to (or 'q' to quit): ")
                if choice.lower() == "q":
                    return None
                index = int(choice) - 1
                if 0 <= index < len(token_names):
                    return token_names[index]
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")

    def _select_token_name_tui(self, token_names: List[str]) -> Optional[str]:
        choices = [Choice(token_name, name=token_name) for token_name in token_names]
        try:
            return inquirer.select(
                message="Select a token to switch to:",
                choices=choices,
                default=None,
            ).execute()
        except KeyboardInterrupt:
            logger.info("Token selection cancelled.")
            return None


class AuthListCommand(BaseUserCommand):
    def run(self):
        logging.set_verbosity_info()
        auth_list()


class WhoamiCommand(BaseUserCommand):
    def run(self):
        token = get_token()
        if token is None:
            print("Not logged in")
            exit()
        try:
            info = self._api.whoami(token)
            print(info["name"])
            orgs = [org["name"] for org in info["orgs"]]
            if orgs:
                print(ANSI.bold("orgs: "), ",".join(orgs))

            if ENDPOINT != "https://huggingface.co":
                print(f"Authenticated through private endpoint: {ENDPOINT}")
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)


class RepoCreateCommand(BaseUserCommand):
    def run(self):
        token = get_token()
        if token is None:
            print("Not logged in")
            exit(1)
        try:
            stdout = subprocess.check_output(["git", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print("Looks like you do not have git installed, please install.")

        try:
            stdout = subprocess.check_output(["git-lfs", "--version"]).decode("utf-8")
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print(
                ANSI.red(
                    "Looks like you do not have git-lfs installed, please install."
                    " You can install from https://git-lfs.github.com/."
                    " Then run `git lfs install` (you only have to do this once)."
                )
            )
        print("")

        user = self._api.whoami(token)["name"]
        namespace = self.args.organization if self.args.organization is not None else user

        repo_id = f"{namespace}/{self.args.name}"

        if self.args.type not in REPO_TYPES:
            print("Invalid repo --type")
            exit(1)

        if self.args.type in REPO_TYPES_URL_PREFIXES:
            prefixed_repo_id = REPO_TYPES_URL_PREFIXES[self.args.type] + repo_id
        else:
            prefixed_repo_id = repo_id

        print(f"You are about to create {ANSI.bold(prefixed_repo_id)}")

        if not self.args.yes:
            choice = input("Proceed? [Y/n] ").lower()
            if not (choice == "" or choice == "y" or choice == "yes"):
                print("Abort")
                exit()
        try:
            url = self._api.create_repo(
                repo_id=repo_id,
                token=token,
                repo_type=self.args.type,
                space_sdk=self.args.space_sdk,
            )
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print("\nYour repo now lives at:")
        print(f"  {ANSI.bold(url)}")
        print("\nYou can clone it locally with the command below, and commit/push as usual.")
        print(f"\n  git clone {url}")
        print("")
