"""Optuna CLI module.
If you want to add a new command, you also need to update the constant `_COMMANDS`
"""

from __future__ import annotations

import argparse
from argparse import ArgumentParser
from argparse import Namespace
import datetime
from enum import Enum
import inspect
import json
import logging
import os
import sys
from typing import Any
import warnings

import sqlalchemy.exc
import yaml

import optuna
from optuna._imports import _LazyImport
from optuna.exceptions import CLIUsageError
from optuna.exceptions import ExperimentalWarning
from optuna.storages import BaseStorage
from optuna.storages import JournalFileStorage
from optuna.storages import JournalRedisStorage
from optuna.storages import JournalStorage
from optuna.storages import RDBStorage
from optuna.storages.journal import JournalFileBackend
from optuna.storages.journal import JournalRedisBackend
from optuna.trial import TrialState


_dataframe = _LazyImport("optuna.study._dataframe")

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _check_storage_url(storage_url: str | None) -> str:
    if storage_url is not None:
        return storage_url

    env_storage = os.environ.get("OPTUNA_STORAGE")
    if env_storage is not None:
        warnings.warn(
            "Specifying the storage url via 'OPTUNA_STORAGE' environment variable"
            " is an experimental feature. The interface can change in the future.",
            ExperimentalWarning,
        )
        return env_storage
    raise CLIUsageError("Storage URL is not specified.")


def _get_storage(storage_url: str | None, storage_class: str | None) -> BaseStorage:
    storage_url = _check_storage_url(storage_url)
    if storage_class:
        if storage_class == JournalRedisBackend.__name__:
            return JournalStorage(JournalRedisBackend(storage_url))
        if storage_class == JournalRedisStorage.__name__:
            return JournalStorage(JournalRedisStorage(storage_url))
        if storage_class == JournalFileBackend.__name__:
            return JournalStorage(JournalFileBackend(storage_url))
        if storage_class == JournalFileStorage.__name__:
            return JournalStorage(JournalFileStorage(storage_url))
        if storage_class == RDBStorage.__name__:
            return RDBStorage(storage_url)
        raise CLIUsageError("Unsupported storage class")

    if storage_url.startswith("redis"):
        return JournalStorage(JournalRedisBackend(storage_url))
    if os.path.isfile(storage_url):
        return JournalStorage(JournalFileBackend(storage_url))
    try:
        return RDBStorage(storage_url)
    except sqlalchemy.exc.ArgumentError:
        raise CLIUsageError("Failed to guess storage class from storage_url")


def _format_value(value: Any) -> Any:
    #  Format value that can be serialized to JSON or YAML.
    if value is None or isinstance(value, (int, float)):
        return value
    elif isinstance(value, datetime.datetime):
        return value.strftime(_DATETIME_FORMAT)
    elif isinstance(value, list):
        return list(_format_value(v) for v in value)
    elif isinstance(value, tuple):
        return tuple(_format_value(v) for v in value)
    elif isinstance(value, dict):
        return {_format_value(k): _format_value(v) for k, v in value.items()}
    else:
        return str(value)


def _convert_to_dict(
    records: list[dict[tuple[str, str], Any]], columns: list[tuple[str, str]], flatten: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    header = []
    ret = []
    if flatten:
        for column in columns:
            if column[1] != "":
                header.append(f"{column[0]}_{column[1]}")
            elif any(isinstance(record.get(column), (list, tuple)) for record in records):
                max_length = 0
                for record in records:
                    if column in record:
                        max_length = max(max_length, len(record[column]))
                for i in range(max_length):
                    header.append(f"{column[0]}_{i}")
            else:
                header.append(column[0])
        for record in records:
            row = {}
            for column in columns:
                if column not in record:
                    continue
                value = _format_value(record[column])
                if column[1] != "":
                    row[f"{column[0]}_{column[1]}"] = value
                elif any(isinstance(record.get(column), (list, tuple)) for record in records):
                    for i, v in enumerate(value):
                        row[f"{column[0]}_{i}"] = v
                else:
                    row[f"{column[0]}"] = value
            ret.append(row)
    else:
        for column in columns:
            if column[0] not in header:
                header.append(column[0])
        for record in records:
            attrs: dict[str, Any] = {column_name: {} for column_name in header}
            for column in columns:
                if column not in record:
                    continue
                value = _format_value(record[column])
                if isinstance(column[1], int):
                    # Reconstruct list of values. `_dataframe._create_records_and_aggregate_column`
                    # returns indices of list as the second key of column.
                    if attrs[column[0]] == {}:
                        attrs[column[0]] = []
                    attrs[column[0]] += [None] * max(column[1] + 1 - len(attrs[column[0]]), 0)
                    attrs[column[0]][column[1]] = value
                elif column[1] != "":
                    attrs[column[0]][column[1]] = value
                else:
                    attrs[column[0]] = value
            ret.append(attrs)

    return ret, header


class ValueType(Enum):
    NONE = 0
    NUMERIC = 1
    STRING = 2


class CellValue:
    def __init__(self, value: Any) -> None:
        self.value = value
        if value is None:
            self.value_type = ValueType.NONE
        elif isinstance(value, (int, float)):
            self.value_type = ValueType.NUMERIC
        else:
            self.value_type = ValueType.STRING

    def __str__(self) -> str:
        if isinstance(self.value, datetime.datetime):
            return self.value.strftime(_DATETIME_FORMAT)
        else:
            return str(self.value)

    def width(self) -> int:
        return len(str(self.value))

    def get_string(self, value_type: ValueType, width: int) -> str:
        value = str(self.value)
        if self.value is None:
            return " " * width
        elif value_type == ValueType.NUMERIC:
            return f"{value:>{width}}"
        else:
            return f"{value:<{width}}"


def _dump_value(records: list[dict[str, Any]], header: list[str]) -> str:
    values = []
    for record in records:
        row = []
        for column_name in header:
            # Below follows the table formatting convention where record[column_name] is treated as
            # an empty string if record[column_name] is None. e.g., {"a": None} is replaced with
            # {"a": ""}
            row.append(str(record[column_name]) if record.get(column_name) is not None else "")
        values.append(" ".join(row))
    return "\n".join(values)


def _dump_table(records: list[dict[str, Any]], header: list[str]) -> str:
    rows = []
    for record in records:
        row = []
        for column_name in header:
            row.append(CellValue(record.get(column_name)))
        rows.append(row)

    separator = "+"
    header_string = "|"
    rows_string = ["|" for _ in rows]
    for column in range(len(header)):
        value_types = [row[column].value_type for row in rows]
        value_type = ValueType.NUMERIC
        for t in value_types:
            if t == ValueType.STRING:
                value_type = ValueType.STRING
        if len(rows) == 0:
            max_width = len(header[column])
        else:
            max_width = max(len(header[column]), max(row[column].width() for row in rows))
        separator += "-" * (max_width + 2) + "+"
        if value_type == ValueType.NUMERIC:
            header_string += f" {header[column]:>{max_width}} |"
        else:
            header_string += f" {header[column]:<{max_width}} |"
        for i, row in enumerate(rows):
            rows_string[i] += " " + row[column].get_string(value_type, max_width) + " |"

    ret = ""
    ret += separator + "\n"
    ret += header_string + "\n"
    ret += separator + "\n"
    for row_string in rows_string:
        ret += row_string + "\n"
    ret += separator + "\n"

    return ret


def _format_output(
    records: list[dict[tuple[str, str], Any]] | dict[tuple[str, str], Any],
    columns: list[tuple[str, str]],
    output_format: str,
    flatten: bool,
) -> str:
    if isinstance(records, list):
        values, header = _convert_to_dict(records, columns, flatten)
    else:
        values, header = _convert_to_dict([records], columns, flatten)

    if output_format == "value":
        return _dump_value(values, header).strip()
    elif output_format == "table":
        return _dump_table(values, header).strip()
    elif output_format == "json":
        if isinstance(records, list):
            return json.dumps(values).strip()
        else:
            return json.dumps(values[0]).strip()
    elif output_format == "yaml":
        if isinstance(records, list):
            return yaml.safe_dump(values).strip()
        else:
            return yaml.safe_dump(values[0]).strip()
    else:
        raise CLIUsageError(f"Optuna CLI does not supported the {output_format} format.")


class _BaseCommand:
    """Base class for commands.

    Note that command classes are not intended to be called by library users.
    They are exclusively used within this file to manage Optuna CLI commands.
    """

    def __init__(self) -> None:
        self.logger = optuna.logging.get_logger(__name__)

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add arguments required for each command.

        Args:
            parser:
                `ArgumentParser` object to add arguments
        """
        pass

    def take_action(self, parsed_args: Namespace) -> int:
        """Define action if the command is called.

        Args:
            parsed_args:
                `Namespace` object including arguments specified by user.

        Returns:
            Running status of the action.
            0 if this method finishes normally, otherwise 1.
        """

        raise NotImplementedError


class _CreateStudy(_BaseCommand):
    """Create a new study."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--study-name",
            default=None,
            help="A human-readable name of a study to distinguish it from others.",
        )
        parser.add_argument(
            "--direction",
            default=None,
            type=str,
            choices=("minimize", "maximize"),
            help="Set direction of optimization to a new study. Set 'minimize' "
            "for minimization and 'maximize' for maximization.",
        )
        parser.add_argument(
            "--skip-if-exists",
            default=False,
            action="store_true",
            help="If specified, the creation of the study is skipped "
            "without any error when the study name is duplicated.",
        )
        parser.add_argument(
            "--directions",
            type=str,
            default=None,
            choices=("minimize", "maximize"),
            help="Set directions of optimization to a new study."
            " Put whitespace between directions. Each direction should be"
            ' either "minimize" or "maximize".',
            nargs="+",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        storage = _get_storage(parsed_args.storage, parsed_args.storage_class)
        study_name = optuna.create_study(
            storage=storage,
            study_name=parsed_args.study_name,
            direction=parsed_args.direction,
            directions=parsed_args.directions,
            load_if_exists=parsed_args.skip_if_exists,
        ).study_name
        print(study_name)
        return 0


class _DeleteStudy(_BaseCommand):
    """Delete a specified study."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--study-name", default=None, help="The name of the study to delete.")

    def take_action(self, parsed_args: Namespace) -> int:
        storage = _get_storage(parsed_args.storage, parsed_args.storage_class)
        study_id = storage.get_study_id_from_name(parsed_args.study_name)
        storage.delete_study(study_id)
        return 0


class _StudySetUserAttribute(_BaseCommand):
    """Set a user attribute to a study."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--study-name",
            required=True,
            help="The name of the study to set the user attribute to.",
        )
        parser.add_argument("--key", "-k", required=True, help="Key of the user attribute.")
        parser.add_argument("--value", required=True, help="Value to be set.")

    def take_action(self, parsed_args: Namespace) -> int:
        storage = _get_storage(parsed_args.storage, parsed_args.storage_class)

        study = optuna.load_study(storage=storage, study_name=parsed_args.study_name)

        study.set_user_attr(parsed_args.key, parsed_args.value)

        self.logger.info("Attribute successfully written.")
        return 0


class _StudyNames(_BaseCommand):
    """Get all study names stored in a specified storage"""

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=("value", "json", "table", "yaml"),
            default="value",
            help="Output format.",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        storage = _get_storage(parsed_args.storage, parsed_args.storage_class)
        all_study_names = optuna.get_all_study_names(storage)
        records = []
        record_key = ("name", "")
        for study_name in all_study_names:
            records.append({record_key: study_name})
        print(_format_output(records, [record_key], parsed_args.format, flatten=False))
        return 0


class _Studies(_BaseCommand):
    """Show a list of studies."""

    _study_list_header = [
        ("name", ""),
        ("direction", ""),
        ("n_trials", ""),
        ("datetime_start", ""),
    ]

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=("value", "json", "table", "yaml"),
            default="table",
            help="Output format.",
        )
        parser.add_argument(
            "--flatten",
            default=False,
            action="store_true",
            help="Flatten nested columns such as directions.",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        storage = _get_storage(parsed_args.storage, parsed_args.storage_class)
        summaries = optuna.get_all_study_summaries(storage, include_best_trial=False)

        records = []
        for s in summaries:
            start = (
                s.datetime_start.strftime(_DATETIME_FORMAT)
                if s.datetime_start is not None
                else None
            )
            record: dict[tuple[str, str], Any] = {}
            record[("name", "")] = s.study_name
            record[("direction", "")] = tuple(d.name for d in s.directions)
            record[("n_trials", "")] = s.n_trials
            record[("datetime_start", "")] = start
            record[("user_attrs", "")] = s.user_attrs
            records.append(record)

        if any(r[("user_attrs", "")] != {} for r in records):
            self._study_list_header.append(("user_attrs", ""))
        print(
            _format_output(
                records, self._study_list_header, parsed_args.format, parsed_args.flatten
            )
        )
        return 0


class _Trials(_BaseCommand):
    """Show a list of trials."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--study-name",
            type=str,
            required=True,
            help="The name of the study which includes trials.",
        )
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=("value", "json", "table", "yaml"),
            default="table",
            help="Output format.",
        )
        parser.add_argument(
            "--flatten",
            default=False,
            action="store_true",
            help="Flatten nested columns such as params and user_attrs.",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        warnings.warn(
            "'trials' is an experimental CLI command. The interface can change in the future.",
            ExperimentalWarning,
        )

        storage = _get_storage(parsed_args.storage, parsed_args.storage_class)
        study = optuna.load_study(storage=storage, study_name=parsed_args.study_name)
        attrs = (
            "number",
            "value" if not study._is_multi_objective() else "values",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )

        records, columns = _dataframe._create_records_and_aggregate_column(study, attrs)
        print(_format_output(records, columns, parsed_args.format, parsed_args.flatten))

        return 0


class _BestTrial(_BaseCommand):
    """Show the best trial."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--study-name",
            type=str,
            required=True,
            help="The name of the study to get the best trial.",
        )
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=("value", "json", "table", "yaml"),
            default="table",
            help="Output format.",
        )
        parser.add_argument(
            "--flatten",
            default=False,
            action="store_true",
            help="Flatten nested columns such as params and user_attrs.",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        warnings.warn(
            "'best-trial' is an experimental CLI command. The interface can change in the future.",
            ExperimentalWarning,
        )

        storage = _get_storage(parsed_args.storage, parsed_args.storage_class)
        study = optuna.load_study(storage=storage, study_name=parsed_args.study_name)
        attrs = (
            "number",
            "value" if not study._is_multi_objective() else "values",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )

        records, columns = _dataframe._create_records_and_aggregate_column(study, attrs)
        print(
            _format_output(
                records[study.best_trial.number], columns, parsed_args.format, parsed_args.flatten
            )
        )
        return 0


class _BestTrials(_BaseCommand):
    """Show a list of trials located at the Pareto front."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--study-name",
            type=str,
            required=True,
            help="The name of the study to get the best trials (trials at the Pareto front).",
        )
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=("value", "json", "table", "yaml"),
            default="table",
            help="Output format.",
        )
        parser.add_argument(
            "--flatten",
            default=False,
            action="store_true",
            help="Flatten nested columns such as params and user_attrs.",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        warnings.warn(
            "'best-trials' is an experimental CLI command. The interface can change in the "
            "future.",
            ExperimentalWarning,
        )

        storage = _get_storage(parsed_args.storage, parsed_args.storage_class)
        study = optuna.load_study(storage=storage, study_name=parsed_args.study_name)
        best_trials = [trial.number for trial in study.best_trials]
        attrs = (
            "number",
            "value" if not study._is_multi_objective() else "values",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )

        records, columns = _dataframe._create_records_and_aggregate_column(study, attrs)
        best_records = list(filter(lambda record: record[("number", "")] in best_trials, records))
        print(_format_output(best_records, columns, parsed_args.format, parsed_args.flatten))
        return 0


class _StorageUpgrade(_BaseCommand):
    """Upgrade the schema of an RDB storage."""

    def take_action(self, parsed_args: Namespace) -> int:
        storage_url = _check_storage_url(parsed_args.storage)
        try:
            storage = RDBStorage(
                storage_url, skip_compatibility_check=True, skip_table_creation=True
            )
        except sqlalchemy.exc.ArgumentError:
            self.logger.error("Invalid RDBStorage URL.")
            return 1
        current_version = storage.get_current_version()
        head_version = storage.get_head_version()
        known_versions = storage.get_all_versions()
        if current_version == head_version:
            self.logger.info("This storage is up-to-date.")
        elif current_version in known_versions:
            self.logger.info("Upgrading the storage schema to the latest version.")
            storage.upgrade()
            self.logger.info("Completed to upgrade the storage.")
        else:
            warnings.warn(
                "Your optuna version seems outdated against the storage version. "
                "Please try updating optuna to the latest version by "
                "`$ pip install -U optuna`."
            )
        return 0


class _Ask(_BaseCommand):
    """Create a new trial and suggest parameters."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--study-name", type=str, help="Name of study.")
        parser.add_argument("--sampler", type=str, help="Class name of sampler object to create.")
        parser.add_argument(
            "--sampler-kwargs",
            type=str,
            help="Sampler object initialization keyword arguments as JSON.",
        )
        parser.add_argument(
            "--search-space",
            type=str,
            help=(
                "Search space as JSON. Keys are names and values are outputs from "
                ":func:`~optuna.distributions.distribution_to_json`."
            ),
        )
        parser.add_argument(
            "-f",
            "--format",
            type=str,
            choices=("value", "json", "table", "yaml"),
            default="json",
            help="Output format.",
        )
        parser.add_argument(
            "--flatten",
            default=False,
            action="store_true",
            help="Flatten nested columns such as params.",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        warnings.warn(
            "'ask' is an experimental CLI command. The interface can change in the future.",
            ExperimentalWarning,
        )

        storage = _get_storage(parsed_args.storage, parsed_args.storage_class)

        create_study_kwargs = {
            "storage": storage,
            "study_name": parsed_args.study_name,
            "load_if_exists": True,
        }

        if parsed_args.sampler is not None:
            if parsed_args.sampler_kwargs is not None:
                sampler_kwargs = json.loads(parsed_args.sampler_kwargs)
            else:
                sampler_kwargs = {}
            sampler_cls = getattr(optuna.samplers, parsed_args.sampler)
            sampler = sampler_cls(**sampler_kwargs)
            create_study_kwargs["sampler"] = sampler
        else:
            if parsed_args.sampler_kwargs is not None:
                raise ValueError(
                    "`--sampler_kwargs` is set without `--sampler`. Please specify `--sampler` as"
                    " well or omit `--sampler-kwargs`."
                )

        if parsed_args.search_space is not None:
            # The search space is expected to be a JSON serialized string, e.g.
            # '{"x": {"name": "FloatDistribution", "attributes": {"low": 0.0, "high": 1.0}},
            #   "y": ...}'.
            search_space = {
                name: optuna.distributions.json_to_distribution(json.dumps(dist))
                for name, dist in json.loads(parsed_args.search_space).items()
            }
        else:
            search_space = {}

        try:
            study = optuna.load_study(
                study_name=create_study_kwargs["study_name"],
                storage=create_study_kwargs["storage"],
                sampler=create_study_kwargs.get("sampler"),
            )

        except KeyError:
            raise KeyError(
                "Implicit study creation within the 'ask' command was dropped in Optuna v4.0.0. "
                "Please use the 'create-study' command beforehand."
            )
        trial = study.ask(fixed_distributions=search_space)

        self.logger.info(f"Asked trial {trial.number} with parameters {trial.params}.")

        record: dict[tuple[str, str], Any] = {("number", ""): trial.number}
        columns = [("number", "")]

        if len(trial.params) == 0 and not parsed_args.flatten:
            record[("params", "")] = {}
            columns.append(("params", ""))
        else:
            for param_name, param_value in trial.params.items():
                record[("params", param_name)] = param_value
                columns.append(("params", param_name))

        print(_format_output(record, columns, parsed_args.format, parsed_args.flatten))
        return 0


class _Tell(_BaseCommand):
    """Finish a trial, which was created by the ask command."""

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--study-name", type=str, help="Name of study.")
        parser.add_argument("--trial-number", type=int, help="Trial number.")
        parser.add_argument("--values", type=float, nargs="+", help="Objective values.")
        parser.add_argument(
            "--state",
            type=str,
            help="Trial state.",
            choices=("complete", "pruned", "fail"),
        )
        parser.add_argument(
            "--skip-if-finished",
            default=False,
            action="store_true",
            help="If specified, tell is skipped without any error when the trial is already "
            "finished.",
        )

    def take_action(self, parsed_args: Namespace) -> int:
        warnings.warn(
            "'tell' is an experimental CLI command. The interface can change in the future.",
            ExperimentalWarning,
        )

        storage = _get_storage(parsed_args.storage, parsed_args.storage_class)

        study = optuna.load_study(
            storage=storage,
            study_name=parsed_args.study_name,
        )

        if parsed_args.state is not None:
            state: TrialState | None = TrialState[parsed_args.state.upper()]
        else:
            state = None

        trial_number = parsed_args.trial_number
        values = parsed_args.values

        study.tell(
            trial=trial_number,
            values=values,
            state=state,
            skip_if_finished=parsed_args.skip_if_finished,
        )

        self.logger.info(f"Told trial {trial_number} with values {values} and state {state}.")

        return 0


_COMMANDS: dict[str, type[_BaseCommand]] = {
    "create-study": _CreateStudy,
    "delete-study": _DeleteStudy,
    "study set-user-attr": _StudySetUserAttribute,
    "study-names": _StudyNames,
    "studies": _Studies,
    "trials": _Trials,
    "best-trial": _BestTrial,
    "best-trials": _BestTrials,
    "storage upgrade": _StorageUpgrade,
    "ask": _Ask,
    "tell": _Tell,
}


def _parse_storage_class_without_suggesting_deprecated_choices(value: str) -> str:
    choices = [
        RDBStorage.__name__,
        JournalFileBackend.__name__,
        JournalRedisBackend.__name__,
    ]
    deprecated_choices = [
        JournalFileStorage.__name__,
        JournalRedisStorage.__name__,
    ]
    if value in choices + deprecated_choices:
        return value
    raise argparse.ArgumentTypeError(
        f"Invalid choice: {value}  (choose from {str(choices)[1:-1]})"
    )


def _add_common_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--storage",
        default=None,
        help=(
            "DB URL. (e.g. sqlite:///example.db) "
            "Also can be specified via OPTUNA_STORAGE environment variable."
        ),
    )
    parser.add_argument(
        "--storage-class",
        help="Storage class hint (e.g. JournalFileBackend)",
        default=None,
        type=_parse_storage_class_without_suggesting_deprecated_choices,
    )
    verbose_group = parser.add_mutually_exclusive_group()
    verbose_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        dest="verbose_level",
        default=1,
        help="Increase verbosity of output. Can be repeated.",
    )
    verbose_group.add_argument(
        "-q",
        "--quiet",
        action="store_const",
        dest="verbose_level",
        const=0,
        help="Suppress output except warnings and errors.",
    )
    parser.add_argument(
        "--log-file",
        action="store",
        default=None,
        help="Specify a file to log output. Disabled by default.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Show tracebacks on errors.",
    )
    return parser


def _add_commands(
    main_parser: ArgumentParser, parent_parser: ArgumentParser
) -> dict[str, ArgumentParser]:
    subparsers = main_parser.add_subparsers()
    command_name_to_subparser = {}

    for command_name, command_type in _COMMANDS.items():
        command = command_type()
        subparser = subparsers.add_parser(
            command_name, parents=[parent_parser], help=inspect.getdoc(command_type)
        )
        command.add_arguments(subparser)
        subparser.set_defaults(handler=command.take_action)
        command_name_to_subparser[command_name] = subparser

    def _print_help(args: Namespace) -> None:
        main_parser.print_help()

    subparsers.add_parser("help", help="Show help message and exit.").set_defaults(
        handler=_print_help
    )
    return command_name_to_subparser


def _get_parser(description: str = "") -> tuple[ArgumentParser, dict[str, ArgumentParser]]:
    # Use `parent_parser` is necessary to avoid namespace conflict for -h/--help
    # between `main_parser` and `subparser`.
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = _add_common_arguments(parent_parser)

    main_parser = ArgumentParser(description=description, parents=[parent_parser])
    main_parser.add_argument(
        "--version", action="version", version="{0} {1}".format("optuna", optuna.__version__)
    )
    command_name_to_subparser = _add_commands(main_parser, parent_parser)
    return main_parser, command_name_to_subparser


def _preprocess_argv(argv: list[str]) -> list[str]:
    # Some preprocess is necessary for argv because some subcommand includes space
    # (e.g. optuna storage upgrade).
    argv = argv[1:] if len(argv) > 1 else ["help"]

    for i in range(len(argv)):
        for j in range(i, i + 2):  # Commands consist of one or two words.
            command_candidate = " ".join(argv[i : j + 1])
            if command_candidate in _COMMANDS:
                options = argv[:i] + argv[j + 1 :]
                return [command_candidate] + options

    # No subcommand is found.
    return argv


def _set_verbosity(args: Namespace) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stderr)

    logging_level = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }.get(args.verbose_level, logging.DEBUG)

    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(optuna.logging.create_default_formatter())
    root_logger.addHandler(stream_handler)

    optuna.logging.set_verbosity(logging_level)


def _set_log_file(args: Namespace) -> None:
    if args.log_file is None:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        filename=args.log_file,
    )
    file_handler.setFormatter(optuna.logging.create_default_formatter())
    root_logger.addHandler(file_handler)


def main() -> int:
    main_parser, command_name_to_subparser = _get_parser()

    argv = sys.argv
    preprocessed_argv = _preprocess_argv(argv)
    args = main_parser.parse_args(preprocessed_argv)

    _set_verbosity(args)
    _set_log_file(args)

    logger = logging.getLogger("optuna")
    try:
        return args.handler(args)
    except CLIUsageError as e:
        if args.debug:
            logger.exception(e)
        else:
            logger.error(e)
            # This code is required to show help for each subcommand.
            # NOTE: the first element of `preprocessed_argv` is command name.
            command_name_to_subparser[preprocessed_argv[0]].print_help()
        return 1
    except AttributeError:
        # Exception for the case -v/--verbose/-q/--quiet/--log-file/--debug
        # without any subcommand.
        argv_str = " ".join(argv[1:])
        logger.error(f"'{argv_str}' is not an optuna command. see 'optuna --help'")
        main_parser.print_help()
        return 1
