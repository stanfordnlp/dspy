import os
import re
import shutil
from argparse import ArgumentParser, Namespace

from datasets.commands import BaseDatasetsCLICommand
from datasets.utils.logging import get_logger


HIGHLIGHT_MESSAGE_PRE = """<<<<<<< This should probably be modified because it mentions: """

HIGHLIGHT_MESSAGE_POST = """=======
>>>>>>>
"""

TO_HIGHLIGHT = [
    "TextEncoderConfig",
    "ByteTextEncoder",
    "SubwordTextEncoder",
    "encoder_config",
    "maybe_build_from_corpus",
    "manual_dir",
]

TO_CONVERT = [
    # (pattern, replacement)
    # Order is important here for some replacements
    (r"tfds\.core", r"datasets"),
    (r"tf\.io\.gfile\.GFile", r"open"),
    (r"tf\.([\w\d]+)", r"datasets.Value('\1')"),
    (r"tfds\.features\.Text\(\)", r"datasets.Value('string')"),
    (r"tfds\.features\.Text\(", r"datasets.Value('string'),"),
    (r"features\s*=\s*tfds.features.FeaturesDict\(", r"features=datasets.Features("),
    (r"tfds\.features\.FeaturesDict\(", r"dict("),
    (r"The TensorFlow Datasets Authors", r"The TensorFlow Datasets Authors and the HuggingFace Datasets Authors"),
    (r"tfds\.", r"datasets."),
    (r"dl_manager\.manual_dir", r"self.config.data_dir"),
    (r"self\.builder_config", r"self.config"),
]


def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.

    Returns: ConvertCommand
    """
    return ConvertCommand(args.tfds_path, args.datasets_directory)


class ConvertCommand(BaseDatasetsCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the datasets-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        train_parser = parser.add_parser(
            "convert",
            help="Convert a TensorFlow Datasets dataset to a HuggingFace Datasets dataset.",
        )
        train_parser.add_argument(
            "--tfds_path",
            type=str,
            required=True,
            help="Path to a TensorFlow Datasets folder to convert or a single tfds file to convert.",
        )
        train_parser.add_argument(
            "--datasets_directory", type=str, required=True, help="Path to the HuggingFace Datasets folder."
        )
        train_parser.set_defaults(func=convert_command_factory)

    def __init__(self, tfds_path: str, datasets_directory: str, *args):
        self._logger = get_logger("datasets-cli/converting")

        self._tfds_path = tfds_path
        self._datasets_directory = datasets_directory

    def run(self):
        if os.path.isdir(self._tfds_path):
            abs_tfds_path = os.path.abspath(self._tfds_path)
        elif os.path.isfile(self._tfds_path):
            abs_tfds_path = os.path.dirname(self._tfds_path)
        else:
            raise ValueError("--tfds_path is neither a directory nor a file. Please check path.")

        abs_datasets_path = os.path.abspath(self._datasets_directory)

        self._logger.info(f"Converting datasets from {abs_tfds_path} to {abs_datasets_path}")

        utils_files = []
        with_manual_update = []
        imports_to_builder_map = {}

        if os.path.isdir(self._tfds_path):
            file_names = os.listdir(abs_tfds_path)
        else:
            file_names = [os.path.basename(self._tfds_path)]

        for f_name in file_names:
            self._logger.info(f"Looking at file {f_name}")
            input_file = os.path.join(abs_tfds_path, f_name)
            output_file = os.path.join(abs_datasets_path, f_name)

            if not os.path.isfile(input_file) or "__init__" in f_name or "_test" in f_name or ".py" not in f_name:
                self._logger.info("Skipping file")
                continue

            with open(input_file, encoding="utf-8") as f:
                lines = f.readlines()

            out_lines = []
            is_builder = False
            needs_manual_update = False
            tfds_imports = []
            for line in lines:
                out_line = line

                # Convert imports
                if "import tensorflow.compat.v2 as tf" in out_line:
                    continue
                elif "@tfds.core" in out_line:
                    continue
                elif "builder=self" in out_line:
                    continue
                elif "import tensorflow_datasets.public_api as tfds" in out_line:
                    out_line = "import datasets\n"
                elif "import tensorflow" in out_line:
                    # order is important here
                    out_line = ""
                    continue
                elif "from absl import logging" in out_line:
                    out_line = "from datasets import logging\n"
                elif "getLogger" in out_line:
                    out_line = out_line.replace("getLogger", "get_logger")
                elif any(expression in out_line for expression in TO_HIGHLIGHT):
                    needs_manual_update = True
                    to_remove = list(filter(lambda e: e in out_line, TO_HIGHLIGHT))
                    out_lines.append(HIGHLIGHT_MESSAGE_PRE + str(to_remove) + "\n")
                    out_lines.append(out_line)
                    out_lines.append(HIGHLIGHT_MESSAGE_POST)
                    continue
                else:
                    for pattern, replacement in TO_CONVERT:
                        out_line = re.sub(pattern, replacement, out_line)

                # Take care of saving utilities (to later move them together with main script)
                if "tensorflow_datasets" in out_line:
                    match = re.match(r"from\stensorflow_datasets.*import\s([^\.\r\n]+)", out_line)
                    tfds_imports.extend(imp.strip() for imp in match.group(1).split(","))
                    out_line = "from . import " + match.group(1)

                # Check we have not forget anything
                if "tf." in out_line or "tfds." in out_line or "tensorflow_datasets" in out_line:
                    raise ValueError(f"Error converting {out_line.strip()}")

                if "GeneratorBasedBuilder" in out_line or "BeamBasedBuilder" in out_line:
                    is_builder = True
                out_lines.append(out_line)

            if is_builder or "wmt" in f_name:
                # We create a new directory for each dataset
                dir_name = f_name.replace(".py", "")
                output_dir = os.path.join(abs_datasets_path, dir_name)
                output_file = os.path.join(output_dir, f_name)
                os.makedirs(output_dir, exist_ok=True)
                self._logger.info(f"Adding directory {output_dir}")
                imports_to_builder_map.update({imp: output_dir for imp in tfds_imports})
            else:
                # Utilities will be moved at the end
                utils_files.append(output_file)

            if needs_manual_update:
                with_manual_update.append(output_file)

            with open(output_file, "w", encoding="utf-8") as f:
                f.writelines(out_lines)
            self._logger.info(f"Converted in {output_file}")

        for utils_file in utils_files:
            try:
                f_name = os.path.basename(utils_file)
                dest_folder = imports_to_builder_map[f_name.replace(".py", "")]
                self._logger.info(f"Moving {dest_folder} to {utils_file}")
                shutil.copy(utils_file, dest_folder)
            except KeyError:
                self._logger.error(f"Cannot find destination folder for {utils_file}. Please copy manually.")

        if with_manual_update:
            for file_path in with_manual_update:
                self._logger.warning(
                    f"You need to manually update file {file_path} to remove configurations using 'TextEncoderConfig'."
                )
