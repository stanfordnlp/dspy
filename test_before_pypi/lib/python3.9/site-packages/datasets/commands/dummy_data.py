import fnmatch
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from datasets import config
from datasets.commands import BaseDatasetsCLICommand
from datasets.download.download_config import DownloadConfig
from datasets.download.download_manager import DownloadManager
from datasets.download.mock_download_manager import MockDownloadManager
from datasets.load import dataset_module_factory, import_main_class
from datasets.utils.deprecation_utils import deprecated
from datasets.utils.logging import get_logger, set_verbosity_warning
from datasets.utils.py_utils import map_nested


logger = get_logger(__name__)

DEFAULT_ENCODING = "utf-8"


def dummy_data_command_factory(args):
    return DummyDataCommand(
        args.path_to_dataset,
        args.auto_generate,
        args.n_lines,
        args.json_field,
        args.xml_tag,
        args.match_text_files,
        args.keep_uncompressed,
        args.cache_dir,
        args.encoding,
    )


class DummyDataGeneratorDownloadManager(DownloadManager):
    def __init__(self, mock_download_manager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mock_download_manager = mock_download_manager
        self.downloaded_dummy_paths = []
        self.expected_dummy_paths = []

    def download(self, url_or_urls):
        output = super().download(url_or_urls)
        dummy_output = self.mock_download_manager.download(url_or_urls)
        map_nested(self.downloaded_dummy_paths.append, output, map_tuple=True)
        map_nested(self.expected_dummy_paths.append, dummy_output, map_tuple=True)
        return output

    def download_and_extract(self, url_or_urls):
        output = super().extract(super().download(url_or_urls))
        dummy_output = self.mock_download_manager.download(url_or_urls)
        map_nested(self.downloaded_dummy_paths.append, output, map_tuple=True)
        map_nested(self.expected_dummy_paths.append, dummy_output, map_tuple=True)
        return output

    def auto_generate_dummy_data_folder(
        self,
        n_lines: int = 5,
        json_field: Optional[str] = None,
        xml_tag: Optional[str] = None,
        match_text_files: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> bool:
        os.makedirs(
            os.path.join(
                self.mock_download_manager.datasets_scripts_dir,
                self.mock_download_manager.dataset_name,
                self.mock_download_manager.dummy_data_folder,
                "dummy_data",
            ),
            exist_ok=True,
        )
        total = 0
        self.mock_download_manager.load_existing_dummy_data = False
        for src_path, relative_dst_path in zip(self.downloaded_dummy_paths, self.expected_dummy_paths):
            dst_path = os.path.join(
                self.mock_download_manager.datasets_scripts_dir,
                self.mock_download_manager.dataset_name,
                self.mock_download_manager.dummy_data_folder,
                relative_dst_path,
            )
            total += self._create_dummy_data(
                src_path,
                dst_path,
                n_lines=n_lines,
                json_field=json_field,
                xml_tag=xml_tag,
                match_text_files=match_text_files,
                encoding=encoding,
            )
        if total == 0:
            logger.error(
                "Dummy data generation failed: no dummy files were created. "
                "Make sure the data files format is supported by the auto-generation."
            )
        return total > 0

    def _create_dummy_data(
        self,
        src_path: str,
        dst_path: str,
        n_lines: int,
        json_field: Optional[str] = None,
        xml_tag: Optional[str] = None,
        match_text_files: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> int:
        encoding = encoding or DEFAULT_ENCODING
        if os.path.isfile(src_path):
            logger.debug(f"Trying to generate dummy data file {dst_path}")
            dst_path_extensions = Path(dst_path).suffixes
            line_by_line_extensions = [".txt", ".csv", ".jsonl", ".tsv"]
            is_line_by_line_text_file = any(extension in dst_path_extensions for extension in line_by_line_extensions)
            if match_text_files is not None:
                file_name = os.path.basename(dst_path)
                for pattern in match_text_files.split(","):
                    is_line_by_line_text_file |= fnmatch.fnmatch(file_name, pattern)
            # Line by line text file (txt, csv etc.)
            if is_line_by_line_text_file:
                Path(dst_path).parent.mkdir(exist_ok=True, parents=True)
                with open(src_path, encoding=encoding) as src_file:
                    with open(dst_path, "w", encoding=encoding) as dst_file:
                        first_lines = []
                        for i, line in enumerate(src_file):
                            if i >= n_lines:
                                break
                            first_lines.append(line)
                        dst_file.write("".join(first_lines).strip())
                return 1
            # json file
            elif ".json" in dst_path_extensions:
                with open(src_path, encoding=encoding) as src_file:
                    json_data = json.load(src_file)
                    if json_field is not None:
                        json_data = json_data[json_field]
                    if isinstance(json_data, dict):
                        if not all(isinstance(v, list) for v in json_data.values()):
                            raise ValueError(
                                f"Couldn't parse columns {list(json_data.keys())}. "
                                "Maybe specify which json field must be used "
                                "to read the data with --json_field <my_field>."
                            )
                        first_json_data = {k: v[:n_lines] for k, v in json_data.items()}
                    else:
                        first_json_data = json_data[:n_lines]
                    if json_field is not None:
                        first_json_data = {json_field: first_json_data}
                    Path(dst_path).parent.mkdir(exist_ok=True, parents=True)
                    with open(dst_path, "w", encoding=encoding) as dst_file:
                        json.dump(first_json_data, dst_file)
                return 1
            # xml file
            elif any(extension in dst_path_extensions for extension in [".xml", ".txm"]):
                if xml_tag is None:
                    logger.warning("Found xml file but 'xml_tag' is set to None. Please provide --xml_tag")
                else:
                    self._create_xml_dummy_data(src_path, dst_path, xml_tag, n_lines=n_lines, encoding=encoding)
                return 1
            logger.warning(
                f"Couldn't generate dummy file '{dst_path}'. " "Ignore that if this file is not useful for dummy data."
            )
            return 0
        # directory, iterate through all files
        elif os.path.isdir(src_path):
            total = 0
            for path, _, files in os.walk(src_path):
                for name in files:
                    if not name.startswith("."):  # ignore files like .DS_Store etc.
                        src_file_path = os.path.join(path, name)
                        dst_file_path = os.path.join(dst_path, Path(src_file_path).relative_to(src_path))
                        total += self._create_dummy_data(
                            src_file_path,
                            dst_file_path,
                            n_lines=n_lines,
                            json_field=json_field,
                            xml_tag=xml_tag,
                            match_text_files=match_text_files,
                            encoding=encoding,
                        )
            return total

    @staticmethod
    def _create_xml_dummy_data(src_path, dst_path, xml_tag, n_lines=5, encoding=DEFAULT_ENCODING):
        Path(dst_path).parent.mkdir(exist_ok=True, parents=True)
        with open(src_path, encoding=encoding) as src_file:
            n_line = 0
            parents = []
            for event, elem in ET.iterparse(src_file, events=("start", "end")):
                if event == "start":
                    parents.append(elem)
                else:
                    _ = parents.pop()
                    if elem.tag == xml_tag:
                        if n_line < n_lines:
                            n_line += 1
                        else:
                            if parents:
                                parents[-1].remove(elem)
            ET.ElementTree(element=elem).write(dst_path, encoding=encoding)

    def compress_autogenerated_dummy_data(self, path_to_dataset):
        root_dir = os.path.join(path_to_dataset, self.mock_download_manager.dummy_data_folder)
        base_name = os.path.join(root_dir, "dummy_data")
        base_dir = "dummy_data"
        logger.info(f"Compressing dummy data folder to '{base_name}.zip'")
        shutil.make_archive(base_name, "zip", root_dir, base_dir)
        shutil.rmtree(base_name)


@deprecated(
    "The `datasets` repository does not host the dataset scripts anymore. Therefore, dummy data is no longer needed to test their loading with CI."
)
class DummyDataCommand(BaseDatasetsCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        test_parser = parser.add_parser("dummy_data", help="Generate dummy data.")
        test_parser.add_argument("--auto_generate", action="store_true", help="Automatically generate dummy data")
        test_parser.add_argument(
            "--n_lines", type=int, default=5, help="Number of lines or samples to keep when auto-generating dummy data"
        )
        test_parser.add_argument(
            "--json_field",
            type=str,
            default=None,
            help="Optional, json field to read the data from when auto-generating dummy data. In the json data files, this field must point to a list of samples as json objects (ex: the 'data' field for squad-like files)",
        )
        test_parser.add_argument(
            "--xml_tag",
            type=str,
            default=None,
            help="Optional, xml tag name of the samples inside the xml files when auto-generating dummy data.",
        )
        test_parser.add_argument(
            "--match_text_files",
            type=str,
            default=None,
            help="Optional, a comma separated list of file patterns that looks for line-by-line text files other than *.txt or *.csv. Example: --match_text_files *.label",
        )
        test_parser.add_argument(
            "--keep_uncompressed",
            action="store_true",
            help="Whether to leave the dummy data folders uncompressed when auto-generating dummy data. Useful for debugging for to do manual adjustements before compressing.",
        )
        test_parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Cache directory to download and cache files when auto-generating dummy data",
        )
        test_parser.add_argument(
            "--encoding",
            type=str,
            default=None,
            help=f"Encoding to use when auto-generating dummy data. Defaults to {DEFAULT_ENCODING}",
        )
        test_parser.add_argument("path_to_dataset", type=str, help="Path to the dataset (example: ./datasets/squad)")
        test_parser.set_defaults(func=dummy_data_command_factory)

    def __init__(
        self,
        path_to_dataset: str,
        auto_generate: bool,
        n_lines: int,
        json_field: Optional[str],
        xml_tag: Optional[str],
        match_text_files: Optional[str],
        keep_uncompressed: bool,
        cache_dir: Optional[str],
        encoding: Optional[str],
    ):
        self._path_to_dataset = path_to_dataset
        if os.path.isdir(path_to_dataset):
            self._dataset_name = path_to_dataset.replace(os.sep, "/").split("/")[-1]
        else:
            self._dataset_name = path_to_dataset.replace(os.sep, "/").split("/")[-2]
        cache_dir = os.path.expanduser(cache_dir or config.HF_DATASETS_CACHE)
        self._auto_generate = auto_generate
        self._n_lines = n_lines
        self._json_field = json_field
        self._xml_tag = xml_tag
        self._match_text_files = match_text_files
        self._keep_uncompressed = keep_uncompressed
        self._cache_dir = cache_dir
        self._encoding = encoding

    def run(self):
        set_verbosity_warning()
        dataset_module = dataset_module_factory(self._path_to_dataset)
        builder_cls = import_main_class(dataset_module.module_path)

        # use `None` as config if no configs
        builder_configs = builder_cls.BUILDER_CONFIGS or [None]
        auto_generate_results = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for builder_config in builder_configs:
                config_name = builder_config.name if builder_config else None
                dataset_builder = builder_cls(config_name=config_name, hash=dataset_module.hash, cache_dir=tmp_dir)
                version = builder_config.version if builder_config else dataset_builder.config.version
                mock_dl_manager = MockDownloadManager(
                    dataset_name=self._dataset_name,
                    config=builder_config,
                    version=version,
                    use_local_dummy_data=True,
                    load_existing_dummy_data=False,
                )

                if self._auto_generate:
                    auto_generate_results.append(
                        self._autogenerate_dummy_data(
                            dataset_builder=dataset_builder,
                            mock_dl_manager=mock_dl_manager,
                            keep_uncompressed=self._keep_uncompressed,
                        )
                    )
                else:
                    self._print_dummy_data_instructions(
                        dataset_builder=dataset_builder, mock_dl_manager=mock_dl_manager
                    )
            if self._auto_generate and not self._keep_uncompressed:
                if all(auto_generate_results):
                    print(f"Automatic dummy data generation succeeded for all configs of '{self._path_to_dataset}'")
                else:
                    print(f"Automatic dummy data generation failed for some configs of '{self._path_to_dataset}'")

    def _autogenerate_dummy_data(self, dataset_builder, mock_dl_manager, keep_uncompressed) -> Optional[bool]:
        dl_cache_dir = (
            os.path.join(self._cache_dir, config.DOWNLOADED_DATASETS_DIR)
            if self._cache_dir
            else config.DOWNLOADED_DATASETS_PATH
        )
        download_config = DownloadConfig(cache_dir=dl_cache_dir)
        dl_manager = DummyDataGeneratorDownloadManager(
            dataset_name=self._dataset_name, mock_download_manager=mock_dl_manager, download_config=download_config
        )
        dataset_builder._split_generators(dl_manager)
        mock_dl_manager.load_existing_dummy_data = False  # don't use real dummy data
        dl_manager.auto_generate_dummy_data_folder(
            n_lines=self._n_lines,
            json_field=self._json_field,
            xml_tag=self._xml_tag,
            match_text_files=self._match_text_files,
            encoding=self._encoding,
        )
        if not keep_uncompressed:
            path_do_dataset = os.path.join(mock_dl_manager.datasets_scripts_dir, mock_dl_manager.dataset_name)
            dl_manager.compress_autogenerated_dummy_data(path_do_dataset)
            # now test that the dummy_data.zip file actually works
            mock_dl_manager.load_existing_dummy_data = True  # use real dummy data
            n_examples_per_split = {}
            os.makedirs(dataset_builder._cache_dir, exist_ok=True)
            try:
                split_generators = dataset_builder._split_generators(mock_dl_manager)
                for split_generator in split_generators:
                    dataset_builder._prepare_split(split_generator, check_duplicate_keys=False)
                    n_examples_per_split[split_generator.name] = split_generator.split_info.num_examples
            except OSError as e:
                logger.error(
                    f"Failed to load dummy data for config '{dataset_builder.config.name}''.\nOriginal error:\n"
                    + str(e)
                )
                return False
            else:
                if all(n_examples > 0 for n_examples in n_examples_per_split.values()):
                    logger.warning(
                        f"Dummy data generation done and dummy data test succeeded for config '{dataset_builder.config.name}''."
                    )
                    return True
                else:
                    empty_splits = [
                        split_name for split_name in n_examples_per_split if n_examples_per_split[split_name] == 0
                    ]
                    logger.warning(
                        f"Dummy data generation done but dummy data test failed since splits {empty_splits} have 0 examples for config '{dataset_builder.config.name}''."
                    )
                    return False
        else:
            generated_dummy_data_dir = os.path.join(self._path_to_dataset, mock_dl_manager.dummy_data_folder)
            logger.info(
                f"Dummy data generated in directory '{generated_dummy_data_dir}' but kept uncompressed. "
                "Please compress this directory into a zip file to use it for dummy data tests."
            )

    def _print_dummy_data_instructions(self, dataset_builder, mock_dl_manager):
        dummy_data_folder = os.path.join(self._path_to_dataset, mock_dl_manager.dummy_data_folder)
        logger.info(f"Creating dummy folder structure for {dummy_data_folder}... ")
        os.makedirs(dummy_data_folder, exist_ok=True)

        try:
            generator_splits = dataset_builder._split_generators(mock_dl_manager)
        except FileNotFoundError as e:
            print(
                f"Dataset {self._dataset_name} with config {mock_dl_manager.config} seems to already open files in the method `_split_generators(...)`. You might consider to instead only open files in the method `_generate_examples(...)` instead. If this is not possible the dummy data has to be created with less guidance. Make sure you create the file {e.filename}."
            )

        files_to_create = set()
        split_names = []
        dummy_file_name = mock_dl_manager.dummy_file_name

        for split in generator_splits:
            logger.info(f"Collecting dummy data file paths to create for {split.name}")
            split_names.append(split.name)
            gen_kwargs = split.gen_kwargs
            generator = dataset_builder._generate_examples(**gen_kwargs)

            try:
                dummy_data_guidance_print = "\n" + 30 * "=" + "DUMMY DATA INSTRUCTIONS" + 30 * "=" + "\n"
                config_string = (
                    f"config {mock_dl_manager.config.name} of " if mock_dl_manager.config is not None else ""
                )
                dummy_data_guidance_print += (
                    "- In order to create the dummy data for "
                    + config_string
                    + f"{self._dataset_name}, please go into the folder '{dummy_data_folder}' with `cd {dummy_data_folder}` . \n\n"
                )

                # trigger generate function
                for key, record in generator:
                    pass

                dummy_data_guidance_print += f"- It appears that the function `_generate_examples(...)` expects one or more files in the folder {dummy_file_name} using the function `glob.glob(...)`. In this case, please refer to the `_generate_examples(...)` method to see under which filename the dummy data files should be created. \n\n"

            except FileNotFoundError as e:
                files_to_create.add(e.filename)

        split_names = ", ".join(split_names)
        if len(files_to_create) > 0:
            # no glob.glob(...) in `_generate_examples(...)`
            if len(files_to_create) == 1 and next(iter(files_to_create)) == dummy_file_name:
                dummy_data_guidance_print += f"- Please create a single dummy data file called '{next(iter(files_to_create))}' from the folder '{dummy_data_folder}'. Make sure that the dummy data file provides at least one example for the split(s) '{split_names}' \n\n"
                files_string = dummy_file_name
            else:
                files_string = ", ".join(files_to_create)
                dummy_data_guidance_print += f"- Please create the following dummy data files '{files_string}' from the folder '{dummy_data_folder}'\n\n"

                dummy_data_guidance_print += f"- For each of the splits '{split_names}', make sure that one or more of the dummy data files provide at least one example \n\n"

            dummy_data_guidance_print += f"- If the method `_generate_examples(...)` includes multiple `open()` statements, you might have to create other files in addition to '{files_string}'. In this case please refer to the `_generate_examples(...)` method \n\n"

        if len(files_to_create) == 1 and next(iter(files_to_create)) == dummy_file_name:
            dummy_data_guidance_print += f"- After the dummy data file is created, it should be zipped to '{dummy_file_name}.zip' with the command `zip {dummy_file_name}.zip {dummy_file_name}` \n\n"

            dummy_data_guidance_print += (
                f"- You can now delete the file '{dummy_file_name}' with the command `rm {dummy_file_name}` \n\n"
            )

            dummy_data_guidance_print += f"- To get the file '{dummy_file_name}' back for further changes to the dummy data, simply unzip {dummy_file_name}.zip with the command `unzip {dummy_file_name}.zip` \n\n"
        else:
            dummy_data_guidance_print += f"- After all dummy data files are created, they should be zipped recursively to '{dummy_file_name}.zip' with the command `zip -r {dummy_file_name}.zip {dummy_file_name}/` \n\n"

            dummy_data_guidance_print += (
                f"- You can now delete the folder '{dummy_file_name}' with the command `rm -r {dummy_file_name}` \n\n"
            )

            dummy_data_guidance_print += f"- To get the folder '{dummy_file_name}' back for further changes to the dummy data, simply unzip {dummy_file_name}.zip with the command `unzip {dummy_file_name}.zip` \n\n"

        dummy_data_guidance_print += (
            f"- Make sure you have created the file '{dummy_file_name}.zip' in '{dummy_data_folder}' \n"
        )

        dummy_data_guidance_print += 83 * "=" + "\n"

        print(dummy_data_guidance_print)
