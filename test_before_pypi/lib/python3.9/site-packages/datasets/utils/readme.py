# loading package files: https://stackoverflow.com/a/20885799
import importlib.resources as pkg_resources
import logging
from pathlib import Path
from typing import Any, List, Tuple

import yaml

from . import resources
from .deprecation_utils import deprecated


BASE_REF_URL = "https://github.com/huggingface/datasets/tree/main/src/datasets/utils"
this_url = f"{BASE_REF_URL}/{__file__}"
logger = logging.getLogger(__name__)


def load_yaml_resource(resource: str) -> Tuple[Any, str]:
    content = pkg_resources.read_text(resources, resource)
    return yaml.safe_load(content), f"{BASE_REF_URL}/resources/{resource}"


readme_structure, known_readme_structure_url = load_yaml_resource("readme_structure.yaml")

FILLER_TEXT = [
    "[Needs More Information]",
    "[More Information Needed]",
    "(https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)",
]

# Dictionary representation of section/readme, error_list, warning_list
ReadmeValidatorOutput = Tuple[dict, List[str], List[str]]


class Section:
    def __init__(self, name: str, level: str, lines: List[str] = None, suppress_parsing_errors: bool = False):
        self.name = name
        self.level = level
        self.lines = lines
        self.text = ""
        self.is_empty_text = True
        self.content = {}
        self.parsing_error_list = []
        self.parsing_warning_list = []
        if self.lines is not None:
            self.parse(suppress_parsing_errors=suppress_parsing_errors)

    def parse(self, suppress_parsing_errors: bool = False):
        current_sub_level = ""
        current_lines = []
        code_start = False
        for line in self.lines:
            if line.strip(" \n") == "":
                continue
            elif line.strip(" \n")[:3] == "```":
                code_start = not code_start
            elif line.split()[0] == self.level + "#" and not code_start:
                if current_sub_level != "":
                    self.content[current_sub_level] = Section(current_sub_level, self.level + "#", current_lines)
                    current_lines = []
                else:
                    if current_lines != []:
                        self.text += "".join(current_lines).strip()
                        if self.text != "" and self.text not in FILLER_TEXT:
                            self.is_empty_text = False
                        current_lines = []

                current_sub_level = " ".join(line.split()[1:]).strip(" \n")
            else:
                current_lines.append(line)
        else:
            if current_sub_level != "":
                if current_sub_level in self.content:
                    self.parsing_error_list.append(
                        f"Multiple sections with the same heading `{current_sub_level}` have been found. Please keep only one of these sections."
                    )
                self.content[current_sub_level] = Section(current_sub_level, self.level + "#", current_lines)
            else:
                if current_lines != []:
                    self.text += "".join(current_lines).strip()
                    if self.text != "" and self.text not in FILLER_TEXT:
                        self.is_empty_text = False

        if self.level == "" and not suppress_parsing_errors:
            if self.parsing_error_list != [] or self.parsing_warning_list != []:
                errors = errors = "\n".join("-\t" + x for x in self.parsing_error_list + self.parsing_warning_list)
                error_string = f"The following issues were found while parsing the README at `{self.name}`:\n" + errors
                raise ValueError(error_string)

    def validate(self, structure: dict) -> ReadmeValidatorOutput:
        """Validates a Section class object recursively using the structure provided as a dictionary.

        Args:
            structute (:obj: `dict`): The dictionary representing expected structure.

        Returns:
            :obj: `ReadmeValidatorOutput`: The dictionary representation of the section, and the errors.
        """
        # Header text validation
        error_list = []
        warning_list = []
        if structure["allow_empty"] is False:
            # If content is expected
            if self.is_empty_text and self.content == {}:
                # If no content is found, mention it in the error_list
                error_list.append(f"Expected some content in section `{self.name}` but it is empty.")

        if structure["allow_empty_text"] is False:
            # If some text is expected
            if self.is_empty_text:
                # If no text is found, mention it in the error_list
                error_list.append(
                    f"Expected some text in section `{self.name}` but it is empty (text in subsections are ignored)."
                )
        # Subsections Validation
        if structure["subsections"] is not None:
            # If subsections are expected
            if self.content == {}:
                # If no subsections are present
                values = [subsection["name"] for subsection in structure["subsections"]]
                # Mention the expected values in the error_list
                error_list.append(
                    f"Section `{self.name}` expected the following subsections: {', '.join(['`'+x+'`' for x in values])}. Found 'None'."
                )
            else:
                # If some subsections are present
                structure_names = [subsection["name"] for subsection in structure["subsections"]]
                has_missing_subsections = False
                for idx, name in enumerate(structure_names):
                    if name not in self.content:
                        # If the expected subsection is not present
                        error_list.append(f"Section `{self.name}` is missing subsection: `{name}`.")
                        has_missing_subsections = True
                    else:
                        # If the subsection is present, validate subsection, return the result
                        # and concat the errors from subsection to section error_list

                        # Skip sublevel validation if current level is `###`
                        if self.level == "###":
                            continue
                        else:
                            _, subsec_error_list, subsec_warning_list = self.content[name].validate(
                                structure["subsections"][idx]
                            )
                        error_list += subsec_error_list
                        warning_list += subsec_warning_list

                if has_missing_subsections:  # we only allow to have extra subsections if all the other ones are here
                    for name in self.content:
                        if name not in structure_names:
                            # If an extra subsection is present
                            warning_list.append(
                                f"`{self.name}` has an extra subsection: `{name}`. Skipping further validation checks for this subsection as expected structure is unknown."
                            )
        if error_list:
            # If there are errors, do not return the dictionary as it is invalid
            return {}, error_list, warning_list
        else:
            return self.to_dict(), error_list, warning_list

    def to_dict(self) -> dict:
        """Returns the dictionary representation of a section."""
        return {
            "name": self.name,
            "text": self.text,
            "is_empty_text": self.is_empty_text,
            "subsections": [value.to_dict() for value in self.content.values()],
        }


@deprecated("Use `huggingface_hub.DatasetCard` instead.")
class ReadMe(Section):  # Level 0
    def __init__(self, name: str, lines: List[str], structure: dict = None, suppress_parsing_errors: bool = False):
        super().__init__(name=name, level="")  # Not using lines here as we need to use a child class parse
        self.structure = structure
        self.yaml_tags_line_count = -2
        self.tag_count = 0
        self.lines = lines
        if self.lines is not None:
            self.parse(suppress_parsing_errors=suppress_parsing_errors)

    def validate(self):
        if self.structure is None:
            content, error_list, warning_list = self._validate(readme_structure)
        else:
            content, error_list, warning_list = self._validate(self.structure)
        if error_list != [] or warning_list != []:
            errors = "\n".join(["-\t" + x for x in error_list + warning_list])
            error_string = f"The following issues were found for the README at `{self.name}`:\n" + errors
            raise ValueError(error_string)

    @classmethod
    def from_readme(cls, path: Path, structure: dict = None, suppress_parsing_errors: bool = False):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        return cls(path, lines, structure, suppress_parsing_errors=suppress_parsing_errors)

    @classmethod
    def from_string(
        cls, string: str, structure: dict = None, root_name: str = "root", suppress_parsing_errors: bool = False
    ):
        lines = string.split("\n")
        return cls(root_name, lines, structure, suppress_parsing_errors=suppress_parsing_errors)

    def parse(self, suppress_parsing_errors: bool = False):
        # Skip Tags
        line_count = 0

        for line in self.lines:
            self.yaml_tags_line_count += 1
            if line.strip(" \n") == "---":
                self.tag_count += 1
                if self.tag_count == 2:
                    break
            line_count += 1
        if self.tag_count == 2:
            self.lines = self.lines[line_count + 1 :]  # Get the last + 1 th item.
        else:
            self.lines = self.lines[self.tag_count :]
        super().parse(suppress_parsing_errors=suppress_parsing_errors)

    def __str__(self):
        """Returns the string of dictionary representation of the ReadMe."""
        return str(self.to_dict())

    def _validate(self, readme_structure):
        error_list = []
        warning_list = []
        if self.yaml_tags_line_count == 0:
            warning_list.append("Empty YAML markers are present in the README.")
        elif self.tag_count == 0:
            warning_list.append("No YAML markers are present in the README.")
        elif self.tag_count == 1:
            warning_list.append("Only the start of YAML tags present in the README.")
        # Check how many first level sections are present.
        num_first_level_keys = len(self.content.keys())
        if num_first_level_keys > 1:
            # If more than one, add to the error list, continue
            error_list.append(
                f"The README has several first-level headings: {', '.join(['`'+x+'`' for x in list(self.content.keys())])}. Only one heading is expected. Skipping further validation for this README."
            )
        elif num_first_level_keys < 1:
            # If less than one, append error.
            error_list.append(
                "The README has no first-level headings. One heading is expected. Skipping further validation for this README."
            )

        else:
            # If one exactly
            start_key = list(self.content.keys())[0]  # Get the key
            if start_key.startswith("Dataset Card for"):  # Check correct start
                # If the starting is correct, validate all the sections
                _, sec_error_list, sec_warning_list = self.content[start_key].validate(
                    readme_structure["subsections"][0]
                )
                error_list += sec_error_list
                warning_list += sec_warning_list
            else:
                # If not found, append error
                error_list.append(
                    "No first-level heading starting with `Dataset Card for` found in README. Skipping further validation for this README."
                )
        if error_list:
            # If there are errors, do not return the dictionary as it is invalid
            return {}, error_list, warning_list
        else:
            return self.to_dict(), error_list, warning_list


if __name__ == "__main__":
    from argparse import ArgumentParser

    ap = ArgumentParser(usage="Validate the content (excluding YAML tags) of a README.md file.")
    ap.add_argument("readme_filepath")
    args = ap.parse_args()
    readme_filepath = Path(args.readme_filepath)
    readme = ReadMe.from_readme(readme_filepath)
