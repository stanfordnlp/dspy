"""
This module will parse the JSON file following the BNF definition:

    <json> ::= <container>

    <primitive> ::= <number> | <string> | <boolean>
    ; Where:
    ; <number> is a valid real number expressed in one of a number of given formats
    ; <string> is a string of valid characters enclosed in quotes
    ; <boolean> is one of the literal strings 'true', 'false', or 'null' (unquoted)

    <container> ::= <object> | <array>
    <array> ::= '[' [ <json> *(', ' <json>) ] ']' ; A sequence of JSON values separated by commas
    <object> ::= '{' [ <member> *(', ' <member>) ] '}' ; A sequence of 'members'
    <member> ::= <string> ': ' <json> ; A pair consisting of a name, and a JSON value

If something is wrong (a missing parantheses or quotes for example) it will use a few simple heuristics to fix the JSON string:
- Add the missing parentheses if the parser believes that the array or object should be closed
- Quote strings or add missing single quotes
- Adjust whitespaces and remove line breaks

All supported use cases are in the unit tests
"""

import argparse
import json
import sys
from typing import Dict, List, Optional, TextIO, Tuple, Union

from .json_parser import JSONParser, JSONReturnType


def repair_json(
    json_str: str = "",
    return_objects: bool = False,
    skip_json_loads: bool = False,
    logging: bool = False,
    json_fd: Optional[TextIO] = None,
    ensure_ascii: bool = True,
    chunk_length: int = 0,
) -> Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]:
    """
    Given a json formatted string, it will try to decode it and, if it fails, it will try to fix it.

    Args:
        json_str (str, optional): The JSON string to repair. Defaults to an empty string.
        return_objects (bool, optional): If True, return the decoded data structure. Defaults to False.
        skip_json_loads (bool, optional): If True, skip calling the built-in json.loads() function to verify that the json is valid before attempting to repair. Defaults to False.
        logging (bool, optional): If True, return a tuple with the repaired json and a log of all repair actions. Defaults to False.
        json_fd (Optional[TextIO], optional): File descriptor for JSON input. Do not use! Use `from_file` or `load` instead. Defaults to None.
        ensure_ascii (bool, optional): Set to False to avoid converting non-latin characters to ascii (for example when using chinese characters). Defaults to True. Ignored if `skip_json_loads` is True.
        chunk_length (int, optional): Size in bytes of the file chunks to read at once. Ignored if `json_fd` is None. Do not use! Use `from_file` or `load` instead. Defaults to 1MB.

    Returns:
        Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]: The repaired JSON or a tuple with the repaired JSON and repair log.
    """
    parser = JSONParser(json_str, json_fd, logging, chunk_length)
    if skip_json_loads:
        parsed_json = parser.parse()
    else:
        try:
            if json_fd:
                parsed_json = json.load(json_fd)
            else:
                parsed_json = json.loads(json_str)
        except json.JSONDecodeError:
            parsed_json = parser.parse()
    # It's useful to return the actual object instead of the json string,
    # it allows this lib to be a replacement of the json library
    if return_objects or logging:
        return parsed_json
    return json.dumps(parsed_json, ensure_ascii=ensure_ascii)


def loads(
    json_str: str,
    skip_json_loads: bool = False,
    logging: bool = False,
) -> Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]:
    """
    This function works like `json.loads()` except that it will fix your JSON in the process.
    It is a wrapper around the `repair_json()` function with `return_objects=True`.

    Args:
        json_str (str): The JSON string to load and repair.
        skip_json_loads (bool, optional): If True, skip calling the built-in json.loads() function to verify that the json is valid before attempting to repair. Defaults to False.
        logging (bool, optional): If True, return a tuple with the repaired json and a log of all repair actions. Defaults to False.

    Returns:
        Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]: The repaired JSON object or a tuple with the repaired JSON object and repair log.
    """
    return repair_json(
        json_str=json_str,
        return_objects=True,
        skip_json_loads=skip_json_loads,
        logging=logging,
    )


def load(
    fd: TextIO,
    skip_json_loads: bool = False,
    logging: bool = False,
    chunk_length: int = 0,
) -> Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]:
    """
    This function works like `json.load()` except that it will fix your JSON in the process.
    It is a wrapper around the `repair_json()` function with `json_fd=fd` and `return_objects=True`.

    Args:
        fd (TextIO): File descriptor for JSON input.
        skip_json_loads (bool, optional): If True, skip calling the built-in json.loads() function to verify that the json is valid before attempting to repair. Defaults to False.
        logging (bool, optional): If True, return a tuple with the repaired json and a log of all repair actions. Defaults to False.
        chunk_length (int, optional): Size in bytes of the file chunks to read at once. Defaults to 1MB.

    Returns:
        Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]: The repaired JSON object or a tuple with the repaired JSON object and repair log.
    """
    return repair_json(
        json_fd=fd,
        chunk_length=chunk_length,
        return_objects=True,
        skip_json_loads=skip_json_loads,
        logging=logging,
    )


def from_file(
    filename: str,
    skip_json_loads: bool = False,
    logging: bool = False,
    chunk_length: int = 0,
) -> Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]:
    """
    This function is a wrapper around `load()` so you can pass the filename as string

    Args:
        filename (str): The name of the file containing JSON data to load and repair.
        skip_json_loads (bool, optional): If True, skip calling the built-in json.loads() function to verify that the json is valid before attempting to repair. Defaults to False.
        logging (bool, optional): If True, return a tuple with the repaired json and a log of all repair actions. Defaults to False.
        chunk_length (int, optional): Size in bytes of the file chunks to read at once. Defaults to 1MB.

    Returns:
        Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]: The repaired JSON object or a tuple with the repaired JSON object and repair log.
    """
    with open(filename) as fd:
        jsonobj = load(
            fd=fd,
            skip_json_loads=skip_json_loads,
            logging=logging,
            chunk_length=chunk_length,
        )

    return jsonobj


def cli(inline_args: Optional[List[str]] = None) -> int:
    """
    Command-line interface for repairing and parsing JSON files.

    Args:
        inline_args (Optional[List[str]]): List of command-line arguments for testing purposes. Defaults to None.
            - filename (str): The JSON file to repair. If omitted, the JSON is read from stdin.
            - -i, --inline (bool): Replace the file inline instead of returning the output to stdout.
            - -o, --output TARGET (str): If specified, the output will be written to TARGET filename instead of stdout.
            - --ensure_ascii (bool): Pass ensure_ascii=True to json.dumps(). Will pass False otherwise.
            - --indent INDENT (int): Number of spaces for indentation (Default 2).

    Returns:
        int: Exit code of the CLI operation.

    Raises:
        Exception: Any exception that occurs during file processing.

    Example:
        >>> cli(['example.json', '--indent', '4'])
        >>> cat json.txt | json_repair
    """
    parser = argparse.ArgumentParser(description="Repair and parse JSON files.")
    # Make the filename argument optional; if omitted, we will read from stdin.
    parser.add_argument(
        "filename",
        nargs="?",
        help="The JSON file to repair (if omitted, reads from stdin)",
    )
    parser.add_argument(
        "-i",
        "--inline",
        action="store_true",
        help="Replace the file inline instead of returning the output to stdout",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="TARGET",
        help="If specified, the output will be written to TARGET filename instead of stdout",
    )
    parser.add_argument(
        "--ensure_ascii",
        action="store_true",
        help="Pass ensure_ascii=True to json.dumps()",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Number of spaces for indentation (Default 2)",
    )

    if inline_args is None:  # pragma: no cover
        args = parser.parse_args()
    else:
        args = parser.parse_args(inline_args)

    # Inline mode requires a filename, so error out if none was provided.
    if args.inline and not args.filename:  # pragma: no cover
        print("Error: Inline mode requires a filename", file=sys.stderr)
        sys.exit(1)

    if args.inline and args.output:  # pragma: no cover
        print("Error: You cannot pass both --inline and --output", file=sys.stderr)
        sys.exit(1)

    ensure_ascii = False
    if args.ensure_ascii:
        ensure_ascii = True

    try:
        # Use from_file if a filename is provided; otherwise read from stdin.
        if args.filename:
            result = from_file(args.filename)
        else:
            data = sys.stdin.read()
            result = loads(data)
        if args.inline or args.output:
            with open(args.output or args.filename, mode="w") as fd:
                json.dump(result, fd, indent=args.indent, ensure_ascii=ensure_ascii)
        else:
            print(json.dumps(result, indent=args.indent, ensure_ascii=ensure_ascii))
    except Exception as e:  # pragma: no cover
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

    return 0  # Success


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli())
