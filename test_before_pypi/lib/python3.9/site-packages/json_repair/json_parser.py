from typing import Any, Dict, List, Literal, Optional, TextIO, Tuple, Union

from .json_context import ContextValues, JsonContext
from .string_file_wrapper import StringFileWrapper

JSONReturnType = Union[Dict[str, Any], List[Any], str, float, int, bool, None]


class JSONParser:
    # Constants
    STRING_DELIMITERS = ['"', "'", "“", "”"]
    NUMBER_CHARS = set("0123456789-.eE/,")

    def __init__(
        self,
        json_str: Union[str, StringFileWrapper],
        json_fd: Optional[TextIO],
        logging: Optional[bool],
        json_fd_chunk_length: int = 0,
    ) -> None:
        # The string to parse
        self.json_str: Union[str, StringFileWrapper] = json_str
        # Alternatively, the file description with a json file in it
        if json_fd:
            # This is a trick we do to treat the file wrapper as an array
            self.json_str = StringFileWrapper(json_fd, json_fd_chunk_length)
        # Index is our iterator that will keep track of which character we are looking at right now
        self.index: int = 0
        # This is used in the object member parsing to manage the special cases of missing quotes in key or value
        self.context = JsonContext()
        # Use this to log the activity, but only if logging is active

        # This is a trick but a beatiful one. We call self.log in the code over and over even if it's not needed.
        # We could add a guard in the code for each call but that would make this code unreadable, so here's this neat trick
        # Replace self.log with a noop
        self.logging = logging
        if logging:
            self.logger: List[Dict[str, str]] = []
            self.log = self._log
        else:
            # No-op
            self.log = lambda *args, **kwargs: None

    def parse(
        self,
    ) -> Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]:
        json = self.parse_json()
        if self.index < len(self.json_str):
            self.log(
                "The parser returned early, checking if there's more json elements",
            )
            json = [json]
            last_index = self.index
            while self.index < len(self.json_str):
                j = self.parse_json()
                if j != "":
                    json.append(j)
                if self.index == last_index:
                    self.index += 1
                last_index = self.index
            # If nothing extra was found, don't return an array
            if len(json) == 1:
                self.log(
                    "There were no more elements, returning the element without the array",
                )
                json = json[0]
        if self.logging:
            return json, self.logger
        else:
            return json

    def parse_json(
        self,
    ) -> JSONReturnType:
        while True:
            char = self.get_char_at()
            # False means that we are at the end of the string provided
            if char is False:
                return ""
            # <object> starts with '{'
            elif char == "{":
                self.index += 1
                return self.parse_object()
            # <array> starts with '['
            elif char == "[":
                self.index += 1
                return self.parse_array()
            # there can be an edge case in which a key is empty and at the end of an object
            # like "key": }. We return an empty string here to close the object properly
            elif self.context.current == ContextValues.OBJECT_VALUE and char == "}":
                self.log(
                    "At the end of an object we found a key with missing value, skipping",
                )
                return ""
            # <string> starts with a quote
            elif not self.context.empty and (
                char in self.STRING_DELIMITERS or char.isalpha()
            ):
                return self.parse_string()
            # <number> starts with [0-9] or minus
            elif not self.context.empty and (
                char.isdigit() or char == "-" or char == "."
            ):
                return self.parse_number()
            elif char in ["#", "/"]:
                return self.parse_comment()
            # If everything else fails, we just ignore and move on
            else:
                self.index += 1

    def parse_object(self) -> Dict[str, JSONReturnType]:
        # <object> ::= '{' [ <member> *(', ' <member>) ] '}' ; A sequence of 'members'
        obj = {}
        # Stop when you either find the closing parentheses or you have iterated over the entire string
        while (self.get_char_at() or "}") != "}":
            # This is what we expect to find:
            # <member> ::= <string> ': ' <json>

            # Skip filler whitespaces
            self.skip_whitespaces_at()

            # Sometimes LLMs do weird things, if we find a ":" so early, we'll change it to "," and move on
            if (self.get_char_at() or "") == ":":
                self.log(
                    "While parsing an object we found a : before a key, ignoring",
                )
                self.index += 1

            # We are now searching for they string key
            # Context is used in the string parser to manage the lack of quotes
            self.context.set(ContextValues.OBJECT_KEY)

            # Save this index in case we need find a duplicate key
            rollback_index = self.index

            # <member> starts with a <string>
            key = ""
            while self.get_char_at():
                # The rollback index needs to be updated here in case the key is empty
                rollback_index = self.index
                key = str(self.parse_string())
                if key == "":
                    self.skip_whitespaces_at()
                if key != "" or (key == "" and self.get_char_at() in [":", "}"]):
                    # If the string is empty but there is a object divider, we are done here
                    break
            if ContextValues.ARRAY in self.context.context and key in obj:
                self.log(
                    "While parsing an object we found a duplicate key, closing the object here and rolling back the index",
                )
                self.index = rollback_index - 1
                # add an opening curly brace to make this work
                self.json_str = (
                    self.json_str[: self.index + 1]
                    + "{"
                    + self.json_str[self.index + 1 :]
                )
                break

            # Skip filler whitespaces
            self.skip_whitespaces_at()

            # We reached the end here
            if (self.get_char_at() or "}") == "}":
                continue

            self.skip_whitespaces_at()

            # An extreme case of missing ":" after a key
            if (self.get_char_at() or "") != ":":
                self.log(
                    "While parsing an object we missed a : after a key",
                )

            self.index += 1
            self.context.reset()
            self.context.set(ContextValues.OBJECT_VALUE)
            # The value can be any valid json
            value = self.parse_json()

            # Reset context since our job is done
            self.context.reset()
            obj[key] = value

            if (self.get_char_at() or "") in [",", "'", '"']:
                self.index += 1

            # Remove trailing spaces
            self.skip_whitespaces_at()

        self.index += 1
        return obj

    def parse_array(self) -> List[JSONReturnType]:
        # <array> ::= '[' [ <json> *(', ' <json>) ] ']' ; A sequence of JSON values separated by commas
        arr = []
        self.context.set(ContextValues.ARRAY)
        # Stop when you either find the closing parentheses or you have iterated over the entire string
        char = self.get_char_at()
        while char and char not in ["]", "}"]:
            self.skip_whitespaces_at()
            value = self.parse_json()

            # It is possible that parse_json() returns nothing valid, so we increase by 1
            if value == "":
                self.index += 1
            elif value == "..." and self.get_char_at(-1) == ".":
                self.log(
                    "While parsing an array, found a stray '...'; ignoring it",
                )
            else:
                arr.append(value)

            # skip over whitespace after a value but before closing ]
            char = self.get_char_at()
            while char and (char.isspace() or char == ","):
                self.index += 1
                char = self.get_char_at()

        # Especially at the end of an LLM generated json you might miss the last "]"
        if char and char != "]":
            self.log(
                "While parsing an array we missed the closing ], ignoring it",
            )

        self.index += 1

        self.context.reset()
        return arr

    def parse_string(self) -> Union[str, bool, None]:
        # <string> is a string of valid characters enclosed in quotes
        # i.e. { name: "John" }
        # Somehow all weird cases in an invalid JSON happen to be resolved in this function, so be careful here

        # Flag to manage corner cases related to missing starting quote
        missing_quotes = False
        doubled_quotes = False
        lstring_delimiter = rstring_delimiter = '"'

        char = self.get_char_at()
        if char in ["#", "/"]:
            return self.parse_comment()
        # A valid string can only start with a valid quote or, in our case, with a literal
        while char and char not in self.STRING_DELIMITERS and not char.isalnum():
            self.index += 1
            char = self.get_char_at()

        if not char:
            # This is an empty string
            return ""

        # Ensuring we use the right delimiter
        if char == "'":
            lstring_delimiter = rstring_delimiter = "'"
        elif char == "“":
            lstring_delimiter = "“"
            rstring_delimiter = "”"
        elif char.isalnum():
            # This could be a <boolean> and not a string. Because (T)rue or (F)alse or (N)ull are valid
            # But remember, object keys are only of type string
            if (
                char.lower() in ["t", "f", "n"]
                and self.context.current != ContextValues.OBJECT_KEY
            ):
                value = self.parse_boolean_or_null()
                if value != "":
                    return value
            self.log(
                "While parsing a string, we found a literal instead of a quote",
            )
            missing_quotes = True

        if not missing_quotes:
            self.index += 1

        # There is sometimes a weird case of doubled quotes, we manage this also later in the while loop
        if self.get_char_at() in self.STRING_DELIMITERS:
            # If the next character is the same type of quote, then we manage it as double quotes
            if self.get_char_at() == lstring_delimiter:
                # If it's an empty key, this was easy
                if (
                    self.context.current == ContextValues.OBJECT_KEY
                    and self.get_char_at(1) == ":"
                ):
                    self.index += 1
                    return ""
                if self.get_char_at(1) == lstring_delimiter:
                    # There's something fishy about this, we found doubled quotes and then again quotes
                    self.log(
                        "While parsing a string, we found a doubled quote and then a quote again, ignoring it",
                    )
                    return ""
                # Find the next delimiter
                i = self.skip_to_character(character=rstring_delimiter, idx=1)
                next_c = self.get_char_at(i)
                # Now check that the next character is also a delimiter to ensure that we have "".....""
                # In that case we ignore this rstring delimiter
                if next_c and (self.get_char_at(i + 1) or "") == rstring_delimiter:
                    self.log(
                        "While parsing a string, we found a valid starting doubled quote",
                    )
                    doubled_quotes = True
                    self.index += 1
                else:
                    # Ok this is not a doubled quote, check if this is an empty string or not
                    i = self.skip_whitespaces_at(idx=1, move_main_index=False)
                    next_c = self.get_char_at(i)
                    if next_c in self.STRING_DELIMITERS + ["{", "["]:
                        # something fishy is going on here
                        self.log(
                            "While parsing a string, we found a doubled quote but also another quote afterwards, ignoring it",
                        )
                        self.index += 1
                        return ""
                    elif next_c not in [",", "]", "}"]:
                        self.log(
                            "While parsing a string, we found a doubled quote but it was a mistake, removing one quote",
                        )
                        self.index += 1
            else:
                # Otherwise we need to do another check before continuing
                i = self.skip_to_character(character=rstring_delimiter, idx=1)
                next_c = self.get_char_at(i)
                if not next_c:
                    # mmmm that delimiter never appears again, this is a mistake
                    self.log(
                        "While parsing a string, we found a quote but it was a mistake, ignoring it",
                    )
                    return ""

        # Initialize our return value
        string_acc = ""

        # Here things get a bit hairy because a string missing the final quote can also be a key or a value in an object
        # In that case we need to use the ":|,|}" characters as terminators of the string
        # So this will stop if:
        # * It finds a closing quote
        # * It iterated over the entire sequence
        # * If we are fixing missing quotes in an object, when it finds the special terminators
        char = self.get_char_at()
        unmatched_delimiter = False
        while char and char != rstring_delimiter:
            if (
                missing_quotes
                and self.context.current == ContextValues.OBJECT_KEY
                and (char == ":" or char.isspace())
            ):
                self.log(
                    "While parsing a string missing the left delimiter in object key context, we found a :, stopping here",
                )
                break
            if self.context.current == ContextValues.OBJECT_VALUE and char in [
                ",",
                "}",
            ]:
                rstring_delimiter_missing = True
                # check if this is a case in which the closing comma is NOT missing instead
                i = self.skip_to_character(character=rstring_delimiter, idx=1)
                next_c = self.get_char_at(i)
                if next_c:
                    i += 1
                    # found a delimiter, now we need to check that is followed strictly by a comma or brace
                    # or the string ended
                    i = self.skip_whitespaces_at(idx=i, move_main_index=False)
                    next_c = self.get_char_at(i)
                    if not next_c or next_c in [",", "}"]:
                        rstring_delimiter_missing = False
                    else:
                        # OK but this could still be some garbage at the end of the string
                        # So we need to check if we find a new lstring_delimiter afterwards
                        # If we do, maybe this is a missing delimiter
                        i = self.skip_to_character(character=lstring_delimiter, idx=i)
                        if doubled_quotes:
                            i = self.skip_to_character(
                                character=lstring_delimiter, idx=i
                            )
                        next_c = self.get_char_at(i)
                        if not next_c:
                            rstring_delimiter_missing = False
                        else:
                            # But again, this could just be something a bit stupid like "lorem, "ipsum" sic"
                            # Check if we find a : afterwards (skipping space)
                            i = self.skip_whitespaces_at(
                                idx=i + 1, move_main_index=False
                            )
                            next_c = self.get_char_at(i)
                            if next_c and next_c != ":":
                                rstring_delimiter_missing = False
                else:
                    # There could be a case in which even the next key:value is missing delimeters
                    # because it might be a systemic issue with the output
                    # So let's check if we can find a : in the string instead
                    i = self.skip_to_character(character=":", idx=1)
                    next_c = self.get_char_at(i)
                    if next_c:
                        # OK then this is a systemic issue with the output
                        break
                    else:
                        # skip any whitespace first
                        i = self.skip_whitespaces_at(idx=1, move_main_index=False)
                        # We couldn't find any rstring_delimeter before the end of the string
                        # check if this is the last string of an object and therefore we can keep going
                        # make an exception if this is the last char before the closing brace
                        j = self.skip_to_character(character="}", idx=i)
                        if j - i > 1:
                            # Ok it's not right after the comma
                            # Let's ignore
                            rstring_delimiter_missing = False
                        # Check that j was not out of bound
                        elif self.get_char_at(j):
                            # Check for an unmatched opening brace in string_acc
                            for c in reversed(string_acc):
                                if c == "{":
                                    # Ok then this is part of the string
                                    rstring_delimiter_missing = False
                                    break
                                elif c == "}":
                                    break
                if rstring_delimiter_missing:
                    self.log(
                        "While parsing a string missing the left delimiter in object value context, we found a , or } and we couldn't determine that a right delimiter was present. Stopping here",
                    )
                    break
            if char == "]" and ContextValues.ARRAY in self.context.context:
                # We found the end of an array and we are in array context
                # So let's check if we find a rstring_delimiter forward otherwise end early
                i = self.skip_to_character(rstring_delimiter)
                if not self.get_char_at(i):
                    # No delimiter found
                    break
            string_acc += char
            self.index += 1
            char = self.get_char_at()
            if char and string_acc[-1] == "\\":
                # This is a special case, if people use real strings this might happen
                self.log("Found a stray escape sequence, normalizing it")
                if char in [rstring_delimiter, "t", "n", "r", "b", "\\"]:
                    string_acc = string_acc[:-1]
                    escape_seqs = {"t": "\t", "n": "\n", "r": "\r", "b": "\b"}
                    string_acc += escape_seqs.get(char, char) or char
                    self.index += 1
                    char = self.get_char_at()
            # If we are in object key context and we find a colon, it could be a missing right quote
            if (
                char == ":"
                and not missing_quotes
                and self.context.current == ContextValues.OBJECT_KEY
            ):
                # Ok now we need to check if this is followed by a value like "..."
                i = self.skip_to_character(character=lstring_delimiter, idx=1)
                next_c = self.get_char_at(i)
                if next_c:
                    i += 1
                    # found the first delimiter
                    i = self.skip_to_character(character=rstring_delimiter, idx=i)
                    next_c = self.get_char_at(i)
                    if next_c:
                        # found a second delimiter
                        i += 1
                        # Skip spaces
                        i = self.skip_whitespaces_at(idx=i, move_main_index=False)
                        next_c = self.get_char_at(i)
                        if next_c and next_c in [",", "}"]:
                            # Ok then this is a missing right quote
                            self.log(
                                "While parsing a string missing the right delimiter in object key context, we found a :, stopping here",
                            )
                            break
                else:
                    # The string ended without finding a lstring_delimiter, I will assume this is a missing right quote
                    self.log(
                        "While parsing a string missing the right delimiter in object key context, we found a :, stopping here",
                    )
                    break
            # ChatGPT sometimes forget to quote stuff in html tags or markdown, so we do this whole thing here
            if char == rstring_delimiter:
                # Special case here, in case of double quotes one after another
                if doubled_quotes and self.get_char_at(1) == rstring_delimiter:
                    self.log(
                        "While parsing a string, we found a doubled quote, ignoring it"
                    )
                    self.index += 1
                elif (
                    missing_quotes
                    and self.context.current == ContextValues.OBJECT_VALUE
                ):
                    # In case of missing starting quote I need to check if the delimeter is the end or the beginning of a key
                    i = 1
                    next_c = self.get_char_at(i)
                    while next_c and next_c not in [
                        rstring_delimiter,
                        lstring_delimiter,
                    ]:
                        i += 1
                        next_c = self.get_char_at(i)
                    if next_c:
                        # We found a quote, now let's make sure there's a ":" following
                        i += 1
                        # found a delimiter, now we need to check that is followed strictly by a comma or brace
                        i = self.skip_whitespaces_at(idx=i, move_main_index=False)
                        next_c = self.get_char_at(i)
                        if next_c and next_c == ":":
                            # Reset the cursor
                            self.index -= 1
                            char = self.get_char_at()
                            self.log(
                                "In a string with missing quotes and object value context, I found a delimeter but it turns out it was the beginning on the next key. Stopping here.",
                            )
                            break
                elif unmatched_delimiter:
                    unmatched_delimiter = False
                    string_acc += str(char)
                    self.index += 1
                    char = self.get_char_at()
                else:
                    # Check if eventually there is a rstring delimiter, otherwise we bail
                    i = 1
                    next_c = self.get_char_at(i)
                    check_comma_in_object_value = True
                    while next_c and next_c not in [
                        rstring_delimiter,
                        lstring_delimiter,
                    ]:
                        # This is a bit of a weird workaround, essentially in object_value context we don't always break on commas
                        # This is because the routine after will make sure to correct any bad guess and this solves a corner case
                        if check_comma_in_object_value and next_c.isalpha():
                            check_comma_in_object_value = False
                        # If we are in an object context, let's check for the right delimiters
                        if (
                            (
                                ContextValues.OBJECT_KEY in self.context.context
                                and next_c in [":", "}"]
                            )
                            or (
                                ContextValues.OBJECT_VALUE in self.context.context
                                and next_c == "}"
                            )
                            or (
                                ContextValues.ARRAY in self.context.context
                                and next_c in ["]", ","]
                            )
                            or (
                                check_comma_in_object_value
                                and self.context.current == ContextValues.OBJECT_VALUE
                                and next_c == ","
                            )
                        ):
                            break
                        i += 1
                        next_c = self.get_char_at(i)
                    # If we stopped for a comma in object_value context, let's check if find a "} at the end of the string
                    if (
                        next_c == ","
                        and self.context.current == ContextValues.OBJECT_VALUE
                    ):
                        i += 1
                        i = self.skip_to_character(character=rstring_delimiter, idx=i)
                        next_c = self.get_char_at(i)
                        # Ok now I found a delimiter, let's skip whitespaces and see if next we find a }
                        i += 1
                        i = self.skip_whitespaces_at(idx=i, move_main_index=False)
                        next_c = self.get_char_at(i)
                        if next_c == "}":
                            # OK this is valid then
                            self.log(
                                "While parsing a string, we misplaced a quote that would have closed the string but has a different meaning here since this is the last element of the object, ignoring it",
                            )
                            unmatched_delimiter = not unmatched_delimiter
                            string_acc += str(char)
                            self.index += 1
                            char = self.get_char_at()
                    elif (
                        next_c == rstring_delimiter and self.get_char_at(i - 1) != "\\"
                    ):
                        # Check if self.index:self.index+i is only whitespaces, break if that's the case
                        if all(
                            str(self.get_char_at(j)).isspace()
                            for j in range(1, i)
                            if self.get_char_at(j)
                        ):
                            break
                        if self.context.current == ContextValues.OBJECT_VALUE:
                            # But this might not be it! This could be just a missing comma
                            # We found a delimiter and we need to check if this is a key
                            # so find a rstring_delimiter and a colon after
                            i = self.skip_to_character(
                                character=rstring_delimiter, idx=i + 1
                            )
                            i += 1
                            next_c = self.get_char_at(i)
                            while next_c and next_c != ":":
                                if next_c in [",", "]", "}"] or (
                                    next_c == rstring_delimiter
                                    and self.get_char_at(i - 1) != "\\"
                                ):
                                    break
                                i += 1
                                next_c = self.get_char_at(i)
                            # Only if we fail to find a ':' then we know this is misplaced quote
                            if next_c != ":":
                                self.log(
                                    "While parsing a string, we a misplaced quote that would have closed the string but has a different meaning here, ignoring it",
                                )
                                unmatched_delimiter = not unmatched_delimiter
                                string_acc += str(char)
                                self.index += 1
                                char = self.get_char_at()
                        elif self.context.current == ContextValues.ARRAY:
                            # If we got up to here it means that this is a situation like this:
                            # ["bla bla bla "puppy" bla bla bla "kitty" bla bla"]
                            # So we need to ignore this quote
                            self.log(
                                "While parsing a string in Array context, we detected a quoted section that would have closed the string but has a different meaning here, ignoring it",
                            )
                            unmatched_delimiter = not unmatched_delimiter
                            string_acc += str(char)
                            self.index += 1
                            char = self.get_char_at()

        if (
            char
            and missing_quotes
            and self.context.current == ContextValues.OBJECT_KEY
            and char.isspace()
        ):
            self.log(
                "While parsing a string, handling an extreme corner case in which the LLM added a comment instead of valid string, invalidate the string and return an empty value",
            )
            self.skip_whitespaces_at()
            if self.get_char_at() not in [":", ","]:
                return ""

        # A fallout of the previous special case in the while loop,
        # we need to update the index only if we had a closing quote
        if char != rstring_delimiter:
            self.log(
                "While parsing a string, we missed the closing quote, ignoring",
            )
            string_acc = string_acc.rstrip()
        else:
            self.index += 1

        if missing_quotes or (string_acc and string_acc[-1] == "\n"):
            # Clean the whitespaces for some corner cases
            string_acc = string_acc.rstrip()

        return string_acc

    def parse_number(self) -> Union[float, int, str, JSONReturnType]:
        # <number> is a valid real number expressed in one of a number of given formats
        number_str = ""
        char = self.get_char_at()
        is_array = self.context.current == ContextValues.ARRAY
        while char and char in self.NUMBER_CHARS and (not is_array or char != ","):
            number_str += char
            self.index += 1
            char = self.get_char_at()
        if number_str and number_str[-1] in "-eE/,":
            # The number ends with a non valid character for a number/currency, rolling back one
            number_str = number_str[:-1]
            self.index -= 1
        try:
            if "," in number_str:
                return str(number_str)
            if "." in number_str or "e" in number_str or "E" in number_str:
                return float(number_str)
            elif number_str == "-":
                # If there is a stray "-" this will throw an exception, throw away this character
                return self.parse_json()
            else:
                return int(number_str)
        except ValueError:
            return number_str

    def parse_boolean_or_null(self) -> Union[bool, str, None]:
        # <boolean> is one of the literal strings 'true', 'false', or 'null' (unquoted)
        starting_index = self.index
        char = (self.get_char_at() or "").lower()
        value: Optional[Tuple[str, Optional[bool]]]
        if char == "t":
            value = ("true", True)
        elif char == "f":
            value = ("false", False)
        elif char == "n":
            value = ("null", None)

        if value:
            i = 0
            while char and i < len(value[0]) and char == value[0][i]:
                i += 1
                self.index += 1
                char = (self.get_char_at() or "").lower()
            if i == len(value[0]):
                return value[1]

        # If nothing works reset the index before returning
        self.index = starting_index
        return ""

    def get_char_at(self, count: int = 0) -> Union[str, Literal[False]]:
        # Why not use something simpler? Because try/except in python is a faster alternative to an "if" statement that is often True
        try:
            return self.json_str[self.index + count]
        except IndexError:
            return False

    def skip_whitespaces_at(self, idx: int = 0, move_main_index=True) -> int:
        """
        This function quickly iterates on whitespaces, syntactic sugar to make the code more concise
        """
        try:
            char = self.json_str[self.index + idx]
        except IndexError:
            return idx
        while char.isspace():
            if move_main_index:
                self.index += 1
            else:
                idx += 1
            try:
                char = self.json_str[self.index + idx]
            except IndexError:
                return idx
        return idx

    def skip_to_character(self, character: str, idx: int = 0) -> int:
        """
        This function quickly iterates to find a character, syntactic sugar to make the code more concise
        """
        try:
            char = self.json_str[self.index + idx]
        except IndexError:
            return idx
        while char != character:
            idx += 1
            try:
                char = self.json_str[self.index + idx]
            except IndexError:
                return idx
        if self.index + idx > 0 and self.json_str[self.index + idx - 1] == "\\":
            # Ah this is an escaped character, try again
            return self.skip_to_character(character=character, idx=idx + 1)
        return idx

    def parse_comment(self) -> str:
        """
        Parse code-like comments:

        - "# comment": A line comment that continues until a newline.
        - "// comment": A line comment that continues until a newline.
        - "/* comment */": A block comment that continues until the closing delimiter "*/".

        The comment is skipped over and an empty string is returned so that comments do not interfere
        with the actual JSON elements.
        """
        char = self.get_char_at()
        termination_characters = ["\n", "\r"]
        if ContextValues.ARRAY in self.context.context:
            termination_characters.append("]")
        if ContextValues.OBJECT_VALUE in self.context.context:
            termination_characters.append("}")
        if ContextValues.OBJECT_KEY in self.context.context:
            termination_characters.append(":")
        # Line comment starting with #
        if char == "#":
            comment = ""
            while char and char not in termination_characters:
                comment += char
                self.index += 1
                char = self.get_char_at()
            self.log(f"Found line comment: {comment}")
            return ""

        # Comments starting with '/'
        elif char == "/":
            next_char = self.get_char_at(1)
            # Handle line comment starting with //
            if next_char == "/":
                comment = "//"
                self.index += 2  # Skip both slashes.
                char = self.get_char_at()
                while char and char not in termination_characters:
                    comment += char
                    self.index += 1
                    char = self.get_char_at()
                self.log(f"Found line comment: {comment}")
                return ""
            # Handle block comment starting with /*
            elif next_char == "*":
                comment = "/*"
                self.index += 2  # Skip '/*'
                while True:
                    char = self.get_char_at()
                    if not char:
                        self.log(
                            "Reached end-of-string while parsing block comment; unclosed block comment."
                        )
                        break
                    comment += char
                    self.index += 1
                    if comment.endswith("*/"):
                        break
                self.log(f"Found block comment: {comment}")
                return ""
            else:
                # Not a recognized comment pattern, skip the slash.
                self.index += 1
                return ""

        else:
            # Should not be reached: if for some reason the current character does not start a comment, skip it.
            self.index += 1
            return ""

    def _log(self, text: str) -> None:
        window: int = 10
        start: int = max(self.index - window, 0)
        end: int = min(self.index + window, len(self.json_str))
        context: str = self.json_str[start:end]
        self.logger.append(
            {
                "text": text,
                "context": context,
            }
        )
