# Plan for Enhancing the XMLAdapter

This document outlines the plan to add support for nested XML, attributes, and robust parsing to the `dspy.adapters.xml_adapter.XMLAdapter`.

## 1. Error-Prone Functions in the Current Implementation

The current `XMLAdapter` is designed for flat key-value structures and will fail when used with signatures that define nested fields (e.g., using nested Pydantic models or other Signatures).

*   **`parse(self, signature, completion)`**
    *   **Problem:** It uses a regular expression (`self.field_pattern`) that cannot handle nested XML tags. It will incorrectly capture an entire nested block as a single string value.
    *   **Failure Point:** The adapter will fail to build a correct nested dictionary from the LM's output, causing a downstream `AdapterParseError` when the structure doesn't match the signature.

*   **`format_field_with_value(self, fields_with_values)`**
    *   **Problem:** This method is responsible for formatting few-shot examples. When it encounters a nested object (like a Pydantic model), it calls `format_field_value`, which serializes the object into a JSON string, not a nested XML string.
    *   **Failure Point:** This will generate incorrect few-shot examples (e.g., `<person>{"name": "John"}</person>`), teaching the language model the wrong output format and leading to unpredictable behavior.

*   **`user_message_output_requirements(self, signature)`**
    *   **Problem:** This method generates the part of the prompt that tells the LM what to output. It is not recursive and only lists the top-level output fields.
    *   **Failure Point:** It provides incomplete instructions for nested signatures, failing to describe the required inner structure of the nested fields.

*   **`_parse_field_value(self, field_info, raw, ...)`**
    *   **Problem:** This helper function relies on `dspy.adapters.utils.parse_value`, which is designed to parse Python literals and JSON strings, not XML strings.
    *   **Failure Point:** This function will fail when it receives a string containing XML tags from the main `parse` method, as it has no logic to interpret XML.

## 2. Required Fixes

To achieve full functionality, we need to replace the current flat-structure logic with recursive, XML-aware logic.

*   **`parse`:** Replace the regex-based parsing with a proper XML parsing engine.
    *   **Must-have:** Use Python's built-in `xml.etree.ElementTree` to parse the completion string into a tree structure.
    *   **Must-have:** Implement a recursive helper function (`_xml_to_dict`) to convert the `ElementTree` object into a nested Python dictionary. This function must correctly handle repeated tags (creating lists) and text content.
    *   **Nice-to-have:** Capture XML attributes and represent them in the dictionary (e.g., using a special key like `@attributes`).

*   **`format_field_with_value`:** Replace the current formatting with a recursive XML generator.
    *   **Must-have:** Implement a recursive helper function (`_dict_to_xml`) that takes a Python dictionary/object and builds a well-formed, nested XML string. This function must handle lists by creating repeated tags.
    *   **Nice-to-have:** Add logic to serialize dictionary keys that represent attributes into proper XML attributes.

*   **`user_message_output_requirements`:** This method needs to be made recursive.
    *   **Must-have:** It should traverse the signature's fields, and if a field is a nested model/signature, it should recursively describe the inner fields.

*   **`_parse_field_value`:** This function will become redundant.
    *   **Must-have:** The new `parse` method will handle all parsing and type validation directly, so this helper should be removed.

## 3. My Plan to Implement the Fixes

I will implement the changes in a logical order, starting with the core parsing logic.

1.  **Refactor `parse()` for Nested Input:**
    *   Import `xml.etree.ElementTree`.
    *   In the `parse` method, wrap the call to `ElementTree.fromstring(completion)` in a `try...except ElementTree.ParseError` block to gracefully handle malformed XML from the LM.
    *   Create a new private helper method, `_xml_to_dict(self, element)`, which will be the recursive engine for converting an XML element to a dictionary.
    *   The `_xml_to_dict` logic will handle:
        *   Text content.
        *   Child elements (recursive call).
        *   Repeated child elements (which will be aggregated into a list).
    *   The main `parse` method will call this helper on the root element to get the final dictionary.
    *   Finally, I will use Pydantic's `TypeAdapter(signature).validate_python(result_dict)` to validate the structure and cast the values to their proper Python types, ensuring the output matches the signature.

2.  **Refactor `format_field_with_value()` for Nested Output:**
    *   Create a new private helper method, `_dict_to_xml(self, data)`, which will recursively build an XML string from a Python dictionary.
    *   This helper will iterate through the dictionary items. If a value is a list, it will create repeated tags. If a value is another dictionary, it will make a recursive call.
    *   Update `format_field_with_value` to use this new helper, ensuring few-shot examples are formatted correctly.

3.  **Update Instructions and Cleanup:**
    *   Refactor `user_message_output_requirements` to recursively generate a more descriptive prompt for nested structures.
    *   Remove the now-redundant `_parse_field_value` method.

4.  **Add Comprehensive Unit Tests:**
    *   Create a new test file in `tests/adapters/` for the `XMLAdapter`.
    *   Add tests for:
        *   Parsing a simple, flat XML string.
        *   Parsing a nested XML string.
        *   Parsing XML with repeated tags (lists).
        *   Formatting a few-shot example with a nested signature.
        *   An end-to-end test with a `dspy.Predict` module using the `XMLAdapter` and a nested signature.
