# Detailed Plan for Upgrading the `XMLAdapter`

This document provides a detailed, function-by-function plan to refactor `dspy.adapters.xml_adapter.XMLAdapter`. The goal is to add robust support for nested data structures (e.g., nested Pydantic models), lists, and attributes, bringing its capabilities in line with the `JSONAdapter`.

## 1. Analysis of the Current `XMLAdapter`

The current implementation is limited to flat key-value pairs and will fail on complex signatures.

*   **`__init__(self, ...)`**: Initializes a regex pattern `self.field_pattern`. This pattern is the root cause of the limitations, as it cannot correctly capture nested XML structures. It will greedily match from the first opening tag to the very last closing tag of the same name, treating the entire inner content as a single string.
*   **`format_field_with_value(self, ...)`**: This function formats few-shot examples. It uses `format_field_value`, which for a nested object (like a Pydantic model), serializes it into a JSON string. This results in incorrect examples like `<person>{"name": "John"}</person>`, teaching the Language Model the wrong format.
*   **`user_message_output_requirements(self, ...)`**: This method generates instructions for the LM. It is not recursive and only lists the top-level output fields (e.g., `<person_info>`), failing to describe the required inner structure (e.g., `<name>` and `<age>`).
*   **`parse(self, ...)`**: This is the primary failure point for parsing. It uses the flawed regex to find all top-level tags. It cannot handle nested data and will fail to build a correct dictionary, leading to a downstream `AdapterParseError`.
*   **`_parse_field_value(self, ...)`**: This helper relies on `dspy.adapters.utils.parse_value`, which is designed for Python literals and JSON, not XML. It will be made redundant by a proper XML parsing workflow.

---

## 2. Proposed Refactoring Plan

We will replace the regex-based logic with a proper XML parsing engine (`xml.etree.ElementTree`) and recursive helpers for both serialization and deserialization.

### New Helper Function 1: `_dict_to_xml`

This will be a new private method responsible for serializing Python objects into XML strings.

*   **Objective**: Recursively convert a Python object (dict, list, Pydantic model) into a well-formed, indented XML string.
*   **Signature**: `_dict_to_xml(self, data: Any, parent_tag: str) -> str`
*   **Input**:
    *   `data`: The Python object to serialize (e.g., `{"person": {"name": "John", "aliases": ["Johnny", "J-man"]}}`).
    *   `parent_tag`: The name of the root XML tag for the current data.
*   **Output**: A well-formed XML string (e.g., `<person><name>John</name><aliases>Johnny</aliases><aliases>J-man</aliases></person>`).
*   **Implementation Details**:
    1.  If `data` is a `pydantic.BaseModel`, convert it to a dictionary using `.model_dump()`.
    2.  If `data` is a dictionary, iterate through its items. For each `key, value` pair, recursively call `_dict_to_xml(value, key)`.
    3.  If `data` is a list, iterate through its items. For each `item`, recursively call `_dict_to_xml(item, parent_tag)`. This correctly creates repeated tags (e.g., `<aliases>...</aliases><aliases>...</aliases>`).
    4.  If `data` is a primitive type (str, int, float, bool), wrap its string representation in the `parent_tag` (e.g., `<name>John</name>`).
    5.  Combine the results into a single, properly indented XML string.

### New Helper Function 2: `_xml_to_dict`

This will be the core recursive engine for converting a parsed XML structure into a Python dictionary.

*   **Objective**: Recursively convert an `xml.etree.ElementTree.Element` object into a nested Python dictionary.
*   **Signature**: `_xml_to_dict(self, element: ElementTree.Element) -> Any`
*   **Input**: An `ElementTree.Element` object.
*   **Output**: A nested Python dictionary or a string.
*   **Implementation Details**:
    1.  Initialize a dictionary, `d`.
    2.  Iterate through the direct children of the `element`.
    3.  For each `child` element, recursively call `_xml_to_dict(child)`.
    4.  When adding the result to `d`:
        *   If the `child.tag` is not already in `d`, add it as `d[child.tag] = result`.
        *   If `child.tag` is already in `d` and the value is *not* a list, convert it to a list: `d[child.tag] = [d[child.tag], result]`.
        *   If `child.tag` is already in `d` and the value *is* a list, append to it: `d[child.tag].append(result)`.
    5.  If `d` is empty after checking all children, it means the element is a leaf node. Return its `element.text`.
    6.  Otherwise, return the dictionary `d`.

### Function-by-Function Changes

#### 1. `format_field_with_value`

*   **Objective**: Generate correct, nested XML for few-shot examples.
*   **Input**: `fields_with_values: Dict[FieldInfoWithName, Any]`
*   **Output**: A string containing well-formed XML for all fields.
*   **New Implementation**:
    1.  Iterate through the `fields_with_values` dictionary.
    2.  For each `field, field_value` pair, call the new `_dict_to_xml(field_value, field.name)` helper.
    3.  Join the resulting XML strings with newlines.

#### 2. `user_message_output_requirements`

*   **Objective**: Generate a prompt that clearly describes the expected nested XML output structure.
*   **Input**: `signature: Type[Signature]`
*   **Output**: A descriptive string.
*   **New Implementation**:
    1.  This function will also need a recursive helper, `_generate_schema_description(field_info)`.
    2.  The helper will check if a field's type is a Pydantic model.
    3.  If it is, it will recursively traverse the model's fields, building a string that looks like a sample XML structure (e.g., `<person_info><name>...</name><age>...</age></person_info>`).
    4.  The main function will call this helper for all output fields and assemble the final instruction string.

#### 3. `parse`

*   **Objective**: Parse an LM's string completion into a structured Python dictionary that matches the signature.
*   **Input**: `signature: Type[Signature]`, `completion: str`
*   **Output**: `dict[str, Any]`
*   **New Implementation**:
    1.  Import `xml.etree.ElementTree` and `pydantic.TypeAdapter`.
    2.  Wrap the parsing logic in a `try...except ElementTree.ParseError` block to handle malformed XML from the LM.
    3.  Find the root XML tag in the `completion` string. The root tag should correspond to the single output field name if there's only one, or a generic "output" tag if there are multiple.
    4.  Parse the relevant part of the completion string into an `ElementTree` object: `root = ElementTree.fromstring(xml_string)`.
    5.  Call the new `_xml_to_dict(root)` helper to get a nested Python dictionary.
    6.  **Crucially**, use Pydantic's `TypeAdapter` to validate and cast the dictionary. This replaces the manual `_parse_field_value` logic entirely.
        *   `validated_data = TypeAdapter(signature).validate_python({"output_field_name": result_dict})`
    7.  Return the validated data.

#### 4. `_parse_field_value`

*   **Action**: **Delete this method.** Its functionality is now fully handled by the `_xml_to_dict` helper and the `pydantic.TypeAdapter` validation step within the `parse` method.

---

## 3. Dummy Run-Through

Let's trace the `Extraction` signature from our discussion.

**Signature:**
```python
class Person(pydantic.BaseModel):
    name: str
    age: int

class Extraction(dspy.Signature):
    person_info = dspy.OutputField(type=Person)
```

**Scenario:** `dspy.Predict(Extraction, adapter=XMLAdapter())` is used.

**1. Prompt Generation (Formatting a few-shot example):**
*   An example output is `{"person_info": Person(name="Jane Doe", age=42)}`.
*   `format_field_with_value` is called with this data.
*   It calls `_dict_to_xml(Person(name="Jane Doe", age=42), "person_info")`.
*   `_dict_to_xml` converts the `Person` object to `{"name": "Jane Doe", "age": 42}`.
*   It then recursively processes this dict, producing the **correct** XML string:
    ```xml
    <person_info>
      <name>Jane Doe</name>
      <age>42</age>
    </person_info>
    ```
*   This correct example is added to the prompt.

**2. LM Response & Parsing:**
*   The LM sees the correct example and produces its own valid, nested XML string in its completion.
*   `parse(signature=Extraction, completion=LM_OUTPUT_STRING)` is called.
*   `parse` finds the `<person_info>...</person_info>` block in the completion.
*   `ElementTree.fromstring()` turns this into an XML element tree.
*   `_xml_to_dict()` is called on the root element. It recursively processes the nodes and returns the Python dictionary: `{'name': 'John Smith', 'age': '28'}`.
*   The `parse` method then wraps this in the field name: `{'person_info': {'name': 'John Smith', 'age': '28'}}`.
*   `TypeAdapter(Extraction).validate_python(...)` is called. Pydantic handles the validation, sees that `person_info` should be a `Person` object, and automatically casts the string `'28'` to the integer `28`.
*   The final, validated output is `{"person_info": Person(name="John Smith", age=28)}`.

**Conclusion:** This plan systematically replaces the brittle regex logic with a robust, recursive, and industry-standard approach. It correctly handles nested data, lists, and type casting, making the `XMLAdapter` a reliable and powerful component of the DSPy ecosystem.

---

## 4. Testing Strategy

A robust testing strategy is crucial to validate the refactoring. The existing test file `tests/adapters/test_xml_adapter.py` provides an excellent starting point, including tests designed to fail with the current implementation and pass with the new one.

### Analysis of Existing Tests

*   **Keep:** The basic tests for flat structures (`test_xml_adapter_format_and_parse_basic`, `test_xml_adapter_parse_multiple_fields`, etc.) must continue to pass.
*   **Target for Success:** The primary goal is to make the "failing" tests pass:
    *   `test_xml_adapter_handles_true_nested_xml_parsing`
    *   `test_xml_adapter_formats_true_nested_xml`
    *   `test_xml_adapter_handles_lists_as_repeated_tags`
*   **Remove:** The tests that verify the old behavior of embedding JSON inside XML tags (`test_xml_adapter_format_and_parse_nested_model`, `test_xml_adapter_format_and_parse_list_of_models`) will become obsolete and must be removed.

### New Tests to Be Added

To ensure comprehensive coverage, the following new tests will be created:

1.  **`test_parse_malformed_xml`**:
    *   **Objective**: Ensure the `parse` method raises a `dspy.utils.exceptions.AdapterParseError` when the LM provides broken or incomplete XML.
    *   **Input**: A string like `<root><child>text</root>`.
    *   **Expected Outcome**: `pytest.raises(AdapterParseError)`.

2.  **`test_format_and_parse_deeply_nested_model`**:
    *   **Objective**: Verify that the recursive helpers can handle more than one level of nesting.
    *   **Signature**: A signature with a Pydantic model that contains another Pydantic model.
    *   **Assertions**: Check that both the formatted XML string and the parsed object correctly represent the deep nesting.

3.  **`test_format_and_parse_empty_list`**:
    *   **Objective**: Ensure that formatting an empty list results in a clean parent tag and that parsing it results in an empty list.
    *   **Signature**: `items: list[str] = dspy.OutputField()`
    *   **Data**: `{"items": []}`
    *   **Assertions**:
        *   `format_field_with_value` produces `<items></items>`.
        *   `parse` on `<items></items>` produces `{"items": []}`.

4.  **`test_end_to_end_with_predict`**:
    *   **Objective**: Confirm the adapter works correctly within the full `dspy` ecosystem.
    *   **Implementation**:
        *   Define a signature with a nested Pydantic model.
        *   Create a `dspy.Predict` module with this signature and `adapter=XMLAdapter()`.
        *   Use a `dspy.testing.MockLM` to provide a pre-defined, correctly formatted XML string as the completion.
        *   Assert that the output of the `Predict` module is the correct, parsed Pydantic object.
