from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

import pyarrow as pa


if TYPE_CHECKING:
    from .features import FeatureType


@dataclass
class Translation:
    """`FeatureConnector` for translations with fixed languages per example.
    Here for compatiblity with tfds.

    Args:
        languages (`dict`):
            A dictionary for each example mapping string language codes to string translations.

    Example:

    ```python
    >>> # At construction time:
    >>> datasets.features.Translation(languages=['en', 'fr', 'de'])
    >>> # During data generation:
    >>> yield {
    ...         'en': 'the cat',
    ...         'fr': 'le chat',
    ...         'de': 'die katze'
    ... }
    ```
    """

    languages: List[str]
    id: Optional[str] = None
    # Automatically constructed
    dtype: ClassVar[str] = "dict"
    pa_type: ClassVar[Any] = None
    _type: str = field(default="Translation", init=False, repr=False)

    def __call__(self):
        return pa.struct({lang: pa.string() for lang in sorted(self.languages)})

    def flatten(self) -> Union["FeatureType", Dict[str, "FeatureType"]]:
        """Flatten the Translation feature into a dictionary."""
        from .features import Value

        return {k: Value("string") for k in sorted(self.languages)}


@dataclass
class TranslationVariableLanguages:
    """`FeatureConnector` for translations with variable languages per example.
    Here for compatiblity with tfds.

    Args:
        languages (`dict`):
            A dictionary for each example mapping string language codes to one or more string translations.
            The languages present may vary from example to example.

    Returns:
        - `language` or `translation` (variable-length 1D `tf.Tensor` of `tf.string`):
            Language codes sorted in ascending order or plain text translations, sorted to align with language codes.

    Example:

    ```python
    >>> # At construction time:
    >>> datasets.features.TranslationVariableLanguages(languages=['en', 'fr', 'de'])
    >>> # During data generation:
    >>> yield {
    ...         'en': 'the cat',
    ...         'fr': ['le chat', 'la chatte,']
    ...         'de': 'die katze'
    ... }
    >>> # Tensor returned :
    >>> {
    ...         'language': ['en', 'de', 'fr', 'fr'],
    ...         'translation': ['the cat', 'die katze', 'la chatte', 'le chat'],
    ... }
    ```
    """

    languages: Optional[List] = None
    num_languages: Optional[int] = None
    id: Optional[str] = None
    # Automatically constructed
    dtype: ClassVar[str] = "dict"
    pa_type: ClassVar[Any] = None
    _type: str = field(default="TranslationVariableLanguages", init=False, repr=False)

    def __post_init__(self):
        self.languages = sorted(set(self.languages)) if self.languages else None
        self.num_languages = len(self.languages) if self.languages else None

    def __call__(self):
        return pa.struct({"language": pa.list_(pa.string()), "translation": pa.list_(pa.string())})

    def encode_example(self, translation_dict):
        lang_set = set(self.languages)
        if set(translation_dict) == {"language", "translation"}:
            return translation_dict
        elif self.languages and set(translation_dict) - lang_set:
            raise ValueError(
                f'Some languages in example ({", ".join(sorted(set(translation_dict) - lang_set))}) are not in valid set ({", ".join(lang_set)}).'
            )

        # Convert dictionary into tuples, splitting out cases where there are
        # multiple translations for a single language.
        translation_tuples = []
        for lang, text in translation_dict.items():
            if isinstance(text, str):
                translation_tuples.append((lang, text))
            else:
                translation_tuples.extend([(lang, el) for el in text])

        # Ensure translations are in ascending order by language code.
        languages, translations = zip(*sorted(translation_tuples))

        return {"language": languages, "translation": translations}

    def flatten(self) -> Union["FeatureType", Dict[str, "FeatureType"]]:
        """Flatten the TranslationVariableLanguages feature into a dictionary."""
        from .features import Sequence, Value

        return {
            "language": Sequence(Value("string")),
            "translation": Sequence(Value("string")),
        }
