from typing import List

import datasets
from datasets.tasks import AudioClassification

from ..folder_based_builder import folder_based_builder


logger = datasets.utils.logging.get_logger(__name__)


class AudioFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """Builder Config for AudioFolder."""

    drop_labels: bool = None
    drop_metadata: bool = None

    def __post_init__(self):
        super().__post_init__()


class AudioFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Audio
    BASE_COLUMN_NAME = "audio"
    BUILDER_CONFIG_CLASS = AudioFolderConfig
    EXTENSIONS: List[str]  # definition at the bottom of the script
    CLASSIFICATION_TASK = AudioClassification(audio_column="audio", label_column="label")


# Obtained with:
# ```
# import soundfile as sf
#
# AUDIO_EXTENSIONS = [f".{format.lower()}" for format in sf.available_formats().keys()]
#
# # .opus decoding is supported if libsndfile >= 1.0.31:
# AUDIO_EXTENSIONS.extend([".opus"])
# ```
# We intentionally do not run this code on launch because:
# (1) Soundfile is an optional dependency, so importing it in global namespace is not allowed
# (2) To ensure the list of supported extensions is deterministic
AUDIO_EXTENSIONS = [
    ".aiff",
    ".au",
    ".avr",
    ".caf",
    ".flac",
    ".htk",
    ".svx",
    ".mat4",
    ".mat5",
    ".mpc2k",
    ".ogg",
    ".paf",
    ".pvf",
    ".raw",
    ".rf64",
    ".sd2",
    ".sds",
    ".ircam",
    ".voc",
    ".w64",
    ".wav",
    ".nist",
    ".wavex",
    ".wve",
    ".xi",
    ".mp3",
    ".opus",
]
AudioFolder.EXTENSIONS = AUDIO_EXTENSIONS
