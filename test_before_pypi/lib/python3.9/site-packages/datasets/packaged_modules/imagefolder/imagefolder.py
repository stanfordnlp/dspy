from typing import List

import datasets
from datasets.tasks import ImageClassification

from ..folder_based_builder import folder_based_builder


logger = datasets.utils.logging.get_logger(__name__)


class ImageFolderConfig(folder_based_builder.FolderBasedBuilderConfig):
    """BuilderConfig for ImageFolder."""

    drop_labels: bool = None
    drop_metadata: bool = None

    def __post_init__(self):
        super().__post_init__()


class ImageFolder(folder_based_builder.FolderBasedBuilder):
    BASE_FEATURE = datasets.Image
    BASE_COLUMN_NAME = "image"
    BUILDER_CONFIG_CLASS = ImageFolderConfig
    EXTENSIONS: List[str]  # definition at the bottom of the script
    CLASSIFICATION_TASK = ImageClassification(image_column="image", label_column="label")


# Obtained with:
# ```
# import PIL.Image
# IMAGE_EXTENSIONS = []
# PIL.Image.init()
# for ext, format in PIL.Image.EXTENSION.items():
#     if format in PIL.Image.OPEN:
#         IMAGE_EXTENSIONS.append(ext[1:])
# ```
# We intentionally do not run this code on launch because:
# (1) Pillow is an optional dependency, so importing Pillow in global namespace is not allowed
# (2) To ensure the list of supported extensions is deterministic
IMAGE_EXTENSIONS = [
    ".blp",
    ".bmp",
    ".dib",
    ".bufr",
    ".cur",
    ".pcx",
    ".dcx",
    ".dds",
    ".ps",
    ".eps",
    ".fit",
    ".fits",
    ".fli",
    ".flc",
    ".ftc",
    ".ftu",
    ".gbr",
    ".gif",
    ".grib",
    ".h5",
    ".hdf",
    ".png",
    ".apng",
    ".jp2",
    ".j2k",
    ".jpc",
    ".jpf",
    ".jpx",
    ".j2c",
    ".icns",
    ".ico",
    ".im",
    ".iim",
    ".tif",
    ".tiff",
    ".jfif",
    ".jpe",
    ".jpg",
    ".jpeg",
    ".mpg",
    ".mpeg",
    ".msp",
    ".pcd",
    ".pxr",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pnm",
    ".psd",
    ".bw",
    ".rgb",
    ".rgba",
    ".sgi",
    ".ras",
    ".tga",
    ".icb",
    ".vda",
    ".vst",
    ".webp",
    ".wmf",
    ".emf",
    ".xbm",
    ".xpm",
]
ImageFolder.EXTENSIONS = IMAGE_EXTENSIONS
