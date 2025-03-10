import io
import json
from itertools import islice
from typing import Any, Callable, Dict, List

import fsspec
import numpy as np
import pyarrow as pa

import datasets
from datasets.features.features import cast_to_python_objects
from datasets.utils.file_utils import SINGLE_FILE_COMPRESSION_EXTENSION_TO_PROTOCOL, xbasename


logger = datasets.utils.logging.get_logger(__name__)


class WebDataset(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 100
    IMAGE_EXTENSIONS: List[str]  # definition at the bottom of the script
    AUDIO_EXTENSIONS: List[str]  # definition at the bottom of the script
    DECODERS: Dict[str, Callable[[Any], Any]]  # definition at the bottom of the script
    NUM_EXAMPLES_FOR_FEATURES_INFERENCE = 5

    @classmethod
    def _get_pipeline_from_tar(cls, tar_path, tar_iterator):
        current_example = {}
        fs: fsspec.AbstractFileSystem = fsspec.filesystem("memory")
        streaming_download_manager = datasets.StreamingDownloadManager()
        for filename, f in tar_iterator:
            if "." in filename:
                example_key, field_name = filename.split(".", 1)
                if current_example and current_example["__key__"] != example_key:
                    yield current_example
                    current_example = {}
                current_example["__key__"] = example_key
                current_example["__url__"] = tar_path
                current_example[field_name.lower()] = f.read()
                if field_name.split(".")[-1] in SINGLE_FILE_COMPRESSION_EXTENSION_TO_PROTOCOL:
                    fs.write_bytes(filename, current_example[field_name.lower()])
                    extracted_file_path = streaming_download_manager.extract(f"memory://{filename}")
                    with fsspec.open(extracted_file_path) as f:
                        current_example[field_name.lower()] = f.read()
                    fs.delete(filename)
                    data_extension = xbasename(extracted_file_path).split(".")[-1]
                else:
                    data_extension = field_name.split(".")[-1]
                if data_extension in cls.DECODERS:
                    current_example[field_name] = cls.DECODERS[data_extension](current_example[field_name])
        if current_example:
            yield current_example

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo()

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        # Download the data files
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        data_files = dl_manager.download(self.config.data_files)
        splits = []
        for split_name, tar_paths in data_files.items():
            if isinstance(tar_paths, str):
                tar_paths = [tar_paths]
            tar_iterators = [dl_manager.iter_archive(tar_path) for tar_path in tar_paths]
            splits.append(
                datasets.SplitGenerator(
                    name=split_name, gen_kwargs={"tar_paths": tar_paths, "tar_iterators": tar_iterators}
                )
            )
        if not self.info.features:
            # Get one example to get the feature types
            pipeline = self._get_pipeline_from_tar(tar_paths[0], tar_iterators[0])
            first_examples = list(islice(pipeline, self.NUM_EXAMPLES_FOR_FEATURES_INFERENCE))
            if any(example.keys() != first_examples[0].keys() for example in first_examples):
                raise ValueError(
                    "The TAR archives of the dataset should be in WebDataset format, "
                    "but the files in the archive don't share the same prefix or the same types."
                )
            pa_tables = [
                pa.Table.from_pylist(cast_to_python_objects([example], only_1d_for_numpy=True))
                for example in first_examples
            ]
            inferred_arrow_schema = pa.concat_tables(pa_tables, promote_options="default").schema
            features = datasets.Features.from_arrow_schema(inferred_arrow_schema)

            # Set Image types
            for field_name in first_examples[0]:
                extension = field_name.rsplit(".", 1)[-1]
                if extension in self.IMAGE_EXTENSIONS:
                    features[field_name] = datasets.Image()
            # Set Audio types
            for field_name in first_examples[0]:
                extension = field_name.rsplit(".", 1)[-1]
                if extension in self.AUDIO_EXTENSIONS:
                    features[field_name] = datasets.Audio()
            self.info.features = features

        return splits

    def _generate_examples(self, tar_paths, tar_iterators):
        image_field_names = [
            field_name for field_name, feature in self.info.features.items() if isinstance(feature, datasets.Image)
        ]
        audio_field_names = [
            field_name for field_name, feature in self.info.features.items() if isinstance(feature, datasets.Audio)
        ]
        all_field_names = list(self.info.features.keys())
        for tar_idx, (tar_path, tar_iterator) in enumerate(zip(tar_paths, tar_iterators)):
            for example_idx, example in enumerate(self._get_pipeline_from_tar(tar_path, tar_iterator)):
                for field_name in all_field_names:
                    if field_name not in example:
                        example[field_name] = None
                for field_name in image_field_names + audio_field_names:
                    if example[field_name] is not None:
                        example[field_name] = {
                            "path": example["__key__"] + "." + field_name,
                            "bytes": example[field_name],
                        }
                yield f"{tar_idx}_{example_idx}", example


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
    "blp",
    "bmp",
    "dib",
    "bufr",
    "cur",
    "pcx",
    "dcx",
    "dds",
    "ps",
    "eps",
    "fit",
    "fits",
    "fli",
    "flc",
    "ftc",
    "ftu",
    "gbr",
    "gif",
    "grib",
    "h5",
    "hdf",
    "png",
    "apng",
    "jp2",
    "j2k",
    "jpc",
    "jpf",
    "jpx",
    "j2c",
    "icns",
    "ico",
    "im",
    "iim",
    "tif",
    "tiff",
    "jfif",
    "jpe",
    "jpg",
    "jpeg",
    "mpg",
    "mpeg",
    "msp",
    "pcd",
    "pxr",
    "pbm",
    "pgm",
    "ppm",
    "pnm",
    "psd",
    "bw",
    "rgb",
    "rgba",
    "sgi",
    "ras",
    "tga",
    "icb",
    "vda",
    "vst",
    "webp",
    "wmf",
    "emf",
    "xbm",
    "xpm",
]
WebDataset.IMAGE_EXTENSIONS = IMAGE_EXTENSIONS


# Obtained with:
# ```
# import soundfile as sf
#
# AUDIO_EXTENSIONS = [f".{format.lower()}" for format in sf.available_formats().keys()]
#
# # .opus decoding is supported if libsndfile >= 1.0.31:
# AUDIO_EXTENSIONS.extend([".mp3", ".opus"])
# ```
# We intentionally do not run this code on launch because:
# (1) Soundfile is an optional dependency, so importing it in global namespace is not allowed
# (2) To ensure the list of supported extensions is deterministic
AUDIO_EXTENSIONS = [
    "aiff",
    "au",
    "avr",
    "caf",
    "flac",
    "htk",
    "svx",
    "mat4",
    "mat5",
    "mpc2k",
    "ogg",
    "paf",
    "pvf",
    "raw",
    "rf64",
    "sd2",
    "sds",
    "ircam",
    "voc",
    "w64",
    "wav",
    "nist",
    "wavex",
    "wve",
    "xi",
    "mp3",
    "opus",
]
WebDataset.AUDIO_EXTENSIONS = AUDIO_EXTENSIONS


def text_loads(data: bytes):
    return data.decode("utf-8")


def tenbin_loads(data: bytes):
    from . import _tenbin

    return _tenbin.decode_buffer(data)


def msgpack_loads(data: bytes):
    import msgpack

    return msgpack.unpackb(data)


def npy_loads(data: bytes):
    import numpy.lib.format

    stream = io.BytesIO(data)
    return numpy.lib.format.read_array(stream, allow_pickle=False)


def npz_loads(data: bytes):
    return np.load(io.BytesIO(data), allow_pickle=False)


def cbor_loads(data: bytes):
    import cbor

    return cbor.loads(data)


def torch_loads(data: bytes):
    import torch

    return torch.load(io.BytesIO(data), weights_only=True)


# Obtained by checking `decoders` in `webdataset.autodecode`
# and removing unsafe extension decoders.
# Removed Pickle decoders:
# - "pyd": lambda data: pickle.loads(data)
# - "pickle": lambda data: pickle.loads(data)
# Modified NumPy decoders to fix CVE-2019-6446 (add allow_pickle=False and weights_only=True):
# - "npy": npy_loads,
# - "npz": lambda data: np.load(io.BytesIO(data)),
# - "pth": lambda data: torch_loads(data)
DECODERS = {
    "txt": text_loads,
    "text": text_loads,
    "transcript": text_loads,
    "cls": int,
    "cls2": int,
    "index": int,
    "inx": int,
    "id": int,
    "json": json.loads,
    "jsn": json.loads,
    "ten": tenbin_loads,
    "tb": tenbin_loads,
    "mp": msgpack_loads,
    "msg": msgpack_loads,
    "npy": npy_loads,
    "npz": npz_loads,
    "cbor": cbor_loads,
    "pth": torch_loads,
}
WebDataset.DECODERS = DECODERS
