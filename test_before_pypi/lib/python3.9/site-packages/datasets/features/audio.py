import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Union

import numpy as np
import pyarrow as pa

from .. import config
from ..download.download_config import DownloadConfig
from ..table import array_cast
from ..utils.file_utils import xopen, xsplitext
from ..utils.py_utils import no_op_if_value_is_null, string_to_dict


if TYPE_CHECKING:
    from .features import FeatureType


@dataclass
class Audio:
    """Audio [`Feature`] to extract audio data from an audio file.

    Input: The Audio feature accepts as input:
    - A `str`: Absolute path to the audio file (i.e. random access is allowed).
    - A `dict` with the keys:

        - `path`: String with relative path of the audio file to the archive file.
        - `bytes`: Bytes content of the audio file.

      This is useful for archived files with sequential access.

    - A `dict` with the keys:

        - `path`: String with relative path of the audio file to the archive file.
        - `array`: Array containing the audio sample
        - `sampling_rate`: Integer corresponding to the sampling rate of the audio sample.

      This is useful for archived files with sequential access.

    Args:
        sampling_rate (`int`, *optional*):
            Target sampling rate. If `None`, the native sampling rate is used.
        mono (`bool`, defaults to `True`):
            Whether to convert the audio signal to mono by averaging samples across
            channels.
        decode (`bool`, defaults to `True`):
            Whether to decode the audio data. If `False`,
            returns the underlying dictionary in the format `{"path": audio_path, "bytes": audio_bytes}`.

    Example:

    ```py
    >>> from datasets import load_dataset, Audio
    >>> ds = load_dataset("PolyAI/minds14", name="en-US", split="train")
    >>> ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    >>> ds[0]["audio"]
    {'array': array([ 2.3443763e-05,  2.1729663e-04,  2.2145823e-04, ...,
         3.8356509e-05, -7.3497440e-06, -2.1754686e-05], dtype=float32),
     'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~JOINT_ACCOUNT/602ba55abb1e6d0fbce92065.wav',
     'sampling_rate': 16000}
    ```
    """

    sampling_rate: Optional[int] = None
    mono: bool = True
    decode: bool = True
    id: Optional[str] = None
    # Automatically constructed
    dtype: ClassVar[str] = "dict"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="Audio", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value: Union[str, bytes, dict]) -> dict:
        """Encode example into a format for Arrow.

        Args:
            value (`str` or `dict`):
                Data passed as input to Audio feature.

        Returns:
            `dict`
        """
        try:
            import soundfile as sf  # soundfile is a dependency of librosa, needed to decode audio files.
        except ImportError as err:
            raise ImportError("To support encoding audio data, please install 'soundfile'.") from err
        if isinstance(value, str):
            return {"bytes": None, "path": value}
        elif isinstance(value, bytes):
            return {"bytes": value, "path": None}
        elif "array" in value:
            # convert the audio array to wav bytes
            buffer = BytesIO()
            sf.write(buffer, value["array"], value["sampling_rate"], format="wav")
            return {"bytes": buffer.getvalue(), "path": None}
        elif value.get("path") is not None and os.path.isfile(value["path"]):
            # we set "bytes": None to not duplicate the data if they're already available locally
            if value["path"].endswith("pcm"):
                # "PCM" only has raw audio bytes
                if value.get("sampling_rate") is None:
                    # At least, If you want to convert "PCM-byte" to "WAV-byte", you have to know sampling rate
                    raise KeyError("To use PCM files, please specify a 'sampling_rate' in Audio object")
                if value.get("bytes"):
                    # If we already had PCM-byte, we don`t have to make "read file, make bytes" (just use it!)
                    bytes_value = np.frombuffer(value["bytes"], dtype=np.int16).astype(np.float32) / 32767
                else:
                    bytes_value = np.memmap(value["path"], dtype="h", mode="r").astype(np.float32) / 32767

                buffer = BytesIO(bytes())
                sf.write(buffer, bytes_value, value["sampling_rate"], format="wav")
                return {"bytes": buffer.getvalue(), "path": None}
            else:
                return {"bytes": None, "path": value.get("path")}
        elif value.get("bytes") is not None or value.get("path") is not None:
            # store the audio bytes, and path is used to infer the audio format using the file extension
            return {"bytes": value.get("bytes"), "path": value.get("path")}
        else:
            raise ValueError(
                f"An audio sample should have one of 'path' or 'bytes' but they are missing or None in {value}."
            )

    def decode_example(
        self, value: dict, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]] = None
    ) -> dict:
        """Decode example audio file into audio data.

        Args:
            value (`dict`):
                A dictionary with keys:

                - `path`: String with relative audio file path.
                - `bytes`: Bytes of the audio file.
            token_per_repo_id (`dict`, *optional*):
                To access and decode
                audio files from private repositories on the Hub, you can pass
                a dictionary repo_id (`str`) -> token (`bool` or `str`)

        Returns:
            `dict`
        """
        if not self.decode:
            raise RuntimeError("Decoding is disabled for this feature. Please use Audio(decode=True) instead.")

        path, file = (value["path"], BytesIO(value["bytes"])) if value["bytes"] is not None else (value["path"], None)
        if path is None and file is None:
            raise ValueError(f"An audio sample should have one of 'path' or 'bytes' but both are None in {value}.")

        try:
            import librosa
            import soundfile as sf
        except ImportError as err:
            raise ImportError("To support decoding audio files, please install 'librosa' and 'soundfile'.") from err

        audio_format = xsplitext(path)[1][1:].lower() if path is not None else None
        if not config.IS_OPUS_SUPPORTED and audio_format == "opus":
            raise RuntimeError(
                "Decoding 'opus' files requires system library 'libsndfile'>=1.0.31, "
                'You can try to update `soundfile` python library: `pip install "soundfile>=0.12.1"`. '
            )
        elif not config.IS_MP3_SUPPORTED and audio_format == "mp3":
            raise RuntimeError(
                "Decoding 'mp3' files requires system library 'libsndfile'>=1.1.0, "
                'You can try to update `soundfile` python library: `pip install "soundfile>=0.12.1"`. '
            )

        if file is None:
            token_per_repo_id = token_per_repo_id or {}
            source_url = path.split("::")[-1]
            pattern = (
                config.HUB_DATASETS_URL if source_url.startswith(config.HF_ENDPOINT) else config.HUB_DATASETS_HFFS_URL
            )
            try:
                repo_id = string_to_dict(source_url, pattern)["repo_id"]
                token = token_per_repo_id[repo_id]
            except (ValueError, KeyError):
                token = None

            download_config = DownloadConfig(token=token)
            with xopen(path, "rb", download_config=download_config) as f:
                array, sampling_rate = sf.read(f)

        else:
            array, sampling_rate = sf.read(file)

        array = array.T
        if self.mono:
            array = librosa.to_mono(array)
        if self.sampling_rate and self.sampling_rate != sampling_rate:
            array = librosa.resample(array, orig_sr=sampling_rate, target_sr=self.sampling_rate)
            sampling_rate = self.sampling_rate

        return {"path": path, "array": array, "sampling_rate": sampling_rate}

    def flatten(self) -> Union["FeatureType", Dict[str, "FeatureType"]]:
        """If in the decodable state, raise an error, otherwise flatten the feature into a dictionary."""
        from .features import Value

        if self.decode:
            raise ValueError("Cannot flatten a decoded Audio feature.")
        return {
            "bytes": Value("binary"),
            "path": Value("string"),
        }

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray]) -> pa.StructArray:
        """Cast an Arrow array to the Audio arrow storage type.
        The Arrow types that can be converted to the Audio pyarrow storage type are:

        - `pa.string()` - it must contain the "path" data
        - `pa.binary()` - it must contain the audio bytes
        - `pa.struct({"bytes": pa.binary()})`
        - `pa.struct({"path": pa.string()})`
        - `pa.struct({"bytes": pa.binary(), "path": pa.string()})`  - order doesn't matter

        Args:
            storage (`Union[pa.StringArray, pa.StructArray]`):
                PyArrow array to cast.

        Returns:
            `pa.StructArray`: Array in the Audio arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`
        """
        if pa.types.is_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
            storage = pa.StructArray.from_arrays([bytes_array, storage], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([storage, path_array], ["bytes", "path"], mask=storage.is_null())
        elif pa.types.is_struct(storage.type) and storage.type.get_all_field_indices("array"):
            storage = pa.array([Audio().encode_example(x) if x is not None else None for x in storage.to_pylist()])
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=storage.is_null())
        return array_cast(storage, self.pa_type)

    def embed_storage(self, storage: pa.StructArray) -> pa.StructArray:
        """Embed audio files into the Arrow array.

        Args:
            storage (`pa.StructArray`):
                PyArrow array to embed.

        Returns:
            `pa.StructArray`: Array in the Audio arrow storage type, that is
                `pa.struct({"bytes": pa.binary(), "path": pa.string()})`.
        """

        @no_op_if_value_is_null
        def path_to_bytes(path):
            with xopen(path, "rb") as f:
                bytes_ = f.read()
            return bytes_

        bytes_array = pa.array(
            [
                (path_to_bytes(x["path"]) if x["bytes"] is None else x["bytes"]) if x is not None else None
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        path_array = pa.array(
            [os.path.basename(path) if path is not None else None for path in storage.field("path").to_pylist()],
            type=pa.string(),
        )
        storage = pa.StructArray.from_arrays([bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null())
        return array_cast(storage, self.pa_type)
