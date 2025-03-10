# Copyright 2020 The HuggingFace Datasets Authors and the TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import tqdm as _tqdm  # _tqdm is the module
from .experimental import experimental
from .info_utils import VerificationMode
from .logging import disable_progress_bar, enable_progress_bar, is_progress_bar_enabled
from .tqdm import (
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
    tqdm,
)
from .version import Version
