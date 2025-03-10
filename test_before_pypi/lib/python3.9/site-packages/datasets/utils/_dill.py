# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Extends `dill` to support pickling more types and produce more consistent dumps."""

import os
import sys
from io import BytesIO
from types import CodeType, FunctionType

import dill
from packaging import version

from .. import config


class Pickler(dill.Pickler):
    dispatch = dill._dill.MetaCatchingDict(dill.Pickler.dispatch.copy())
    _legacy_no_dict_keys_sorting = False

    def save(self, obj, save_persistent_id=True):
        obj_type = type(obj)
        if obj_type not in self.dispatch:
            if "regex" in sys.modules:
                import regex  # type: ignore

                if obj_type is regex.Pattern:
                    pklregister(obj_type)(_save_regexPattern)
            if "spacy" in sys.modules:
                import spacy  # type: ignore

                if issubclass(obj_type, spacy.Language):
                    pklregister(obj_type)(_save_spacyLanguage)
            if "tiktoken" in sys.modules:
                import tiktoken  # type: ignore

                if obj_type is tiktoken.Encoding:
                    pklregister(obj_type)(_save_tiktokenEncoding)
            if "torch" in sys.modules:
                import torch  # type: ignore

                if issubclass(obj_type, torch.Tensor):
                    pklregister(obj_type)(_save_torchTensor)

                if obj_type is torch.Generator:
                    pklregister(obj_type)(_save_torchGenerator)

                # Unwrap `torch.compile`-ed modules
                if issubclass(obj_type, torch.nn.Module):
                    obj = getattr(obj, "_orig_mod", obj)
            if "transformers" in sys.modules:
                import transformers  # type: ignore

                if issubclass(obj_type, transformers.PreTrainedTokenizerBase):
                    pklregister(obj_type)(_save_transformersPreTrainedTokenizerBase)

        # Unwrap `torch.compile`-ed functions
        if obj_type is FunctionType:
            obj = getattr(obj, "_torchdynamo_orig_callable", obj)
        dill.Pickler.save(self, obj, save_persistent_id=save_persistent_id)

    def _batch_setitems(self, items):
        if self._legacy_no_dict_keys_sorting:
            return super()._batch_setitems(items)
        # Ignore the order of keys in a dict
        try:
            # Faster, but fails for unorderable elements
            items = sorted(items)
        except Exception:  # TypeError, decimal.InvalidOperation, etc.
            from datasets.fingerprint import Hasher

            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        dill.Pickler._batch_setitems(self, items)

    def memoize(self, obj):
        # Don't memoize strings since two identical strings can have different Python ids
        if type(obj) is not str:  # noqa: E721
            dill.Pickler.memoize(self, obj)


def pklregister(t):
    """Register a custom reducer for the type."""

    def proxy(func):
        Pickler.dispatch[t] = func
        return func

    return proxy


def dump(obj, file):
    """Pickle an object to a file."""
    Pickler(file, recurse=True).dump(obj)


def dumps(obj):
    """Pickle an object to a string."""
    file = BytesIO()
    dump(obj, file)
    return file.getvalue()


if config.DILL_VERSION < version.parse("0.3.6"):

    def log(pickler, msg):
        dill._dill.log.info(msg)

elif config.DILL_VERSION.release[:3] in [
    version.parse("0.3.6").release,
    version.parse("0.3.7").release,
    version.parse("0.3.8").release,
]:

    def log(pickler, msg):
        dill._dill.logger.trace(pickler, msg)


@pklregister(set)
def _save_set(pickler, obj):
    log(pickler, f"Se: {obj}")
    try:
        # Faster, but fails for unorderable elements
        args = (sorted(obj),)
    except Exception:  # TypeError, decimal.InvalidOperation, etc.
        from datasets.fingerprint import Hasher

        args = (sorted(obj, key=Hasher.hash),)

    pickler.save_reduce(set, args, obj=obj)
    log(pickler, "# Se")


def _save_regexPattern(pickler, obj):
    import regex  # type: ignore

    log(pickler, f"Re: {obj}")
    args = (obj.pattern, obj.flags)
    pickler.save_reduce(regex.compile, args, obj=obj)
    log(pickler, "# Re")


def _save_tiktokenEncoding(pickler, obj):
    import tiktoken  # type: ignore

    log(pickler, f"Enc: {obj}")
    args = (obj.name, obj._pat_str, obj._mergeable_ranks, obj._special_tokens)
    pickler.save_reduce(tiktoken.Encoding, args, obj=obj)
    log(pickler, "# Enc")


def _save_torchTensor(pickler, obj):
    import torch  # type: ignore

    # `torch.from_numpy` is not picklable in `torch>=1.11.0`
    def create_torchTensor(np_array, dtype=None):
        tensor = torch.from_numpy(np_array)
        if dtype:
            tensor = tensor.type(dtype)
        return tensor

    log(pickler, f"To: {obj}")
    if obj.dtype == torch.bfloat16:
        args = (obj.detach().to(torch.float).cpu().numpy(), torch.bfloat16)
    else:
        args = (obj.detach().cpu().numpy(),)
    pickler.save_reduce(create_torchTensor, args, obj=obj)
    log(pickler, "# To")


def _save_torchGenerator(pickler, obj):
    import torch  # type: ignore

    def create_torchGenerator(state):
        generator = torch.Generator()
        generator.set_state(state)
        return generator

    log(pickler, f"Ge: {obj}")
    args = (obj.get_state(),)
    pickler.save_reduce(create_torchGenerator, args, obj=obj)
    log(pickler, "# Ge")


def _save_spacyLanguage(pickler, obj):
    import spacy  # type: ignore

    def create_spacyLanguage(config, bytes):
        lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
        lang_inst = lang_cls.from_config(config)
        return lang_inst.from_bytes(bytes)

    log(pickler, f"Sp: {obj}")
    args = (obj.config, obj.to_bytes())
    pickler.save_reduce(create_spacyLanguage, args, obj=obj)
    log(pickler, "# Sp")


def _save_transformersPreTrainedTokenizerBase(pickler, obj):
    log(pickler, f"Tok: {obj}")
    # Ignore the `cache` attribute
    state = obj.__dict__
    if "cache" in state and isinstance(state["cache"], dict):
        state["cache"] = {}
    pickler.save_reduce(type(obj), (), state=state, obj=obj)
    log(pickler, "# Tok")


if config.DILL_VERSION < version.parse("0.3.6"):

    @pklregister(CodeType)
    def _save_code(pickler, obj):
        """
        From dill._dill.save_code
        This is a modified version that removes the origin (filename + line no.)
        of functions created in notebooks or shells for example.
        """
        dill._dill.log.info(f"Co: {obj}")
        # The filename of a function is the .py file where it is defined.
        # Filenames of functions created in notebooks or shells start with '<'
        # ex: <ipython-input-13-9ed2afe61d25> for ipython, and <stdin> for shell
        # Filenames of functions created in ipykernel the filename
        # look like f"{tempdir}/ipykernel_{id1}/{id2}.py"
        # Moreover lambda functions have a special name: '<lambda>'
        # ex: (lambda x: x).__code__.co_name == "<lambda>"  # True
        #
        # For the hashing mechanism we ignore where the function has been defined
        # More specifically:
        # - we ignore the filename of special functions (filename starts with '<')
        # - we always ignore the line number
        # - we only use the base name of the file instead of the whole path,
        # to be robust in case a script is moved for example.
        #
        # Only those two lines are different from the original implementation:
        co_filename = (
            ""
            if obj.co_filename.startswith("<")
            or (
                len(obj.co_filename.split(os.path.sep)) > 1
                and obj.co_filename.split(os.path.sep)[-2].startswith("ipykernel_")
            )
            or obj.co_name == "<lambda>"
            else os.path.basename(obj.co_filename)
        )
        co_firstlineno = 1
        # The rest is the same as in the original dill implementation
        if dill._dill.PY3:
            if hasattr(obj, "co_posonlyargcount"):
                args = (
                    obj.co_argcount,
                    obj.co_posonlyargcount,
                    obj.co_kwonlyargcount,
                    obj.co_nlocals,
                    obj.co_stacksize,
                    obj.co_flags,
                    obj.co_code,
                    obj.co_consts,
                    obj.co_names,
                    obj.co_varnames,
                    co_filename,
                    obj.co_name,
                    co_firstlineno,
                    obj.co_lnotab,
                    obj.co_freevars,
                    obj.co_cellvars,
                )
            else:
                args = (
                    obj.co_argcount,
                    obj.co_kwonlyargcount,
                    obj.co_nlocals,
                    obj.co_stacksize,
                    obj.co_flags,
                    obj.co_code,
                    obj.co_consts,
                    obj.co_names,
                    obj.co_varnames,
                    co_filename,
                    obj.co_name,
                    co_firstlineno,
                    obj.co_lnotab,
                    obj.co_freevars,
                    obj.co_cellvars,
                )
        else:
            args = (
                obj.co_argcount,
                obj.co_nlocals,
                obj.co_stacksize,
                obj.co_flags,
                obj.co_code,
                obj.co_consts,
                obj.co_names,
                obj.co_varnames,
                co_filename,
                obj.co_name,
                co_firstlineno,
                obj.co_lnotab,
                obj.co_freevars,
                obj.co_cellvars,
            )
        pickler.save_reduce(CodeType, args, obj=obj)
        dill._dill.log.info("# Co")
        return

elif config.DILL_VERSION.release[:3] in [
    version.parse("0.3.6").release,
    version.parse("0.3.7").release,
    version.parse("0.3.8").release,
]:
    # From: https://github.com/uqfoundation/dill/blob/dill-0.3.6/dill/_dill.py#L1104
    @pklregister(CodeType)
    def save_code(pickler, obj):
        dill._dill.logger.trace(pickler, "Co: %s", obj)

        ############################################################################################################
        # Modification here for huggingface/datasets
        # The filename of a function is the .py file where it is defined.
        # Filenames of functions created in notebooks or shells start with '<'
        # ex: <ipython-input-13-9ed2afe61d25> for ipython, and <stdin> for shell
        # Filenames of functions created in ipykernel the filename
        # look like f"{tempdir}/ipykernel_{id1}/{id2}.py"
        # Moreover lambda functions have a special name: '<lambda>'
        # ex: (lambda x: x).__code__.co_name == "<lambda>"  # True
        #
        # For the hashing mechanism we ignore where the function has been defined
        # More specifically:
        # - we ignore the filename of special functions (filename starts with '<')
        # - we always ignore the line number
        # - we only use the base name of the file instead of the whole path,
        # to be robust in case a script is moved for example.
        #
        # Only those two lines are different from the original implementation:
        co_filename = (
            ""
            if obj.co_filename.startswith("<")
            or (
                len(obj.co_filename.split(os.path.sep)) > 1
                and obj.co_filename.split(os.path.sep)[-2].startswith("ipykernel_")
            )
            or obj.co_name == "<lambda>"
            else os.path.basename(obj.co_filename)
        )
        co_firstlineno = 1
        # The rest is the same as in the original dill implementation, except for the replacements:
        # - obj.co_filename => co_filename
        # - obj.co_firstlineno => co_firstlineno
        ############################################################################################################

        if hasattr(obj, "co_endlinetable"):  # python 3.11a (20 args)
            args = (
                obj.co_lnotab,  # for < python 3.10 [not counted in args]
                obj.co_argcount,
                obj.co_posonlyargcount,
                obj.co_kwonlyargcount,
                obj.co_nlocals,
                obj.co_stacksize,
                obj.co_flags,
                obj.co_code,
                obj.co_consts,
                obj.co_names,
                obj.co_varnames,
                co_filename,  # Modification for huggingface/datasets ############################################
                obj.co_name,
                obj.co_qualname,
                co_firstlineno,  # Modification for huggingface/datasets #########################################
                obj.co_linetable,
                obj.co_endlinetable,
                obj.co_columntable,
                obj.co_exceptiontable,
                obj.co_freevars,
                obj.co_cellvars,
            )
        elif hasattr(obj, "co_exceptiontable"):  # python 3.11 (18 args)
            args = (
                obj.co_lnotab,  # for < python 3.10 [not counted in args]
                obj.co_argcount,
                obj.co_posonlyargcount,
                obj.co_kwonlyargcount,
                obj.co_nlocals,
                obj.co_stacksize,
                obj.co_flags,
                obj.co_code,
                obj.co_consts,
                obj.co_names,
                obj.co_varnames,
                co_filename,  # Modification for huggingface/datasets ############################################
                obj.co_name,
                obj.co_qualname,
                co_firstlineno,  # Modification for huggingface/datasets #########################################
                obj.co_linetable,
                obj.co_exceptiontable,
                obj.co_freevars,
                obj.co_cellvars,
            )
        elif hasattr(obj, "co_linetable"):  # python 3.10 (16 args)
            args = (
                obj.co_lnotab,  # for < python 3.10 [not counted in args]
                obj.co_argcount,
                obj.co_posonlyargcount,
                obj.co_kwonlyargcount,
                obj.co_nlocals,
                obj.co_stacksize,
                obj.co_flags,
                obj.co_code,
                obj.co_consts,
                obj.co_names,
                obj.co_varnames,
                co_filename,  # Modification for huggingface/datasets ############################################
                obj.co_name,
                co_firstlineno,  # Modification for huggingface/datasets #########################################
                obj.co_linetable,
                obj.co_freevars,
                obj.co_cellvars,
            )
        elif hasattr(obj, "co_posonlyargcount"):  # python 3.8 (16 args)
            args = (
                obj.co_argcount,
                obj.co_posonlyargcount,
                obj.co_kwonlyargcount,
                obj.co_nlocals,
                obj.co_stacksize,
                obj.co_flags,
                obj.co_code,
                obj.co_consts,
                obj.co_names,
                obj.co_varnames,
                co_filename,  # Modification for huggingface/datasets ############################################
                obj.co_name,
                co_firstlineno,  # Modification for huggingface/datasets #########################################
                obj.co_lnotab,
                obj.co_freevars,
                obj.co_cellvars,
            )
        else:  # python 3.7 (15 args)
            args = (
                obj.co_argcount,
                obj.co_kwonlyargcount,
                obj.co_nlocals,
                obj.co_stacksize,
                obj.co_flags,
                obj.co_code,
                obj.co_consts,
                obj.co_names,
                obj.co_varnames,
                co_filename,  # Modification for huggingface/datasets ############################################
                obj.co_name,
                co_firstlineno,  # Modification for huggingface/datasets #########################################
                obj.co_lnotab,
                obj.co_freevars,
                obj.co_cellvars,
            )

        pickler.save_reduce(dill._dill._create_code, args, obj=obj)
        dill._dill.logger.trace(pickler, "# Co")
        return
