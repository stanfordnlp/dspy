import dspy
import random

from datasets import Dataset
from tqdm import tqdm, trange
from typing import List, Union, Optional, Mapping

from .signatures import (
    ExplainTask,
    GenerateFieldDescription,
    GenerateInputFieldsData,
    GenerateOutputFieldsData,
    UnderstandTask,
)
from .config import SynthesizerArguments
from .instructions import INPUT_GENERATION_TASK_WITH_EXAMPLES
from .utils import format_examples

__all__ = ["Synthesizer"]

class Synthesizer:
    def __init__(self, config: SynthesizerArguments):
        self.config = config
        self.input_lm = config.input_lm_model or dspy.settings.lm
        self.output_lm = config.output_lm_model or dspy.settings.lm

        self.explain_task = dspy.Predict(ExplainTask)
        self.understand_task = dspy.Predict(UnderstandTask)
        self.generate_field_description = dspy.Predict(GenerateFieldDescription)

        self.generate_input_data = GenerateInputFieldsData
        self.generate_output_data = GenerateOutputFieldsData

    def _get_field_data(self, key: str, keys_dict: Mapping[str, str]):
        if key.startswith("$"):
            field_details = self.generate_field_description(
                task_description=keys_dict["task_description"],
                field_name=key,
            )

            field_name = key
            field_description = field_details.field_description

            return field_name, field_description

        else:
            field_name = key
            field_description = keys_dict[key]

            return field_name, field_description

    def _prepare_synthetic_data_predictors(
        self,
        input_keys: Mapping[str, str],
        output_keys: Mapping[str, str],
        ground_source: Optional[Union[List[dspy.Example], dspy.Signature]] = None,
    ):
        for key in tqdm(input_keys, desc="Preparing Input Fields"):
            field_name, field_description = self._get_field_data(key, input_keys)

            output_field = dspy.OutputField(
                prefix=f"{field_name}:",
                desc=field_description,
            )
            self.generate_input_data = self.generate_input_data.insert(
                -1,
                field_name,
                output_field,
            )

            if ground_source:
                self.generate_input_data = self.generate_input_data.insert(
                    -1,
                    "ground_source",
                    dspy.InputField(
                        prefix=f"Pre-Generated Examples:",
                        desc="Pre-Generated Examples to differ the inputs around.",
                        format=format_examples,
                    ),
                )

            input_field = dspy.InputField(
                prefix=f"{field_name}:",
                desc=field_description,
            )
            self.generate_output_data = self.generate_output_data.insert(
                -1,
                field_name,
                input_field,
            )

        for key in tqdm(output_keys, desc="Preparing Output Fields"):
            field_name, field_description = self._get_field_data(key, output_keys)

            output_field = dspy.OutputField(
                prefix=f"{field_name}:",
                desc=field_description,
            )
            self.generate_output_data = self.generate_output_data.insert(
                -1,
                field_name,
                output_field,
            )

        return dspy.ChainOfThought(self.generate_input_data), dspy.Predict(self.generate_output_data)

    def _get_dataset_metadata(self, ground_source: Union[List[dspy.Example], dspy.Signature]):
        if isinstance(ground_source, dspy.SignatureMeta):
            task_description = ground_source.__doc__
            if task_description.startswith("Given the fields"):
                task_description = self.understand_task(examples=ground_source.__doc__).explanation
            
            input_keys = {k:v.json_schema_extra["desc"] for k,v in ground_source.input_fields.items()}
            output_keys = {k:v.json_schema_extra["desc"] for k,v in ground_source.output_fields.items()}

            return task_description, input_keys, output_keys

        elif isinstance(ground_source, list) and isinstance(ground_source[0], dspy.Example):
            task_description = self.explain_task(examples=ground_source).explanation
            input_keys = {key:f"${{{key}}}" for key in ground_source[0].inputs()}
            output_keys = {key:f"${{{key}}}" for key in ground_source[0].labels()}

            return task_description, input_keys, output_keys

        else:
            raise ValueError("Ground source must be either a list of examples or a signature.")

    def generate(
        self,
        ground_source: Union[List[dspy.Example], dspy.Signature],
        num_data: int,
        batch_size: int = 1,
    ):
        batch_size = batch_size or 1
        task_description, input_keys, output_keys = self._get_dataset_metadata(ground_source)

        if self.config.num_example_for_optim:
            self.generate_input_data.__doc__ = INPUT_GENERATION_TASK_WITH_EXAMPLES
        self.generate_output_data.__doc__ = task_description

        self.input_predictor, self.output_predictor = self._prepare_synthetic_data_predictors(
            input_keys=input_keys,
            output_keys=output_keys,
            ground_source=ground_source if self.config.num_example_for_optim else None,
        )

        data = []

        for idx in trange(0, num_data, batch_size, desc="Generating Synthetic Data"):
            iter_temperature = 0.7+0.01*idx
            iter_seed = random.randint(0, 1000000)

            inputs = None

            with dspy.context(lm=self.input_lm):
                if self.config.num_example_for_optim:
                    example_for_optimization = random.sample(ground_source, self.config.num_example_for_optim)
                    inputs = self.input_predictor(
                        task_description=task_description,
                        knowledge_seed=iter_seed,
                        ground_source=example_for_optimization,
                        config=dict(temperature=iter_temperature, n=batch_size)
                    )
                else:
                    inputs = self.input_predictor(
                        task_description=task_description,
                        knowledge_seed=iter_seed,
                        config=dict(temperature=iter_temperature, n=batch_size)
                    )

            input_kwargs = [{
                key: getattr(completions, key)
                for key in input_keys
            } for completions in inputs.completions]

            for kwargs in input_kwargs:
                outputs = None

                with dspy.context(lm=self.output_lm, temperature=iter_temperature):
                    if self.config.output_teacher_module:
                        outputs = self.config.output_teacher_module(**kwargs)

                    else:
                        outputs = self.output_predictor(**kwargs, config=dict(temperature=iter_temperature))

                output_kwargs = {
                    key: getattr(outputs, key)
                    for key in output_keys
                }

                data.append(dspy.Example(**kwargs, **output_kwargs).with_inputs(*input_keys))

        return data

    def export(self, data: List[dspy.Example], path: str, mode: str = None, **kwargs):
        extention = mode or path.split(".")[-1]

        dataset = Dataset.from_list(
            [example.toDict() for example in data],
        )

        if extention == "csv":
            dataset.to_csv(path_or_buf=path, **kwargs)

        elif extention == "json":
            dataset.to_json(path_or_buf=path, **kwargs)

        elif extention == "arrow" or extention == "hf":
            dataset.save_to_disk(path)
