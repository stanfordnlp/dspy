# Modules (detailed)

This documentation provides an overview of the DSPy Modules.

## DSPy Modules

| Module | Description | Use Case |
| --- | --- | --- |
| [ChainOfThought](<./ChainOfThought.md>) | | |
| [ChainOfThoughtWithHint](ChainOfThoughtWithHint.md) | | |
| [MultiChainComparison](MultiChainComparison.md) | | |
| [Predict](Predict.md) | | |
| [ReAct](ReAct.md) | | |
| [Retrieve](Retrieve.md) | | |


```{toctree}
:hidden:
---
maxdepth: 1
---

ChainOfThought <ChainOfThought.md>
ChainOfThoughtWithHint <ChainOfThoughtWithHint.md>
MultiChainComparison <MultiChainComparison.md>
Predict <Predict.md>
ReAct <ReAct.md>
Retrieve <Retrieve.md>
```



#TODO: figure out if we should keep this / if so where

This handler is used to ignore assertion failures and return None.

#### `suggest_backtrack_handler(func, bypass_suggest=True, max_backtracks=2)`

This handler is used for backtracking suggestions. It re-runs the latest predictor up to `max_backtracks` times, with updated signature if a suggestion fails.

#### `handle_assert_forward(assertion_handler, **handler_args)`

This function is used to handle assertions. It wraps the `forward` method of a module with an assertion handler.

#### `assert_transform_module(module, assertion_handler=default_assertion_handler, **handler_args)`

```py
            inputs, outputs = inputs.split(","), outputs.split(",")
            inputs, outputs = [field.strip() for field in inputs], [field.strip() for field in outputs]

            assert all(len(field.split()) == 1 for field in (inputs + outputs))

            inputs_ = ', '.join([f"`{field}`" for field in inputs])
            outputs_ = ', '.join([f"`{field}`" for field in outputs])

            instructions = f"""Given the fields {inputs_}, produce the fields {outputs_}."""

            inputs = {k: InputField() for k in inputs}
            outputs = {k: OutputField() for k in outputs}

            for k, v in inputs.items():
                v.finalize(k, infer_prefix(k))
            
            for k, v in outputs.items():
                v.finalize(k, infer_prefix(k))

            self.signature = dsp.Template(instructions, **inputs, **outputs)
```

