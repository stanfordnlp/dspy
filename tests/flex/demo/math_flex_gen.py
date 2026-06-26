"""Implementation of a dspy.Flex module -- managed automatically.

This file starts life as a baseline that delegates to dspy.RLM, and is rewritten in place
by dspy.GEPA when you optimize the module (decomposing the task into predictors and Python code). 
It is a normal, runnable dspy.Module.

- You may edit the module class between the __FLEX_MODULE_BEGIN__/__FLEX_MODULE_END__ markers;
  on the next run dspy.Flex parses that class back out and runs your code as-is.
- __FLEX_SIGNATURE__ records the Signature this module was flexed from (for you and for GEPA).
- __FLEX_SIGNATURE_HASH__ guards against stale code: if you change the Signature, the hash no
  longer matches and dspy.Flex regenerates the baseline for the new Signature (re-run dspy.GEPA
  to re-optimize).

Leave the marker comments and the signature-hash line intact.
"""

# __FLEX_SIGNATURE_HASH__: 1dc5932bd8951ce6faebb94b26138071f5af510ddee4f09c01a0883c3a2b483b

# __FLEX_SIGNATURE_BEGIN__
# Signature: MathWord
# Objective (docstring): Solve a problem that I won't tell you about
# Input fields:
#   - idk: str
# Output fields:
#   - answer: int
# __FLEX_SIGNATURE_END__

import dspy


# __FLEX_MODULE_BEGIN__
class MathWordModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.rlm = dspy.RLM(dspy.Signature("idk: str -> answer: int", "Solve a problem that I won't tell you about"))

    def forward(self, **inputs):
        result = self.rlm(**inputs)
        return dspy.Prediction(answer=result.answer)
# __FLEX_MODULE_END__
