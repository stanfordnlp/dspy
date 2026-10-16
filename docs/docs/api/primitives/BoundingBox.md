# dspy.BoundingBox

`BoundingBox` is a `dspy.Type` for an axis-aligned rectangle. Its four required
fields are `top`, `bottom`, `left`, and `right`: finite numbers in [0, 1], measured
from the image's top-left corner. `top < bottom` and `left < right` are required.
Invalid geometry raises a validation error; DSPy does not clamp or reorder it.

```python
import dspy
from PIL import Image

class Locate(dspy.Signature):
    photo: dspy.Image = dspy.InputField()
    visible: bool = dspy.OutputField(desc="Is there a coffee mug in inputs.photo?")
    mug: dspy.BoundingBox = dspy.OutputField(desc="The coffee mug in inputs.photo.")

lm = dspy.LM("typesafe:jev-1.14.0", engine="lm15")  # TYPESAFE_API_KEY
predict = dspy.Predict(Locate)
try:
    with dspy.context(lm=lm, adapter=dspy.TypeSafeAdapter()):
        result = predict(photo=dspy.Image.from_path("kitchen.jpg"))
        # Or: result = await predict.acall(photo=...)

    if result.visible:
        box = result.mug
        # width/height must describe the upright image actually sent.
        with Image.open("kitchen.jpg") as photo:
            width, height = photo.size
        # Pixel order for PIL.crop: left, top, right, bottom.
        pixels = (round(box.left * width), round(box.top * height),
                  round(box.right * width), round(box.bottom * height))

    response = lm.history[-1]["response"]
    confidence = response.provider_data["typesafe"]["answers"]["mug"]["confidence"]
    presence_probability = response.probabilities["visible"]["true"]
finally:
    lm.close()  # await lm.aclose() for async use
```

Confidence is provider evidence, not a coordinate or a probability distribution
over boxes. Boxes do not appear in `response.probabilities`. History access is
appropriate for sequential debugging; for concurrent applications that need
per-call evidence, use the typed `dspy.lm15.Request`/`Response` API or LM callbacks
rather than reading a shared history's last entry. History must be enabled.

## TypeSafe's experimental contract

- Images and Box questions are experimental. lm15 sends the required
  `TypeSafe-Danger-Zone-Images: true` header automatically. Use Jev 1.14
  (`jev-1.14.0` or `jev-preview`), not an older model.
- Use a separate top-level output field for each target. Boxes can accompany
  `bool`, string `Literal` choices, and integer `Literal[0, 1, ...]` scores.
  `TypeSafeAdapter` does not translate the separate experimental rich
  `Noul`/`Choice`/`Score` value types or their predictor calibration overrides.
- Native Jev does not support `list[BoundingBox]`, nested or nullable boxes,
  arbitrary generated text, tools, or streaming. Unsupported schemas fail
  before the provider call. Use `dspy.Predict`, not a text-reasoning
  `ChainOfThought`, for this route.
- There is **no not-found box**. Ask a presence question alongside it and ignore
  the box when absent. A provider can still emit invalid geometry, which raises
  `dspy.AdapterParseError`; handle that as a failed prediction.
- Load PNG/JPEG bytes explicitly with `Image.from_path` or `Image.from_url`.
  A bare URL is not fetched by the native provider. DSPy does not resize, rotate,
  or apply EXIF orientation here. Follow the provider's current
  [image limits](https://docs.typesafe.ai/images).
- Multiple images retain their named input context. Describe both the target
  and the input image in each `OutputField(desc=...)`. A reference image can
  identify the target; cropped images require caller-owned coordinate mapping
  back to the original image.
- Cached responses retain coordinates and evidence. DSPy's existing sync and
  async cache namespaces are separate. Reload an edited image from its path
  to create an input with the new bytes.

See TypeSafe's [Box guide](https://docs.typesafe.ai/primitives/box) for reference
images, presence guards, confidence thresholds, tiling, and crop-edge questions.
Valid geometry and high confidence are not guarantees of localization accuracy.

## Generative providers and reusable geometry

Use the same type with `JSONAdapter`, `ChatAdapter`, `XMLAdapter`, or
`TwoStepAdapter` on a generative LM. Nested models, `list[BoundingBox]`, and optional boxes work with
those adapters, subject to the provider's structured-output support. They do
not become native TypeSafe detection lists. Generated coordinates pass the
same runtime validation; they have no measured TypeSafe confidence.

`BoundingBox` serializes as a JSON object, including inside demonstrations and
nested models. It is immutable and can also be an input field. No imaging
dependency is needed to construct or validate a box.

## Reproduce the live checks

From the repository, with `TYPESAFE_API_KEY` set:

```bash
python scripts/validate_bounding_boxes.py --output /tmp/bounding-box-proof
```

The script requires Pillow and optionally enables cmpnd tracing when
`CMPND_API_KEY` is set. It downloads a fixed COCO validation sample, records
annotation provenance and input hashes, alternates sync/async calls, preserves
raw answers, and produces ground-truth/prediction overlays and IoU measurements.
It also exercises multiple boxes, mixed judgments, image disambiguation,
reference-image matching, absence, edge questions, and caching. This small
sample is integration evidence, **not COCO mAP or a production-quality claim**.

<!-- START_API_REF -->
::: dspy.BoundingBox
    handler: python
    options:
        show_root_heading: true
        heading_level: 2
        show_source: true
<!-- END_API_REF -->
