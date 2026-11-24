import json

from pydantic import ConfigDict, Field, field_validator, model_serializer, model_validator

from dspy.adapters.types.base_type import Type


class BoundingBox(Type):
    """An axis-aligned rectangle in normalized image coordinates.

    Edges are finite numbers in [0, 1], measured from the top-left corner.
    Multiply left/right by the image width and top/bottom by its height to
    obtain pixel coordinates. Use the dimensions of the upright image actually
    sent to the model, not those of an original image before resizing/cropping.

    Boxes have positive area: top < bottom and left < right. Invalid outputs
    are rejected, never reordered or clamped. Provider confidence is not part
    of the geometry; native TypeSafe evidence remains in the LM response.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    top: float = Field(description="Top edge as a fraction of image height, from 0 to 1.")
    bottom: float = Field(description="Bottom edge as a fraction of image height, from 0 to 1.")
    left: float = Field(description="Left edge as a fraction of image width, from 0 to 1.")
    right: float = Field(description="Right edge as a fraction of image width, from 0 to 1.")

    @field_validator("top", "bottom", "left", "right")
    @classmethod
    def validate_edge(cls, value):
        # Runtime validation keeps the schema portable to providers that reject
        # numeric minimum/maximum constraints in structured outputs.
        if not 0 <= value <= 1:
            raise ValueError("Bounding box edges must be finite numbers in [0, 1].")
        return value

    @model_validator(mode="after")
    def validate_order(self):
        if self.top >= self.bottom or self.left >= self.right:
            raise ValueError("Bounding boxes require top < bottom and left < right.")
        return self

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        # Geometry is structured JSON, not a media marker or a JSON-encoded string.
        return handler(self)

    def format(self) -> str:
        return json.dumps(self.model_dump())
