import pydantic


def get_pydantic_object_serializer():
    # Pydantic V2 has a more robust JSON encoder, but we need to handle V1 as well.
    if hasattr(pydantic, "__version__") and pydantic.__version__.startswith("2."):
        from pydantic.v1.json import pydantic_encoder
        return pydantic_encoder
    else:
        from pydantic.json import pydantic_encoder
        return pydantic_encoder
