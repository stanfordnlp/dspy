from typing import Any


def get_google_ai_safety_settings() -> list[dict[str, Any]]:
    """Returns the default kwargs for Google AI."""
    return [
        {"category": cat, "threshold": "BLOCK_NONE"}
        for cat in (
            "HARM_CATEGORY_UNSPECIFIED",
            "HARM_CATEGORY_DEROGATORY",
            "HARM_CATEGORY_TOXICITY",
            "HARM_CATEGORY_VIOLENCE",
            "HARM_CATEGORY_SEXUAL",
            "HARM_CATEGORY_MEDICAL",
            "HARM_CATEGORY_DANGEROUS",
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        )
    ]
