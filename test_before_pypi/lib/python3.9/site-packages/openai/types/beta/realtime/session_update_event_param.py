# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List, Union, Iterable
from typing_extensions import Literal, Required, TypedDict

__all__ = [
    "SessionUpdateEventParam",
    "Session",
    "SessionInputAudioTranscription",
    "SessionTool",
    "SessionTurnDetection",
]


class SessionInputAudioTranscription(TypedDict, total=False):
    language: str
    """The language of the input audio.

    Supplying the input language in
    [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) (e.g. `en`)
    format will improve accuracy and latency.
    """

    model: str
    """
    The model to use for transcription, `whisper-1` is the only currently supported
    model.
    """

    prompt: str
    """An optional text to guide the model's style or continue a previous audio
    segment.

    The [prompt](https://platform.openai.com/docs/guides/speech-to-text#prompting)
    should match the audio language.
    """


class SessionTool(TypedDict, total=False):
    description: str
    """
    The description of the function, including guidance on when and how to call it,
    and guidance about what to tell the user when calling (if anything).
    """

    name: str
    """The name of the function."""

    parameters: object
    """Parameters of the function in JSON Schema."""

    type: Literal["function"]
    """The type of the tool, i.e. `function`."""


class SessionTurnDetection(TypedDict, total=False):
    create_response: bool
    """Whether or not to automatically generate a response when a VAD stop event
    occurs.

    `true` by default.
    """

    interrupt_response: bool
    """
    Whether or not to automatically interrupt any ongoing response with output to
    the default conversation (i.e. `conversation` of `auto`) when a VAD start event
    occurs. `true` by default.
    """

    prefix_padding_ms: int
    """Amount of audio to include before the VAD detected speech (in milliseconds).

    Defaults to 300ms.
    """

    silence_duration_ms: int
    """Duration of silence to detect speech stop (in milliseconds).

    Defaults to 500ms. With shorter values the model will respond more quickly, but
    may jump in on short pauses from the user.
    """

    threshold: float
    """Activation threshold for VAD (0.0 to 1.0), this defaults to 0.5.

    A higher threshold will require louder audio to activate the model, and thus
    might perform better in noisy environments.
    """

    type: str
    """Type of turn detection, only `server_vad` is currently supported."""


class Session(TypedDict, total=False):
    input_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"]
    """The format of input audio.

    Options are `pcm16`, `g711_ulaw`, or `g711_alaw`. For `pcm16`, input audio must
    be 16-bit PCM at a 24kHz sample rate, single channel (mono), and little-endian
    byte order.
    """

    input_audio_transcription: SessionInputAudioTranscription
    """
    Configuration for input audio transcription, defaults to off and can be set to
    `null` to turn off once on. Input audio transcription is not native to the
    model, since the model consumes audio directly. Transcription runs
    asynchronously through
    [OpenAI Whisper transcription](https://platform.openai.com/docs/api-reference/audio/createTranscription)
    and should be treated as rough guidance rather than the representation
    understood by the model. The client can optionally set the language and prompt
    for transcription, these fields will be passed to the Whisper API.
    """

    instructions: str
    """The default system instructions (i.e.

    system message) prepended to model calls. This field allows the client to guide
    the model on desired responses. The model can be instructed on response content
    and format, (e.g. "be extremely succinct", "act friendly", "here are examples of
    good responses") and on audio behavior (e.g. "talk quickly", "inject emotion
    into your voice", "laugh frequently"). The instructions are not guaranteed to be
    followed by the model, but they provide guidance to the model on the desired
    behavior.

    Note that the server sets default instructions which will be used if this field
    is not set and are visible in the `session.created` event at the start of the
    session.
    """

    max_response_output_tokens: Union[int, Literal["inf"]]
    """
    Maximum number of output tokens for a single assistant response, inclusive of
    tool calls. Provide an integer between 1 and 4096 to limit output tokens, or
    `inf` for the maximum available tokens for a given model. Defaults to `inf`.
    """

    modalities: List[Literal["text", "audio"]]
    """The set of modalities the model can respond with.

    To disable audio, set this to ["text"].
    """

    model: Literal[
        "gpt-4o-realtime-preview",
        "gpt-4o-realtime-preview-2024-10-01",
        "gpt-4o-realtime-preview-2024-12-17",
        "gpt-4o-mini-realtime-preview",
        "gpt-4o-mini-realtime-preview-2024-12-17",
    ]
    """The Realtime model used for this session."""

    output_audio_format: Literal["pcm16", "g711_ulaw", "g711_alaw"]
    """The format of output audio.

    Options are `pcm16`, `g711_ulaw`, or `g711_alaw`. For `pcm16`, output audio is
    sampled at a rate of 24kHz.
    """

    temperature: float
    """Sampling temperature for the model, limited to [0.6, 1.2]. Defaults to 0.8."""

    tool_choice: str
    """How the model chooses tools.

    Options are `auto`, `none`, `required`, or specify a function.
    """

    tools: Iterable[SessionTool]
    """Tools (functions) available to the model."""

    turn_detection: SessionTurnDetection
    """Configuration for turn detection.

    Can be set to `null` to turn off. Server VAD means that the model will detect
    the start and end of speech based on audio volume and respond at the end of user
    speech.
    """

    voice: Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
    """The voice the model uses to respond.

    Voice cannot be changed during the session once the model has responded with
    audio at least once. Current voice options are `alloy`, `ash`, `ballad`,
    `coral`, `echo` `sage`, `shimmer` and `verse`.
    """


class SessionUpdateEventParam(TypedDict, total=False):
    session: Required[Session]
    """Realtime session object configuration."""

    type: Required[Literal["session.update"]]
    """The event type, must be `session.update`."""

    event_id: str
    """Optional client-generated ID used to identify this event."""
