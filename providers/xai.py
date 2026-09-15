"""
lm15.providers.xai — xAI Grok adapter (Chat Completions dialect).

xAI's API speaks the Chat Completions dialect at ``https://api.x.ai/v1``
(compat preset ``"xai"``, pinned live 2026-09-01).  What makes it a
first-class adapter instead of a preset route is authentication: xAI sells
subscription access (SuperGrok / X Premium) through a device-code OAuth
flow, and the resulting access token is sent as an ordinary bearer key.

Credential resolution order (``oauth-unless-explicit`` policy, spec/auth.md
AUTH-1) — through the router:

1. an explicit ``RouterConfig.api_keys`` entry (or an ``api_key`` argument
   when constructing this class directly): deliberate, in-process
   configuration always wins;
2. the stored subscription OAuth login when one is usable: lm15's own
   credential store, then the Pi agent store (``~/.pi/agent/auth.json``);
   refreshed tokens are written back to their source file (xAI rotates
   refresh tokens);
3. ``XAI_API_KEY`` from the environment, only when no usable subscription
   login is stored.

Why the subscription outranks the env var: both are stored state, but using
the subscription costs nothing per token while a key bills every call — and
lm15's durable constraint is that normal inference must not unexpectedly
spend money.  Stated residual trade-off: with a subscription stored, a set
``XAI_API_KEY`` is silently ignored; if you need that key's account, pass it
explicitly (``api_key=`` / ``RouterConfig.api_keys``).  Run
``lm15.doctor.explain_auth("xai")`` to see which rung won.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, ClassVar

from ..access import DEFAULT_XAI_BASE_URL, XAI
from ..errors import ProviderError, UnsupportedFeatureError
from ..features import ProviderManifest
from ..transports import TransportRequest
from .common import path_id
from ..types import ImageGenerationRequest, ImageGenerationResponse, ImagePart, Request, Usage, VideoGenerationRequest, VideoJobInfo, VideoPart
from .base import Credential, HttpResponse, SyncTransport, default_transport
from .openai_chat import OpenAIChatLM


class XaiLM(OpenAIChatLM):
    """Chat Completions adapter for xAI, with subscription OAuth fallback.

    xAI is a provider, not an access path: its image and video wire, and
    its refusals (reasoning off, logprobs, the MAP-8 cells), are provider
    facts and live here. Only the credential path is composed: the
    ``lm15.access.XAI`` policy carries the ``oauth-unless-explicit`` chain,
    the login hint, and the endpoint surfaces.
    """

    manifest: ClassVar[ProviderManifest] = XAI

    def __init__(
        self,
        api_key: Credential | None = None,
        *,
        credentials_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = DEFAULT_XAI_BASE_URL,
    ) -> None:
        super().__init__(
            api_key=api_key,
            transport=transport or default_transport(),
            base_url=base_url,
            compat="xai",
            access=XAI,
            credentials_path=credentials_path,
        )

    def _payload(self, request: Request, stream: bool) -> dict[str, Any]:
        # Grok reasoning models have no off switch.  The inherited deepseek
        # wire shape (thinking={"type": "disabled"}) is accepted by
        # api.x.ai but silently ignored — verified live 2026-09-01:
        # grok-4.6 still spent 158 reasoning tokens.  A silent paid no-op
        # on an explicit disable is worse than an error, so raise.  For
        # non-reasoning Grok variants, omit the reasoning config
        # (Config(reasoning=None)) — they never reason anyway.
        reasoning = request.config.reasoning
        if reasoning is not None and reasoning.is_off:
            raise UnsupportedFeatureError(
                "xai: reasoning cannot be disabled — Grok reasoning models have no "
                "off switch, and xAI silently ignores disable fields on the wire. "
                "Omit the reasoning config, or pick a non-reasoning Grok model.",
                provider=self.provider,
            )
        # docs.x.ai (models page, 2026-09-01): "logprobs and top_logprobs are
        # not supported by models grok-4.20 and newer. These fields will be
        # silently ignored if set."  Verified live 2026-09-01 on grok-4.6:
        # HTTP 200, the choice carries no logprobs key at all.  Every Grok
        # model served today is 4.20 or newer, so sending the field is a
        # guaranteed silent no-op — raise instead.
        if request.config.logprobs is not None:
            raise UnsupportedFeatureError(
                "xai: config.logprobs is not supported — grok-4.20 and newer "
                "silently ignore logprobs/top_logprobs on the wire (docs.x.ai, "
                "verified live 2026-09-01). OpenAI and Gemini carry logprobs.",
                provider=self.provider,
            )
        tc = request.config.tool_choice
        if tc is not None and tc.allowed and not (len(tc.allowed) == 1 and tc.mode == "required"):
            # MAP-8 rule 1 (live 2026-09-02): api.x.ai accepts allowed_tools and
            # ignores it — with {lookup} allowed and weather asked, it called
            # weather.  A silent widen; the forced single-function form held.
            raise UnsupportedFeatureError(
                "xai: tool_choice.allowed subsets are silently ignored by api.x.ai "
                "(verified live 2026-09-02); force a single tool with mode='required', "
                "or send only the allowed tools in Request.tools",
                provider=self.provider,
            )
        if tc is not None and tc.mode == "required" and request.config.response_format is not None:
            # MAP-8 rule 3 (live 2026-09-02): a forced tool next to a
            # response_format returned JSON text and no call.
            raise UnsupportedFeatureError(
                "xai: a forced tool (mode='required') cannot be combined with response_format — "
                "api.x.ai returns JSON text and drops the call (verified live 2026-09-02)",
                provider=self.provider,
            )
        return super()._payload(request, stream)

    def _image_generate_request(self, request: ImageGenerationRequest) -> TransportRequest:
        base = self.base_url.rstrip("/")
        payload: dict[str, Any] = {"model": request.model, "prompt": request.prompt, **(request.extensions or {})}
        if request.size is not None:
            # No wire slot: xAI sizes through quality/resolution knobs with
            # their own names (extensions).  Raising beats guessing a mapping.
            raise UnsupportedFeatureError(
                "xai: size has no wire slot; use extensions for xAI's quality/resolution fields",
                provider=self.provider,
            )
        if not request.images:
            return self._emit(method="POST", url=f"{base}/images/generations", headers=self._headers(), payload=payload, read_timeout=300.0)
        if len(request.images) > 1:
            raise UnsupportedFeatureError(
                "xai: image edits take exactly one input image; the wire has no slot for more",
                provider=self.provider,
            )
        payload["image"] = _xai_image_input(request.images[0], self.provider)
        return self._emit(method="POST", url=f"{base}/images/edits", headers=self._headers(), payload=payload, read_timeout=300.0)

    def _image_generation_from_response(self, request: ImageGenerationRequest, resp: HttpResponse) -> ImageGenerationResponse:
        data = resp.json()
        images: list[ImagePart] = []
        for item in data.get("data", []) or []:
            if not isinstance(item, dict):
                continue
            mime = item.get("mime_type")
            media_type = mime if isinstance(mime, str) and mime else "application/octet-stream"
            if item.get("b64_json"):
                images.append(ImagePart(media_type=media_type, data=str(item["b64_json"])))
            elif item.get("url"):
                images.append(ImagePart(media_type=media_type, url=str(item["url"])))
        if not images:
            raise ProviderError("xai: image response carries no images", provider=self.provider)
        # Captured: usage reports cost_in_usd_ticks only — no token counts
        # exist, so Usage stays empty and the figure lives in provider_data.
        return ImageGenerationResponse(images=tuple(images), usage=Usage(), provider_data=data)

    # ─── Video generation (grok-imagine; captured live 2026-09-01) ──────
    #
    # POST /videos/generations -> {"request_id"}; GET /videos/{id} ->
    # pending + progress %, then done + a PUBLIC MP4 URL (downloads with
    # no auth, verified) — so the result is URL-addressed, no fetch step.
    # There is NO list endpoint (probed: 404): the ticket you store is
    # the only copy.

    _VIDEO_STATUS_MAP: ClassVar[dict[str, str]] = {
        "pending": "running",
        "done": "completed",
        "failed": "failed",
    }

    def _video_submit_request(self, request: VideoGenerationRequest) -> TransportRequest:
        if request.seconds is not None:
            raise UnsupportedFeatureError(
                "xai: video duration has no wire slot", provider=self.provider,
            )
        if request.images:
            # The image-generation wire silently IGNORES unknown fields
            # (pixel-verified 2026-09-01); an unverified image-input mapping
            # here could silently produce prompt-only videos.  Raise until
            # the field is live-receipted.
            raise UnsupportedFeatureError(
                "xai: video input images are not mapped yet; "
                "use extensions until the mapping is live-receipted",
                provider=self.provider,
            )
        payload: dict[str, Any] = {"model": request.model, "prompt": request.prompt, **(request.extensions or {})}
        return self._emit(
            method="POST", url=f"{self.base_url.rstrip('/')}/videos/generations",
            headers=self._headers(), payload=payload, read_timeout=120.0,
        )

    def _video_job_from_body(self, body: str, video_id: "str | None" = None) -> VideoJobInfo:
        data = json.loads(body)
        request_id = data.get("request_id")
        if isinstance(request_id, str) and request_id:
            # The submit acknowledgement: a bare ticket, not yet started.
            return VideoJobInfo(id=request_id, status="queued", provider_data=data)
        if video_id is None:
            raise ProviderError("xai: video body carries no request_id", provider=self.provider)
        return self._video_status_info(video_id, data)

    def _video_status_request(self, video_id: str) -> TransportRequest:
        return self._emit(
            method="GET", url=f"{self.base_url.rstrip('/')}/videos/{path_id(video_id)}",
            headers=self._headers(), read_timeout=60.0,
        )

    def _video_status_info(self, video_id: str, data: "dict[str, Any]") -> VideoJobInfo:
        wire_status = str(data.get("status") or "")
        status = self._VIDEO_STATUS_MAP.get(wire_status)
        if status is None:
            raise ProviderError(f"xai: unknown video status {wire_status!r}", provider=self.provider)
        progress = data.get("progress")
        return VideoJobInfo(
            id=video_id,
            status=status,
            progress=int(progress) if isinstance(progress, (int, float)) and not isinstance(progress, bool) else None,
            model=data.get("model"),
            provider_data=data,
        )

    def _video_list_request(self, limit: int, model: "str | None") -> TransportRequest:
        raise UnsupportedFeatureError(
            "xai: the wire has no video list endpoint (probed 2026-09-01: 404) — "
            "the ticket you stored is the only copy",
            provider=self.provider,
        )

    def _video_result_fetch(self, status_body: "dict[str, Any]") -> None:
        return None  # the terminal body carries a public URL

    def _video_part(self, status_body: "dict[str, Any]", fetched: object) -> VideoPart:
        video = status_body.get("video") if isinstance(status_body.get("video"), dict) else {}
        url = video.get("url")
        if not isinstance(url, str) or not url:
            raise ProviderError("xai: terminal video carries no url", provider=self.provider)
        return VideoPart(media_type="video/mp4", url=url)

    def normalize_error(self, status: int, body: str) -> ProviderError:
        # xAI's own envelope is {"code": str, "error": str} (captured
        # 2026-09-01: model-not-found 400, unauthenticated 401) — refold it
        # into the OpenAI shape so the shared mapping preserves the wire
        # code as provider_code instead of dropping it.
        try:
            data = json.loads(body)
            if isinstance(data, dict) and isinstance(data.get("error"), str):
                body = json.dumps({"error": {"message": data["error"], "code": data.get("code")}})
        except ValueError:
            pass
        # Auth failures on the subscription path guide the user back to
        # login (the access policy's hint, applied by the shared mapping
        # when the stored login was the rung that won).
        return super().normalize_error(status, body)


# ─── Media generation (captured live 2026-09-01) ─────────────────────
#
# Images: /images/generations for text-to-image; /images/edits for
# image-to-image.  The split matters: `generations` silently IGNORES
# input images (verified by pixel check), so edits must never route
# there.  The edit input is `image:{url|file_id}` — an https URL, a
# data URI (verified honored), or an xAI file id.  Exactly one input
# image; the wire has no slot for more.  Responses state `mime_type`
# per image (JPEG, captured); usage reports only a dollar-tick cost,
# which stays verbatim in provider_data (no token counts exist).
# Speech: no endpoint (voice is app-only) — the base hooks raise.


def _xai_image_input(part: "ImagePart", provider: str) -> dict[str, str]:
    if part.url is not None:
        return {"url": part.url}
    if part.file_id is not None:
        return {"file_id": part.file_id}
    if part.data is not None:
        return {"url": f"data:{part.media_type};base64,{part.data}"}
    if part.path is not None:
        encoded = base64.b64encode(part.path.read_bytes()).decode("ascii")
        return {"url": f"data:{part.media_type};base64,{encoded}"}
    raise UnsupportedFeatureError(f"{provider}: input image carries no content", provider=provider)
