"""Per-session token protection for state-changing requests.

The server is intended for single-user localhost use, but a malicious web page
the user visits in another tab can still issue cross-origin POSTs to the local
server and trigger side effects (notably native builds via ``/api/build*``).

We mint a single per-process token at import time, serve it only through
``GET /api/session``, and require it as a request header on every unsafe
(state-changing) ``/api`` request. Because the server sends no CORS headers,
the browser's same-origin policy prevents a cross-origin script from reading
the response body of ``/api/session`` -- so an attacker page cannot learn the
token and cannot forge a valid state-changing request. Safe methods
(GET/HEAD/OPTIONS) are left open: they have no side effects and the same-origin
policy already blocks cross-origin reads of their responses.
"""

from __future__ import annotations

import secrets

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

#: Per-process token. Regenerated on every server start.
SESSION_TOKEN = secrets.token_urlsafe(32)

#: Header the SPA must send on state-changing requests.
SESSION_HEADER = "x-dsp-session"

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_PROTECTED_PREFIX = "/api/"

#: Reject request bodies larger than this (defense-in-depth against memory
#: exhaustion from oversized graph payloads). Field-level bounds in the API
#: models guard the specific numeric/list inputs; this caps the whole body.
MAX_BODY_BYTES = 16 * 1024 * 1024  # 16 MiB


class SessionTokenMiddleware(BaseHTTPMiddleware):
    """Reject state-changing ``/api`` requests lacking a valid session token."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method not in _SAFE_METHODS and request.url.path.startswith(_PROTECTED_PREFIX):
            provided = request.headers.get(SESSION_HEADER, "")
            if not secrets.compare_digest(provided, SESSION_TOKEN):
                return JSONResponse(
                    {"detail": "Missing or invalid session token"},
                    status_code=403,
                )
        return await call_next(request)


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose declared body exceeds :data:`MAX_BODY_BYTES`.

    This checks ``Content-Length`` (present for the SPA's JSON requests). It is
    a cheap guard, not a hard streaming limit; chunked uploads without a
    declared length bypass it, but the per-field bounds remain the real defense.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        super().__init__(app)
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > self._max_bytes:
                return JSONResponse(
                    {"detail": "Request body too large"},
                    status_code=413,
                )
        return await call_next(request)
