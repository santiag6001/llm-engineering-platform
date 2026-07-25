"""Expected backend failure types."""

from __future__ import annotations


class BackendError(Exception):
    """Base class for normalized backend failures."""


class BackendUnavailableError(BackendError):
    """The backend could not be reached."""


class BackendTimeoutError(BackendError):
    """The backend did not respond within its configured timeout."""


class BackendProtocolError(BackendError):
    """The backend returned an invalid response."""


class BackendHTTPError(BackendError):
    """The backend returned a non-success HTTP response."""

    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        error_type: str = "backend_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code
