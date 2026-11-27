"""Unit tests for error_handler's suppression of raw exception detail (Medium finding).

Regression test: ``domain_exception_handler`` echoed ``exc.detail`` — raw
driver/SDK exception text — straight into the client-facing envelope for
every domain error, including ``DatabaseExecutionError``/``LLMProviderError``
whose detail is literally ``str(original_exception)``. That's the same text
already folded into ``message``, so echoing it again only doubled the
unsanitized-internals surface with no benefit to the client.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from nl_to_sql.api.middleware.error_handler import domain_exception_handler
from nl_to_sql.core.exceptions import (
    ConnectionValidationError,
    DatabaseExecutionError,
    LLMProviderError,
)


def _fake_request() -> MagicMock:
    request = MagicMock()
    request.state.request_id = "req-123"
    return request


class TestDetailSuppression:
    async def test_database_execution_error_detail_not_echoed(self) -> None:
        exc = DatabaseExecutionError(
            "Failed to execute SQL: connection refused",
            detail="asyncpg.exceptions.ConnectionDoesNotExistError: "
            "server closed the connection at 10.0.0.5:5432",
        )
        response = await domain_exception_handler(_fake_request(), exc)
        body = json.loads(response.body)
        assert "details" not in body
        # message stays intact — legitimate cases (a widget's own SQL error)
        # still surface useful info.
        assert "Failed to execute SQL" in body["message"]

    async def test_llm_provider_error_detail_not_echoed(self) -> None:
        exc = LLMProviderError(
            "Groq API call failed", detail="APIConnectionError: internal proxy 10.1.2.3 unreachable"
        )
        response = await domain_exception_handler(_fake_request(), exc)
        body = json.loads(response.body)
        assert "details" not in body

    async def test_other_domain_errors_still_include_detail(self) -> None:
        """Non-suppressed domain errors keep their existing detail behavior."""
        exc = ConnectionValidationError("Invalid connection URL", detail="scheme not allowed")
        response = await domain_exception_handler(_fake_request(), exc)
        body = json.loads(response.body)
        assert body["details"] == {"detail": "scheme not allowed"}
