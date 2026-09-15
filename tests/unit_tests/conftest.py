"""Exercise the installed SDK over HTTP without a live server."""

import json
from collections import deque
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from goodmem import Goodmem

AUDIT = {"createdAt": 1, "updatedAt": 1, "createdById": "user", "updatedById": "user"}
MEMORY = {
    **AUDIT,
    "memoryId": "memory-1",
    "spaceId": "space-1",
    "contentType": "text/plain",
    "processingStatus": "COMPLETED",
    "pageImageStatus": "COMPLETED",
    "pageImageCount": 0,
    "metadata": {"source": "https://example.org/policies", "department": "support"},
}
CHUNK = {
    "retrievedItem": {
        "chunk": {
            "resultSetId": "set-1",
            "memoryIndex": 0,
            "relevanceScore": -0.82,
            "chunk": {
                **AUDIT,
                "chunkId": "chunk-1",
                "memoryId": "memory-1",
                "chunkSequenceNumber": 0,
                "chunkText": "Refunds are available within 30 days.",
                "vectorStatus": "COMPLETED",
            },
        }
    }
}
SPACE = {
    **AUDIT,
    "spaceId": "space-1",
    "name": "Policies",
    "ownerId": "user",
    "labels": {},
    "spaceEmbedders": [],
}


def ndjson(*events: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        text="\n".join(json.dumps(event) for event in events),
        headers={"content-type": "application/x-ndjson"},
    )


class Wire:
    def __init__(self) -> None:
        self.responses: deque[httpx.Response | Exception] = deque()
        self.requests: list[httpx.Request] = []
        self.http = httpx.Client(
            base_url="https://goodmem.test",
            headers={"X-API-Key": "test-key"},
            transport=httpx.MockTransport(self.handle),
        )
        self.sdk = Goodmem(http_client=self.http)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert self.responses, f"Unexpected request: {request.method} {request.url}"
        response = self.responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def wire() -> Iterator[Wire]:
    transport = Wire()
    try:
        yield transport
        assert not transport.responses, "Expected HTTP requests were not made"
    finally:
        transport.http.close()
