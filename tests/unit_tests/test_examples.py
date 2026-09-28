"""Execute the documented graph and agent through their real framework APIs."""

from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from examples.react_agent_with_memory import build_agent
from examples.retrieve_with_graph import build_graph
from langgraph_goodmem import GoodMemIngestionError, GoodMemRetriever, add_documents
from tests.unit_tests.conftest import CHUNK, MEMORY, MEMORY_ID, SPACE_ID, Wire, ndjson


class RetrievalCallbacks(BaseCallbackHandler):
    def __init__(self) -> None:
        self.documents: list[Document] = []

    def on_retriever_end(self, documents: list[Document], **kwargs: Any) -> None:
        self.documents.extend(documents)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_document_node_preserves_callbacks_and_metadata(
    wire: Wire, asynchronous: bool
) -> None:
    wire.responses.append(ndjson(CHUNK, {"memoryDefinition": MEMORY}))
    graph = build_graph(GoodMemRetriever(client=wire.sdk, space_ids=[SPACE_ID]))
    callbacks = RetrievalCallbacks()
    config = {"callbacks": [callbacks]}
    state = {"question": "refund", "documents": []}
    result = (
        await graph.ainvoke(state, config=config)
        if asynchronous
        else graph.invoke(state, config=config)
    )
    assert result["documents"] == callbacks.documents
    assert callbacks.documents[0].metadata["source"] == MEMORY["metadata"]["source"]


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolCallingModel":
        return self


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_agent_example_completes_a_tool_call(
    wire: Wire, asynchronous: bool
) -> None:
    wire.responses.append(ndjson(CHUNK, {"memoryDefinition": MEMORY}))
    model = ToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_policies",
                        "args": {"query": "refund policy"},
                        "id": "search-1",
                    }
                ],
            ),
            AIMessage(content="Answer after retrieval."),
        ]
    )
    agent = build_agent(model, GoodMemRetriever(client=wire.sdk, space_ids=[SPACE_ID]))
    state = {"messages": [{"role": "user", "content": "What is the refund policy?"}]}
    result = await agent.ainvoke(state) if asynchronous else agent.invoke(state)
    messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(messages) == 1 and "30 days" in messages[0].content
    assert MEMORY["metadata"]["source"] in messages[0].content
    assert messages[0].artifact[0].metadata["memory_id"] == MEMORY_ID
    assert result["messages"][-1].content == "Answer after retrieval."


@pytest.mark.parametrize("section", ["Search from a graph", "Add documents"])
def test_readme_examples_run_independently(
    wire: Wire, monkeypatch: pytest.MonkeyPatch, section: str
) -> None:
    monkeypatch.setenv("GOODMEM_BASE_URL", "https://goodmem.test")
    monkeypatch.setenv("GOODMEM_API_KEY", "test-key")
    monkeypatch.setenv("GOODMEM_SPACE_ID", SPACE_ID)
    if section == "Search from a graph":
        wire.responses.append(ndjson(CHUNK, {"memoryDefinition": MEMORY}))
    else:
        wire.responses.extend(
            [
                httpx.Response(
                    200, json={"results": [{"success": True, "memory": MEMORY}]}
                ),
                httpx.Response(200, json=MEMORY),
            ]
        )
    # Keep the documented constructors and environment configuration real;
    # intercept only HTTP, so missing imports and invalid SDK arguments fail.
    monkeypatch.setattr(
        httpx.HTTPTransport,
        "handle_request",
        lambda self, request: wire.handle(request),
    )
    readme = Path("README.md").read_text()
    section_text = readme.split(f"## {section}\n", 1)[1].split("\n## ", 1)[0]
    code = section_text.split("```python\n", 1)[1].split("```", 1)[0]
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec(compile(code, "README.md", "exec"), namespace)
    if section == "Search from a graph":
        assert namespace["result"]["documents"][0].id == "chunk-1"
    else:
        assert namespace["memory_ids"] == [MEMORY_ID]
        assert [request.method for request in wire.requests] == ["POST", "GET"]
    assert all(
        request.url.host == "goodmem.test"
        and request.headers["X-API-Key"] == "test-key"
        for request in wire.requests
    )


def test_documents_are_written_before_the_graph_searches(wire: Wire) -> None:
    wire.responses.extend(
        [
            httpx.Response(
                200, json={"results": [{"success": True, "memory": MEMORY}]}
            ),
            httpx.Response(200, json=MEMORY),
            ndjson(CHUNK, {"memoryDefinition": MEMORY}),
        ]
    )
    ids = add_documents(
        wire.sdk,
        SPACE_ID,
        [
            Document(
                page_content="Refunds are available within 30 days.",
                metadata=MEMORY["metadata"],
            )
        ],
    )
    result = build_graph(
        GoodMemRetriever(client=wire.sdk, space_ids=[SPACE_ID])
    ).invoke({"question": "refund", "documents": []})
    assert ids == [result["documents"][0].metadata["memory_id"]]
    assert [request.url.path for request in wire.requests] == [
        "/v1/memories:batchCreate",
        f"/v1/memories/{MEMORY_ID}",
        "/v1/memories:retrieve",
    ]


def test_ingestion_wait_failure_keeps_created_ids(wire: Wire) -> None:
    pending = MEMORY | {"processingStatus": "PENDING"}
    wire.responses.extend(
        [
            httpx.Response(
                200, json={"results": [{"success": True, "memory": pending}]}
            ),
            httpx.Response(200, json=pending),
        ]
    )
    with pytest.raises(GoodMemIngestionError) as caught:
        add_documents(
            wire.sdk, SPACE_ID, [Document(page_content="note")], indexing_timeout=0
        )
    assert caught.value.created_memory_ids == [MEMORY_ID]
