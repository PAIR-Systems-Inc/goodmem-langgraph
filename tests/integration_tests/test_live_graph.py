"""Live graph workflows; creates and deletes only this test module's own space."""

import base64
import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from goodmem import Goodmem
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from goodmem_langgraph import (
    GoodMemCreateSpace,
    GoodMemDeleteSpace,
    GoodMemGetMemory,
    GoodMemListSpaces,
    GoodMemRetrieveMemories,
    GoodMemRetriever,
    add_documents,
)

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def corpus() -> Iterator[tuple[Goodmem, str, list[str]]]:
    required = ["GOODMEM_BASE_URL", "GOODMEM_API_KEY", "GOODMEM_EMBEDDER_ID"]
    if any(not os.getenv(key) for key in required):
        pytest.skip("Set GOODMEM_BASE_URL, GOODMEM_API_KEY and GOODMEM_EMBEDDER_ID")
    with Goodmem(
        base_url=os.environ["GOODMEM_BASE_URL"],
        api_key=os.environ["GOODMEM_API_KEY"],
        verify=os.getenv("GOODMEM_VERIFY_SSL", "true").lower() != "false",
        timeout=120,
    ) as client:
        name = f"langgraph-live-{uuid.uuid4()}"
        space = GoodMemCreateSpace(client=client).invoke(
            {"name": name, "embedder_id": os.environ["GOODMEM_EMBEDDER_ID"]}
        )
        sid = space["space_id"]
        try:
            assert (
                GoodMemListSpaces(client=client).invoke({"name_filter": name})[0][
                    "space_id"
                ]
                == sid
            )
            ids = add_documents(
                client,
                sid,
                [
                    Document(
                        page_content="Refunds are available within 30 days.",
                        metadata={
                            "source": "https://example.org/refunds",
                            "department": "support",
                        },
                    ),
                    Document(
                        page_content="Employees must file refund expense claims within 10 days.",
                        metadata={
                            "source": "https://example.org/expenses",
                            "department": "finance",
                        },
                    ),
                ],
                indexing_timeout=120,
            )
            yield client, sid, ids
        finally:
            GoodMemDeleteSpace(client=client).invoke({"space_id": sid})


def search_graph(client: Goodmem, sid: str, **options: Any) -> Any:
    search = create_retriever_tool(
        GoodMemRetriever(client=client, space_ids=[sid], **options),
        "search_policies",
        "Search company refund policies.",
        document_prompt=PromptTemplate.from_template(
            "Source: {source}\n{page_content}"
        ),
        response_format="content_and_artifact",
    )
    builder = StateGraph(MessagesState)
    builder.add_node("search", ToolNode([search], handle_tool_errors=True))
    builder.add_edge(START, "search")
    builder.add_edge("search", END)
    return builder.compile()


def question() -> dict[str, Any]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_policies",
                        "args": {"query": "What is the refund policy?"},
                        "id": "call-1",
                    }
                ],
            )
        ]
    }


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_live_filtered_search(
    corpus: tuple[Goodmem, str, list[str]], asynchronous: bool
) -> None:
    client, sid, ids = corpus
    graph = search_graph(
        client, sid, filter="CAST(val('$.department') AS TEXT) = 'support'"
    )
    result = (
        await graph.ainvoke(question()) if asynchronous else graph.invoke(question())
    )
    message = result["messages"][-1]
    assert message.status == "success", message.content
    assert (
        "30 days" in message.content
        and "https://example.org/refunds" in message.content
    )
    assert "10 days" not in message.content
    assert message.artifact and {
        doc.metadata["memory_id"] for doc in message.artifact
    } == {ids[0]}


def test_live_reranking_without_llm(corpus: tuple[Goodmem, str, list[str]]) -> None:
    if not os.getenv("GOODMEM_RERANKER_ID"):
        pytest.skip("Set GOODMEM_RERANKER_ID")
    client, sid, _ = corpus
    message = search_graph(
        client, sid, reranker_id=os.environ["GOODMEM_RERANKER_ID"], k=2, fetch_k=8
    ).invoke(question())["messages"][-1]
    assert message.status == "success", message.content
    assert message.artifact and "https://example.org/" in message.content


def test_live_failed_reranking_is_visible(
    corpus: tuple[Goodmem, str, list[str]],
) -> None:
    client, sid, _ = corpus
    missing = str(uuid.uuid4())
    message = search_graph(client, sid, reranker_id=missing).invoke(question())[
        "messages"
    ][-1]
    # The server falls back to vector hits; they are returned and flagged.
    assert message.status == "success" and message.artifact, message.content
    for document in message.artifact:
        assert document.metadata["goodmem_partial"] is True
        assert "RERANKING_FAILED" in str(document.metadata["goodmem_statuses"])
    events = GoodMemRetrieveMemories(client=client).invoke(
        {"message": "refund policy", "space_ids": [sid], "reranker_id": missing}
    )
    assert any(
        event.get("status", {}).get("code") == "RERANKING_FAILED" for event in events
    )
    assert any(event.get("retrieved_item") for event in events)


def test_live_get_memory_content(corpus: tuple[Goodmem, str, list[str]]) -> None:
    client, _, ids = corpus
    result = GoodMemGetMemory(client=client).invoke(
        {"memory_id": ids[0], "include_content": True}
    )
    assert (
        base64.b64decode(result["original_content"]).decode()
        == "Refunds are available within 30 days."
    )


def test_live_empty_filter(corpus: tuple[Goodmem, str, list[str]]) -> None:
    client, sid, _ = corpus
    message = search_graph(
        client, sid, filter="CAST(val('$.department') AS TEXT) = 'nonexistent'"
    ).invoke(question())["messages"][-1]
    assert message.status == "success" and message.artifact == []
