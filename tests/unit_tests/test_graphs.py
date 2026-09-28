"""Run actual LangGraph graphs against the installed SDK and mock HTTP."""

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool, create_retriever_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from langgraph_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteMemory,
    GoodMemListSpaces,
    GoodMemRetrieveMemories,
    GoodMemRetriever,
    GoodMemUpdateSpace,
)
from tests.unit_tests.conftest import (
    CHUNK,
    EMBEDDER_ID,
    MEMORY,
    MEMORY_ID,
    RERANKER_ID,
    SPACE,
    SPACE_ID,
    SPACE_ID_2,
    Wire,
    ndjson,
)


def graph_for(*tools: BaseTool) -> Any:
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode(list(tools), handle_tool_errors=True))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile()


def call(name: str, args: dict[str, Any], call_id: str = "call-1") -> dict[str, Any]:
    return {
        "messages": [
            AIMessage(
                content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
            )
        ]
    }


def search_tool(wire: Wire, **settings: Any) -> BaseTool:
    return create_retriever_tool(
        GoodMemRetriever(client=wire.sdk, space_ids=[SPACE_ID], **settings),
        name="search_policies",
        description="Search support policies for refund and return questions.",
        document_prompt=PromptTemplate.from_template(
            "Source: {source}\n{page_content}"
        ),
        response_format="content_and_artifact",
    )


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_scoped_search_returns_citations_and_document_artifacts(
    wire: Wire, asynchronous: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOODMEM_BASE_URL", "https://wrong-server.test")
    monkeypatch.setenv("GOODMEM_API_KEY", "wrong-key")
    wire.responses.append(ndjson(CHUNK, {"memoryDefinition": MEMORY}))
    search = search_tool(
        wire,
        filter="CAST(val('$.department') AS TEXT) = 'support'",
        reranker_id=RERANKER_ID,
        k=2,
        fetch_k=8,
    )
    assert set(search.get_input_schema().model_fields) == {"query"}
    graph = graph_for(search)
    state = call(search.name, {"query": "What is the refund policy?"})
    result = await graph.ainvoke(state) if asynchronous else graph.invoke(state)
    message = result["messages"][-1]
    assert isinstance(message, ToolMessage) and message.status == "success"
    assert (
        "30 days" in message.content and MEMORY["metadata"]["source"] in message.content
    )
    assert message.tool_call_id == "call-1"
    document = message.artifact[0]
    assert document.id == "chunk-1" and document.metadata["memory_id"] == MEMORY_ID
    assert document.metadata["department"] == "support"
    request = wire.requests[0]
    assert (
        request.url.host == "goodmem.test"
        and request.headers["X-API-Key"] == "test-key"
    )
    body = json.loads(request.content)
    assert body["spaceKeys"] == [
        {
            "spaceId": SPACE_ID,
            "filter": "CAST(val('$.department') AS TEXT) = 'support'",
        }
    ]
    assert body["requestedSize"] == 8
    config = body["postProcessor"]["config"]
    assert config["reranker_id"] == RERANKER_ID and config["max_results"] == 2
    assert "llm_id" not in config
    assert not wire.http.is_closed


def test_two_search_tools_keep_their_configured_scopes(wire: Wire) -> None:
    policies = search_tool(wire)
    tickets = create_retriever_tool(
        GoodMemRetriever(client=wire.sdk, space_ids=[SPACE_ID_2]),
        "search_tickets",
        "Search customer support tickets.",
    )
    wire.responses.extend([ndjson(), ndjson()])
    graph = graph_for(policies, tickets)
    for tool, expected in [(policies, SPACE_ID), (tickets, SPACE_ID_2)]:
        # Additional model arguments cannot change the retriever's configured scope.
        graph.invoke(
            call(tool.name, {"query": "refund", "space_ids": ["attacker-space"]})
        )
        assert json.loads(wire.requests[-1].content)["spaceKeys"] == [
            {"spaceId": expected}
        ]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_empty_search_finishes_after_one_request(
    wire: Wire, asynchronous: bool
) -> None:
    wire.responses.append(ndjson())
    graph = graph_for(search_tool(wire))
    state = call("search_policies", {"query": "No matching documents"})
    result = await graph.ainvoke(state) if asynchronous else graph.invoke(state)
    assert result["messages"][-1].artifact == []
    assert len(wire.requests) == 1


@pytest.mark.parametrize(
    "code", ["RERANKING_FAILED", "VECTOR_SEARCH_PARTIAL", "SUMMARIZATION_FAILED"]
)
def test_failed_retrieval_keeps_the_hits_and_flags_them(wire: Wire, code: str) -> None:
    wire.responses.append(
        ndjson(
            CHUNK,
            {"memoryDefinition": MEMORY},
            {"status": {"code": code, "message": "A retrieval stage failed"}},
        )
    )
    result = graph_for(search_tool(wire)).invoke(
        call("search_policies", {"query": "refund"})
    )
    message = result["messages"][-1]
    assert message.status == "success"
    assert "Refunds are available within 30 days." in message.content
    [document] = message.artifact
    assert document.metadata["goodmem_partial"] is True
    assert code in str(document.metadata["goodmem_statuses"])


def test_failed_retrieval_without_hits_warns(wire: Wire) -> None:
    wire.responses.append(
        ndjson({"status": {"code": "RERANKING_FAILED", "message": "Reranker down"}})
    )
    with pytest.warns(UserWarning, match="RERANKING_FAILED"):
        result = graph_for(search_tool(wire)).invoke(
            call("search_policies", {"query": "refund"})
        )
    message = result["messages"][-1]
    assert message.status == "success" and message.artifact == []


def test_raw_retrieval_preserves_chunks_and_failure_diagnostics(wire: Wire) -> None:
    wire.responses.append(
        ndjson(
            CHUNK,
            {"status": {"code": "RERANKING_FAILED", "message": "Reranker unavailable"}},
        )
    )
    tool = GoodMemRetrieveMemories(client=wire.sdk)
    result = graph_for(tool).invoke(
        call(
            tool.name,
            {
                "message": "refund",
                "space_ids": [SPACE_ID],
                "reranker_id": RERANKER_ID,
            },
        )
    )
    events = json.loads(result["messages"][-1].content)
    assert events[0]["retrieved_item"]["chunk"]["chunk"]["chunk_text"]
    assert events[1]["status"]["code"] == "RERANKING_FAILED"


@pytest.mark.parametrize("raw_tool", [False, True])
def test_unknown_server_status_does_not_abort_graph(wire: Wire, raw_tool: bool) -> None:
    wire.responses.append(
        ndjson(
            CHUNK,
            {"memoryDefinition": MEMORY},
            {"status": {"code": "FUTURE_SERVER_NOTICE", "message": "New notice"}},
        )
    )
    tool = GoodMemRetrieveMemories(client=wire.sdk) if raw_tool else search_tool(wire)
    args = (
        {"message": "refund", "space_ids": [SPACE_ID]}
        if raw_tool
        else {"query": "refund"}
    )
    message = graph_for(tool).invoke(call(tool.name, args))["messages"][-1]
    assert message.status == "success" and "30 days" in message.content
    if raw_tool:
        assert json.loads(message.content)[-1]["status"]["message"] == "New notice"


@pytest.mark.parametrize("raw_tool", [False, True])
def test_malformed_response_is_not_an_empty_success(wire: Wire, raw_tool: bool) -> None:
    wire.responses.append(
        httpx.Response(
            200, text="not JSON", headers={"content-type": "application/x-ndjson"}
        )
    )
    tool = GoodMemRetrieveMemories(client=wire.sdk) if raw_tool else search_tool(wire)
    args = (
        {"message": "refund", "space_ids": [SPACE_ID]}
        if raw_tool
        else {"query": "refund"}
    )
    result = graph_for(tool).invoke(call(tool.name, args))
    assert result["messages"][-1].status == "error"
    assert len(wire.requests) == 1


def test_space_listing_follows_pagination_in_a_tool_node(wire: Wire) -> None:
    wire.responses.extend(
        [
            httpx.Response(200, json={"spaces": [SPACE], "nextToken": "page-2"}),
            httpx.Response(200, json={"spaces": [SPACE | {"spaceId": SPACE_ID_2}]}),
        ]
    )
    tool = GoodMemListSpaces(client=wire.sdk)
    message = graph_for(tool).invoke(call(tool.name, {"max_items": None}))["messages"][
        -1
    ]
    assert [space["space_id"] for space in json.loads(message.content)] == [
        SPACE_ID,
        SPACE_ID_2,
    ]
    assert wire.requests[1].url.params["next_token"] == "page-2"


def test_duplicate_space_conflict_is_reported_to_agent(wire: Wire) -> None:
    wire.responses.append(httpx.Response(409, json={"message": "Space already exists"}))
    tool = GoodMemCreateSpace(client=wire.sdk)
    result = graph_for(tool).invoke(
        call(tool.name, {"name": "Policies", "embedder_id": EMBEDDER_ID})
    )
    assert result["messages"][-1].status == "error"
    assert "already exists" in result["messages"][-1].content
    assert wire.requests[0].method == "POST" and len(wire.requests) == 1


def test_indexing_timeout_keeps_the_accepted_id_in_the_tool_message(wire: Wire) -> None:
    pending = MEMORY | {"processingStatus": "PENDING"}
    wire.responses.extend([httpx.Response(200, json=pending)] * 2)
    tool = GoodMemCreateMemory(client=wire.sdk)
    result = graph_for(tool).invoke(
        call(
            tool.name,
            {"space_id": SPACE_ID, "original_content": "note", "indexing_timeout": 0},
        )
    )
    message = result["messages"][-1]
    assert message.status == "error" and f"{MEMORY_ID} was created" in message.content
    assert [request.method for request in wire.requests] == ["POST", "GET"]


def test_update_schema_no_longer_advertises_public_read() -> None:
    assert "public_read" not in GoodMemUpdateSpace().get_input_schema().model_fields


@pytest.mark.parametrize(
    "memory_id",
    [f"../spaces/{SPACE_ID}", f"..%2Fspaces%2F{SPACE_ID}", f"{MEMORY_ID}#x"],
)
def test_model_supplied_ids_cannot_reach_another_endpoint(
    wire: Wire, memory_id: str
) -> None:
    # langchain-goodmem 0.2.1 and 0.2.2 sent DELETE /v1/spaces/<id> for the first
    # id; the 0.2.3 floor refuses non-UUID ids before any request.
    result = graph_for(GoodMemDeleteMemory(client=wire.sdk)).invoke(
        call("goodmem_delete_memory", {"memory_id": memory_id})
    )
    message = result["messages"][-1]
    assert message.status == "error" and "UUID" in message.content
    assert wire.requests == []
