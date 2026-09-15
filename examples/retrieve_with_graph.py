"""Retrieve Documents in a StateGraph node, with sync and async execution.

Set GOODMEM_BASE_URL, GOODMEM_API_KEY and GOODMEM_SPACE_ID, then run this file.
No LLM is required. Configure a reranker with GOODMEM_RERANKER_ID if desired.
"""

import os
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, START, StateGraph

from langgraph_goodmem import GoodMemRetriever


class SearchState(TypedDict):
    question: str
    documents: list[Document]


def build_graph(retriever: GoodMemRetriever) -> Any:
    """Create a graph whose search node preserves callbacks and async execution."""

    def retrieve(
        state: SearchState, config: RunnableConfig
    ) -> dict[str, list[Document]]:
        return {"documents": retriever.invoke(state["question"], config=config)}

    async def aretrieve(
        state: SearchState, config: RunnableConfig
    ) -> dict[str, list[Document]]:
        return {"documents": await retriever.ainvoke(state["question"], config=config)}

    builder = StateGraph(SearchState)
    builder.add_node("retrieve", RunnableLambda(retrieve, afunc=aretrieve))
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", END)
    return builder.compile()


def main() -> None:
    retriever = GoodMemRetriever(
        space_ids=[os.environ["GOODMEM_SPACE_ID"]],
        reranker_id=os.getenv("GOODMEM_RERANKER_ID"),
        k=5,
    )
    graph = build_graph(retriever)
    result = graph.invoke({"question": "What is the refund policy?", "documents": []})
    for document in result["documents"]:
        print(document.metadata["source"], document.page_content, sep="\n")


if __name__ == "__main__":
    main()
