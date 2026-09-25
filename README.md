# langgraph-goodmem

Give LangGraph agents searchable, persistent memory with [GoodMem](https://goodmem.ai).
GoodMem handles document storage, chunking, embedding, search, and optional reranking.
Use it from a graph node or give an agent a search tool with access to the spaces you choose.

This package shares its tools, retriever, and ingestion functions with
[langchain-goodmem](https://github.com/PAIR-Systems-Inc/goodmem-langchain), so fixes reach both integrations.

## Install

```bash
pip install langgraph-goodmem
```

Requires Python 3.10+. Configure an existing GoodMem server and space:

```bash
export GOODMEM_BASE_URL="https://your-goodmem-server.example.com"
export GOODMEM_API_KEY="your-api-key"
export GOODMEM_SPACE_ID="your-space-uuid"
```

## Search from a graph

This complete example searches your space without an LLM. The retrieval node adds
`Document` objects, including source metadata, to the graph's state.

```python
import os
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph_goodmem import GoodMemRetriever

class State(TypedDict):
    question: str
    documents: list[Document]

retriever = GoodMemRetriever(space_ids=[os.environ["GOODMEM_SPACE_ID"]], k=5)

def search(state: State, config: RunnableConfig):
    return {"documents": retriever.invoke(state["question"], config=config)}

builder = StateGraph(State)
builder.add_node("search", search)
builder.add_edge(START, "search")
builder.add_edge("search", END)
graph = builder.compile()
result = graph.invoke({"question": "What is the refund policy?", "documents": []})
for document in result["documents"]:
    print(document.metadata["source"], document.page_content, sep="\n")
```

## Use an agent

The [agent example](examples/react_agent_with_memory.py) gives `create_agent` a
scoped search tool, also usable in `ToolNode`. Give searches distinct names and
descriptions; your code controls their spaces, filters, and rerankers.

Install `langgraph-goodmem[agents]` plus your chosen model provider's LangChain
package, configure its credentials, and set `GOODMEM_CHAT_MODEL=provider:model`.

## Add documents

```python
import os

from goodmem import Goodmem
from langchain_core.documents import Document
from langgraph_goodmem import add_documents

with Goodmem(base_url=os.environ["GOODMEM_BASE_URL"],
             api_key=os.environ["GOODMEM_API_KEY"]) as client:
    memory_ids = add_documents(client, os.environ["GOODMEM_SPACE_ID"], [
        Document(page_content="Refunds are available within 30 days.",
                 metadata={"source": "https://example.com/refunds"})
    ])
```

Ingestion waits for the memories it created. Empty searches return immediately.
`GoodMemIngestionError.created_memory_ids` identifies accepted writes if indexing
fails, so you can check them with `wait_for_memory` instead of uploading again.

## More options

- Set `filter="CAST(val('$.department') AS TEXT) = 'support'"` on the retriever for metadata filtering.
- Set `reranker_id` and optionally `fetch_k` to rerank without an LLM.
- Use `retriever.invoke`, `ainvoke`, `batch`, or `abatch` for `Document` results.
  Async calls currently run the shared synchronous SDK in a thread executor.
- Pass `client=Goodmem(...)` to share a connection. For local self-signed TLS,
  use `Goodmem(..., verify=False)`; keep verification enabled in production.
- Administrative tools remain available for trusted workflows. `GoodMemRetrieveMemories`
  returns raw SDK events, including failure statuses. How `GoodMemRetriever` reports
  a failure status depends on the installed `langchain-goodmem`: 0.2.1 raises
  `GoodMemRetrievalError`; 0.2.2 returns the Documents with `goodmem_partial` and
  `goodmem_statuses` metadata, or an empty list with a warning.

This package provides retrieval and ingestion; it does not implement LangGraph's
`BaseStore` or a checkpointer for graph execution state.
See the [0.2 migration notes](CHANGELOG.md) for API changes.
