# langgraph-goodmem

[![PyPI](https://img.shields.io/pypi/v/langgraph-goodmem.svg)](https://pypi.org/project/langgraph-goodmem/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

LangGraph integration for [GoodMem](https://goodmem.ai) — long-term agent memory with semantic storage and retrieval.

GoodMem is a memory layer for AI agents that handles embedding, vector search, reranking, and LLM-powered answering server-side. This package exposes GoodMem operations as LangGraph tools that can be used with any LangGraph agent or graph.

## Installation

```bash
pip install langgraph-goodmem
```

Requires Python 3.10+.

## Tools

| Tool | Description |
|---|---|
| `GoodMemListEmbedders` | List available embedder models |
| `GoodMemListSpaces` | List all spaces in your account |
| `GoodMemGetSpace` | Fetch a specific space by ID |
| `GoodMemCreateSpace` | Create a new space or reuse an existing one |
| `GoodMemUpdateSpace` | Update a space's name or metadata |
| `GoodMemDeleteSpace` | Delete a space and all of its memories |
| `GoodMemCreateMemory` | Store text or files as memories |
| `GoodMemListMemories` | List memories in a space |
| `GoodMemGetMemory` | Fetch a specific memory by ID |
| `GoodMemRetrieveMemories` | Semantic similarity search across spaces |
| `GoodMemDeleteMemory` | Permanently delete a memory |

## Quick start

```python
from langgraph_goodmem import (
    GoodMemCreateSpace,
    GoodMemCreateMemory,
    GoodMemRetrieveMemories,
)
from langgraph.prebuilt import create_react_agent

goodmem_kwargs = {
    "goodmem_base_url": "http://localhost:8080",
    "goodmem_api_key": "your-api-key",
}

tools = [
    GoodMemCreateSpace(**goodmem_kwargs),
    GoodMemCreateMemory(**goodmem_kwargs),
    GoodMemRetrieveMemories(**goodmem_kwargs),
]

agent = create_react_agent(model="gpt-4o", tools=tools)
```

## Usage in a custom LangGraph graph

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph_goodmem import (
    GoodMemCreateSpace,
    GoodMemCreateMemory,
    GoodMemRetrieveMemories,
)

tools = [
    GoodMemCreateSpace(**goodmem_kwargs),
    GoodMemCreateMemory(**goodmem_kwargs),
    GoodMemRetrieveMemories(**goodmem_kwargs),
]

tool_node = ToolNode(tools)
```

## Environment variables

| Variable | Description |
|---|---|
| `GOODMEM_BASE_URL` | Base URL of the GoodMem API server |
| `GOODMEM_API_KEY` | API key for authentication |
| `GOODMEM_VERIFY_SSL` | Set to `false` to skip TLS verification for self-signed certs (default: `true`) |

## Example

A full ReAct agent example is in [examples/react_agent_with_memory.py](examples/react_agent_with_memory.py).

## License

MIT — see [LICENSE](LICENSE).
