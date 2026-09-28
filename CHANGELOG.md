# Changelog

## 0.2.1 — 2026-09-28

Requires `langchain-goodmem` 0.2.3 or later. Earlier versions in the old range
(0.2.1 and 0.2.2) send a model-supplied ID such as `../spaces/<id>` into the URL
path, so `goodmem_delete_memory` could delete a whole space, and 0.2.2 and earlier
let a model upload any local file. 0.2.3 refuses non-UUID IDs before any request
and confines file uploads to an operator-set directory.

`GoodMemRetriever` now returns the Documents it received when the server reports
a problem, flagged with `goodmem_partial` and `goodmem_statuses` metadata, instead
of raising `GoodMemRetrievalError`; with no Documents it returns an empty list and
emits a warning.

## 0.2.0 — 2026-09-15

LangGraph now uses the published `langchain-goodmem` implementations directly.
The separate HTTP client, response parser, and tool implementations have been removed.
The dependency requires at least `langchain-goodmem` 0.2.1, including its fix for
retrieval statuses introduced by newer servers.

### Available to graphs and agents

- `GoodMemRetriever` returns LangChain Documents with source and memory/chunk IDs,
  configured metadata filters, and optional reranking without an LLM.
- `add_documents` stores Documents and metadata; `wait_for_memory` checks indexing
  of a specific memory. The ingestion exception retains accepted memory IDs.
- The same tools work in `ToolNode` and LangChain agents. The examples favor scoped
  search tools whose spaces and retrieval configuration are controlled by the developer.
- The agent example accepts `GOODMEM_CHAT_MODEL=provider:model`. Install the
  `agents` extra and your chosen provider's LangChain package separately.
- Searches run once, pagination uses the SDK, and retrieval diagnostics are preserved.
- CI exercises real LangGraph execution over mock HTTP, standard LangChain suites,
  and distributions. Optional live tests create and clean up their own spaces.

### Migrating from 0.1

This is a clean API break. Import public components from `langgraph_goodmem` or
`langgraph_goodmem.tools`. Individual tool modules and `GoodMemClient` are removed;
use `goodmem.Goodmem` for direct API access or client injection.

| 0.1 usage | 0.2 usage |
| --- | --- |
| `GoodMemCreateMemory(...).invoke({"text_content": text, ...})` | Use `original_content`. Creation waits for that memory by default; `wait=False` returns immediately. |
| Retrieval `query`, comma-separated `space_ids`, `max_results` | Use `message`, a list of `space_ids`, and `requested_size`. `max_results` limits post-processing results. |
| `wait_for_indexing` on searches | Removed. Wait during ingestion or call `wait_for_memory` with an accepted memory ID. |
| JSON strings with `success`, `results`, `totalResults` | Native dictionaries/lists with SDK snake_case fields. LangChain serializes these into `ToolMessage` content when invoked by an agent. |
| Retrieval wrapper containing chunks only | Raw SDK events include `retrieved_item`, `memory_definition`, `abstract_reply`, and `status`. Inspect statuses, or use `GoodMemRetriever` for Documents and known-failure exceptions. |
| Create-or-reuse space and chunking arguments | Creation creates a new space with SDK defaults. List/select existing spaces explicitly; configure advanced chunking through the SDK. |
| `public_read` or arbitrary space metadata | Removed. Update labels with `merge_labels` / `replace_labels`. |
| Get-memory automatically downloads content | Set `include_content=True` for the SDK's base64 `original_content` field. |
| Errors returned as `success=False` | Standard tool exceptions; use LangGraph's `handle_tool_errors` or LangChain's `handle_tool_error` as appropriate. |

The shared package still uses a synchronous SDK client; async framework calls use
a thread executor. File-upload and administrative tools retain the authority of
the configured SDK client. Only give them to workflows that need that authority.

## 0.1.0

Initial release with standalone HTTP tools.
