"""GoodMem Retrieve Memories tool."""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langgraph_goodmem._client import GoodMemClient


class RetrieveMemoriesInput(BaseModel):
    """Input schema for the GoodMem Retrieve Memories tool."""

    query: str = Field(
        description=(
            "A natural language query used to find semantically similar memory chunks."
        ),
    )
    space_ids: str = Field(
        description=(
            "One or more space UUIDs to search across, separated by commas "
            "(e.g., 'id1,id2')."
        ),
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of matching chunks to return.",
    )
    include_memory_definition: bool = Field(
        default=True,
        description=(
            "Fetch the full memory metadata (source document info, "
            "processing status) alongside the matched chunks."
        ),
    )
    wait_for_indexing: bool = Field(
        default=True,
        description=(
            "Retry for up to 60 seconds when no results are found. "
            "Enable this when memories were just added and may still be "
            "undergoing chunking and embedding."
        ),
    )
    reranker_id: str | None = Field(
        default=None,
        description=(
            "UUID of a reranker model to refine the order of retrieved "
            "chunks via direct query-chunk scoring."
        ),
    )
    llm_id: str | None = Field(
        default=None,
        description=(
            "UUID of an LLM that will produce a contextual summary "
            "(`abstractReply`) over the retrieved chunks."
        ),
    )
    relevance_threshold: float | None = Field(
        default=None,
        description=(
            "Minimum relevance score (0-1) below which results are dropped. "
            "Only applied when a post-processor is configured."
        ),
    )
    llm_temperature: float | None = Field(
        default=None,
        description=(
            "Creativity setting for LLM generation (0-2). Only used when "
            "`llm_id` is also provided."
        ),
    )
    chronological_resort: bool | None = Field(
        default=None,
        description=(
            "Reorder final results by creation time after reranking and thresholding."
        ),
    )


class GoodMemRetrieveMemories(BaseTool):
    """Perform similarity-based semantic retrieval from GoodMem spaces.

    Returns matching chunks ranked by relevance, with optional full memory
    definitions. Supports polling for indexing completion.

    Setup:
        Install `langgraph-goodmem` and set environment variables:

        .. code-block:: bash

            pip install langgraph-goodmem
            export GOODMEM_API_KEY="your-api-key"
            export GOODMEM_BASE_URL="http://localhost:8080"

    Instantiate:
        .. code-block:: python

            from langgraph_goodmem import GoodMemRetrieveMemories

            tool = GoodMemRetrieveMemories(
                goodmem_base_url="http://localhost:8080",
                goodmem_api_key="your-api-key",
            )

    Invocation:
        .. code-block:: python

            result = tool.invoke(
                {
                    "query": "machine learning best practices",
                    "space_ids": "space-uuid-1,space-uuid-2",
                    "max_results": 10,
                }
            )
    """

    name: str = "goodmem_retrieve_memories"
    description: str = (
        "Perform similarity-based semantic retrieval across one or more "
        "GoodMem spaces. Returns matching chunks ranked by relevance."
    )
    args_schema: type[BaseModel] = RetrieveMemoriesInput

    goodmem_base_url: str = Field(description="GoodMem API base URL.")
    goodmem_api_key: str = Field(description="GoodMem API key.")
    goodmem_verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates."
    )

    def _run(
        self,
        query: str,
        space_ids: str,
        max_results: int = 5,
        include_memory_definition: bool = True,
        wait_for_indexing: bool = True,
        reranker_id: str | None = None,
        llm_id: str | None = None,
        relevance_threshold: float | None = None,
        llm_temperature: float | None = None,
        chronological_resort: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """Retrieve semantically similar memories.

        Args:
            query: The search query.
            space_ids: Comma-separated space UUIDs.
            max_results: Maximum results to return.
            include_memory_definition: Include full memory metadata.
            wait_for_indexing: Poll for indexing completion.
            reranker_id: Optional reranker UUID for result refinement.
            llm_id: Optional LLM UUID for `abstractReply` generation.
            relevance_threshold: Minimum score (0-1) for inclusion.
            llm_temperature: LLM creativity (0-2).
            chronological_resort: Reorder final results by creation time.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            JSON string with the retrieval results.
        """
        client = GoodMemClient(
            base_url=self.goodmem_base_url,
            api_key=self.goodmem_api_key,
            verify_ssl=self.goodmem_verify_ssl,
        )
        try:
            result = client.retrieve_memories(
                query=query,
                space_ids=space_ids,
                max_results=max_results,
                include_memory_definition=include_memory_definition,
                wait_for_indexing=wait_for_indexing,
                reranker_id=reranker_id,
                llm_id=llm_id,
                relevance_threshold=relevance_threshold,
                llm_temperature=llm_temperature,
                chronological_resort=chronological_resort,
            )
        except Exception as e:
            result = {"success": False, "error": str(e)}
        return json.dumps(result)
