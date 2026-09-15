"""LangChain's standard contracts for the components exported to LangGraph."""

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import httpx
import pytest
from goodmem import Goodmem
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import RetrieversIntegrationTests
from langchain_tests.unit_tests import ToolsUnitTests

from langgraph_goodmem import GoodMemRetrieveMemories, GoodMemRetriever
from tests.unit_tests.conftest import CHUNK, MEMORY, ndjson


class TestRetrievalTool(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return GoodMemRetrieveMemories

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"message": "refund", "space_ids": ["space-1"]}


class TestRetriever(RetrieversIntegrationTests):
    @pytest.fixture(autouse=True)
    def configure(self) -> Iterator[None]:
        chunks = []
        for index in range(3):
            chunk = deepcopy(CHUNK)
            chunk["retrievedItem"]["chunk"]["chunk"]["chunkId"] = f"chunk-{index}"
            chunks.append(chunk)
        with httpx.Client(
            base_url="https://goodmem.test",
            transport=httpx.MockTransport(
                lambda request: ndjson(*chunks, {"memoryDefinition": MEMORY})
            ),
        ) as http:
            self.parameters = {
                "client": Goodmem(http_client=http),
                "space_ids": ["space-1"],
            }
            yield

    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        return GoodMemRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        return self.parameters

    @property
    def retriever_query_example(self) -> str:
        return "What is the refund policy?"
