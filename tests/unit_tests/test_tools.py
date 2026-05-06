"""Unit tests for GoodMem LangGraph tool wrappers."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

from langgraph_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteMemory,
    GoodMemGetMemory,
    GoodMemListEmbedders,
    GoodMemListSpaces,
    GoodMemRetrieveMemories,
)

_COMMON_KWARGS: dict[str, Any] = {
    "goodmem_base_url": "http://localhost:8080",
    "goodmem_api_key": "test-key",
    "goodmem_verify_ssl": False,
}


class TestToolInstantiation:
    def test_create_space_is_base_tool(self) -> None:
        tool = GoodMemCreateSpace(**_COMMON_KWARGS)
        assert tool.name == "goodmem_create_space"
        assert tool.args_schema is not None

    def test_create_memory_is_base_tool(self) -> None:
        tool = GoodMemCreateMemory(**_COMMON_KWARGS)
        assert tool.name == "goodmem_create_memory"

    def test_retrieve_memories_is_base_tool(self) -> None:
        tool = GoodMemRetrieveMemories(**_COMMON_KWARGS)
        assert tool.name == "goodmem_retrieve_memories"

    def test_get_memory_is_base_tool(self) -> None:
        tool = GoodMemGetMemory(**_COMMON_KWARGS)
        assert tool.name == "goodmem_get_memory"

    def test_delete_memory_is_base_tool(self) -> None:
        tool = GoodMemDeleteMemory(**_COMMON_KWARGS)
        assert tool.name == "goodmem_delete_memory"

    def test_list_spaces_is_base_tool(self) -> None:
        tool = GoodMemListSpaces(**_COMMON_KWARGS)
        assert tool.name == "goodmem_list_spaces"

    def test_list_embedders_is_base_tool(self) -> None:
        tool = GoodMemListEmbedders(**_COMMON_KWARGS)
        assert tool.name == "goodmem_list_embedders"


class TestCreateSpaceTool:
    @patch("langgraph_goodmem.tools.create_space.GoodMemClient")
    def test_invoke_success(self, mock_cls: MagicMock) -> None:
        mock_client = mock_cls.return_value
        mock_client.create_space.return_value = {
            "success": True,
            "spaceId": "s1",
            "name": "test",
            "embedderId": "e1",
            "message": "Space created successfully",
            "reused": False,
        }
        tool = GoodMemCreateSpace(**_COMMON_KWARGS)
        raw = tool.invoke({"name": "test", "embedder_id": "e1"})
        result = json.loads(raw)
        assert result["success"] is True
        assert result["spaceId"] == "s1"

    @patch("langgraph_goodmem.tools.create_space.GoodMemClient")
    def test_invoke_error_returns_json(self, mock_cls: MagicMock) -> None:
        mock_client = mock_cls.return_value
        mock_client.create_space.side_effect = RuntimeError("connection failed")
        tool = GoodMemCreateSpace(**_COMMON_KWARGS)
        raw = tool.invoke({"name": "test", "embedder_id": "e1"})
        result = json.loads(raw)
        assert result["success"] is False
        assert "connection failed" in result["error"]


class TestCreateMemoryTool:
    @patch("langgraph_goodmem.tools.create_memory.GoodMemClient")
    def test_invoke_text(self, mock_cls: MagicMock) -> None:
        mock_client = mock_cls.return_value
        mock_client.create_memory.return_value = {
            "success": True,
            "memoryId": "m1",
            "spaceId": "s1",
            "status": "PENDING",
            "contentType": "text/plain",
            "message": "Memory created successfully",
        }
        tool = GoodMemCreateMemory(**_COMMON_KWARGS)
        raw = tool.invoke({"space_id": "s1", "text_content": "hello"})
        result = json.loads(raw)
        assert result["success"] is True
        assert result["memoryId"] == "m1"

    @patch("langgraph_goodmem.tools.create_memory.GoodMemClient")
    def test_invoke_error_returns_json(self, mock_cls: MagicMock) -> None:
        mock_client = mock_cls.return_value
        mock_client.create_memory.side_effect = ValueError("No content")
        tool = GoodMemCreateMemory(**_COMMON_KWARGS)
        raw = tool.invoke({"space_id": "s1"})
        result = json.loads(raw)
        assert result["success"] is False


class TestRetrieveMemoriesTool:
    @patch("langgraph_goodmem.tools.retrieve_memories.GoodMemClient")
    def test_invoke_success(self, mock_cls: MagicMock) -> None:
        mock_client = mock_cls.return_value
        mock_client.retrieve_memories.return_value = {
            "success": True,
            "resultSetId": "rs1",
            "results": [{"chunkText": "hello"}],
            "memories": [],
            "totalResults": 1,
            "query": "hello",
        }
        tool = GoodMemRetrieveMemories(**_COMMON_KWARGS)
        raw = tool.invoke(
            {
                "query": "hello",
                "space_ids": "s1",
            }
        )
        result = json.loads(raw)
        assert result["success"] is True
        assert result["totalResults"] == 1


class TestGetMemoryTool:
    @patch("langgraph_goodmem.tools.get_memory.GoodMemClient")
    def test_invoke_success(self, mock_cls: MagicMock) -> None:
        mock_client = mock_cls.return_value
        mock_client.get_memory.return_value = {
            "success": True,
            "memory": {"memoryId": "m1"},
        }
        tool = GoodMemGetMemory(**_COMMON_KWARGS)
        raw = tool.invoke({"memory_id": "m1"})
        result = json.loads(raw)
        assert result["success"] is True
        assert result["memory"]["memoryId"] == "m1"


class TestDeleteMemoryTool:
    @patch("langgraph_goodmem.tools.delete_memory.GoodMemClient")
    def test_invoke_success(self, mock_cls: MagicMock) -> None:
        mock_client = mock_cls.return_value
        mock_client.delete_memory.return_value = {
            "success": True,
            "memoryId": "m1",
            "message": "Memory deleted successfully",
        }
        tool = GoodMemDeleteMemory(**_COMMON_KWARGS)
        raw = tool.invoke({"memory_id": "m1"})
        result = json.loads(raw)
        assert result["success"] is True


class TestListSpacesTool:
    @patch("langgraph_goodmem.tools.list_spaces.GoodMemClient")
    def test_invoke_success(self, mock_cls: MagicMock) -> None:
        mock_client = mock_cls.return_value
        mock_client.list_spaces.return_value = [{"spaceId": "s1"}]
        tool = GoodMemListSpaces(**_COMMON_KWARGS)
        raw = tool.invoke({})
        result = json.loads(raw)
        assert result["success"] is True
        assert result["totalSpaces"] == 1


class TestListEmbeddersTool:
    @patch("langgraph_goodmem.tools.list_embedders.GoodMemClient")
    def test_invoke_success(self, mock_cls: MagicMock) -> None:
        mock_client = mock_cls.return_value
        mock_client.list_embedders.return_value = [{"embedderId": "e1"}]
        tool = GoodMemListEmbedders(**_COMMON_KWARGS)
        raw = tool.invoke({})
        result = json.loads(raw)
        assert result["success"] is True
        assert result["totalEmbedders"] == 1
