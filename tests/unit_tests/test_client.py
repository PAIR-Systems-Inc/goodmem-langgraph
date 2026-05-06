"""Unit tests for GoodMemClient with mocked HTTP responses."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from langgraph_goodmem._client import GoodMemClient


@pytest.fixture()
def client() -> GoodMemClient:
    return GoodMemClient(
        base_url="http://localhost:8080",
        api_key="test-key",
        timeout=5.0,
        verify_ssl=False,
    )


def _mock_response(
    status_code: int = 200,
    json_data: dict | list | None = None,
    text: str = "",
    content_type: str = "application/json",
) -> httpx.Response:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    if json_data is not None:
        resp.json.return_value = json_data
        resp.text = json.dumps(json_data)
    else:
        resp.text = text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


class TestClientInit:
    def test_base_url_trailing_slash_stripped(self) -> None:
        c = GoodMemClient(base_url="http://example.com/", api_key="k")
        assert c.base_url == "http://example.com"

    def test_headers_include_api_key(self, client: GoodMemClient) -> None:
        headers = client._headers()
        assert headers["X-API-Key"] == "test-key"
        assert headers["Content-Type"] == "application/json"

    def test_ndjson_headers(self, client: GoodMemClient) -> None:
        headers = client._ndjson_headers()
        assert headers["Accept"] == "application/x-ndjson"


class TestListSpaces:
    def test_list_spaces_returns_list(self, client: GoodMemClient) -> None:
        spaces = [{"spaceId": "s1", "name": "test"}]
        mock_resp = _mock_response(json_data=spaces)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            result = client.list_spaces()
        assert result == spaces

    def test_list_spaces_returns_dict_with_spaces_key(
        self, client: GoodMemClient
    ) -> None:
        body = {"spaces": [{"spaceId": "s1"}]}
        mock_resp = _mock_response(json_data=body)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            result = client.list_spaces()
        assert result == [{"spaceId": "s1"}]

    def test_list_spaces_http_error(self, client: GoodMemClient) -> None:
        mock_resp = _mock_response(status_code=500)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            with pytest.raises(httpx.HTTPStatusError):
                client.list_spaces()


class TestCreateSpace:
    def test_create_new_space(self, client: GoodMemClient) -> None:
        list_resp = _mock_response(json_data=[])
        create_resp = _mock_response(
            json_data={"spaceId": "new-id", "name": "my-space"}
        )
        with (
            patch.object(httpx.Client, "get", return_value=list_resp),
            patch.object(httpx.Client, "post", return_value=create_resp),
        ):
            result = client.create_space(name="my-space", embedder_id="emb-1")
        assert result["success"] is True
        assert result["spaceId"] == "new-id"
        assert result["reused"] is False

    def test_create_space_reuses_existing(self, client: GoodMemClient) -> None:
        existing = [{"spaceId": "existing-id", "name": "my-space"}]
        list_resp = _mock_response(json_data=existing)
        with patch.object(httpx.Client, "get", return_value=list_resp):
            result = client.create_space(name="my-space", embedder_id="emb-1")
        assert result["reused"] is True
        assert result["spaceId"] == "existing-id"

    def test_create_space_none_chunking(self, client: GoodMemClient) -> None:
        list_resp = _mock_response(json_data=[])
        create_resp = _mock_response(json_data={"spaceId": "id", "name": "s"})
        with (
            patch.object(httpx.Client, "get", return_value=list_resp),
            patch.object(httpx.Client, "post", return_value=create_resp) as mock_post,
        ):
            client.create_space(name="s", embedder_id="e", chunking_strategy="none")
        call_json = mock_post.call_args.kwargs["json"]
        assert call_json["defaultChunkingConfig"] == {"none": {}}


class TestCreateMemory:
    def test_create_text_memory(self, client: GoodMemClient) -> None:
        mock_resp = _mock_response(
            json_data={
                "memoryId": "m1",
                "spaceId": "s1",
                "processingStatus": "PENDING",
            }
        )
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            result = client.create_memory(space_id="s1", text_content="hello world")
        assert result["success"] is True
        assert result["memoryId"] == "m1"

    def test_create_memory_empty_string_accepted(self, client: GoodMemClient) -> None:
        mock_resp = _mock_response(
            json_data={
                "memoryId": "m2",
                "spaceId": "s1",
                "processingStatus": "PENDING",
            }
        )
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            result = client.create_memory(space_id="s1", text_content="")
        assert result["success"] is True

    def test_create_memory_no_content_raises(self, client: GoodMemClient) -> None:
        with pytest.raises(ValueError, match="No content provided"):
            client.create_memory(space_id="s1")

    def test_create_memory_with_file(
        self, client: GoodMemClient, tmp_path: Path
    ) -> None:
        f = tmp_path / "test.txt"
        f.write_text("file content")
        mock_resp = _mock_response(
            json_data={
                "memoryId": "m3",
                "spaceId": "s1",
                "processingStatus": "PENDING",
            }
        )
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            result = client.create_memory(space_id="s1", file_path=str(f))
        assert result["success"] is True


class TestGetMemory:
    def test_get_memory_with_content(self, client: GoodMemClient) -> None:
        mem_resp = _mock_response(json_data={"memoryId": "m1", "spaceId": "s1"})
        content_resp = _mock_response(json_data={"text": "hello"})
        with patch.object(httpx.Client, "get", side_effect=[mem_resp, content_resp]):
            result = client.get_memory(memory_id="m1")
        assert result["success"] is True
        assert result["content"] == {"text": "hello"}

    def test_get_memory_content_error_captured(self, client: GoodMemClient) -> None:
        mem_resp = _mock_response(json_data={"memoryId": "m1", "spaceId": "s1"})
        err_resp = _mock_response(status_code=404)
        with patch.object(httpx.Client, "get", side_effect=[mem_resp, err_resp]):
            result = client.get_memory(memory_id="m1")
        assert result["success"] is True
        assert "contentError" in result


class TestDeleteMemory:
    def test_delete_memory(self, client: GoodMemClient) -> None:
        mock_resp = _mock_response(json_data={})
        with patch.object(httpx.Client, "delete", return_value=mock_resp):
            result = client.delete_memory(memory_id="m1")
        assert result["success"] is True
        assert result["memoryId"] == "m1"


class TestListEmbedders:
    def test_list_embedders_list_response(self, client: GoodMemClient) -> None:
        embedders = [{"embedderId": "e1", "name": "test"}]
        mock_resp = _mock_response(json_data=embedders)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            result = client.list_embedders()
        assert result == embedders

    def test_list_embedders_dict_response(self, client: GoodMemClient) -> None:
        body = {"embedders": [{"embedderId": "e1"}]}
        mock_resp = _mock_response(json_data=body)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            result = client.list_embedders()
        assert result == [{"embedderId": "e1"}]


class TestRetrieveMemories:
    def test_retrieve_memories_parses_ndjson(self, client: GoodMemClient) -> None:
        ndjson = "\n".join(
            [
                json.dumps(
                    {
                        "retrievedItem": {
                            "chunk": {
                                "chunk": {
                                    "chunkId": "c1",
                                    "chunkText": "hello",
                                    "memoryId": "m1",
                                },
                                "relevanceScore": 0.95,
                                "memoryIndex": 0,
                            }
                        }
                    }
                ),
                json.dumps({"resultSetBoundary": {"resultSetId": "rs1"}}),
            ]
        )
        mock_resp = _mock_response(text=ndjson)
        mock_resp.json = MagicMock(side_effect=Exception("not json"))  # type: ignore[method-assign]
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            result = client.retrieve_memories(
                query="hello",
                space_ids="s1",
                wait_for_indexing=False,
            )
        assert result["success"] is True
        assert result["totalResults"] == 1
        assert result["results"][0]["chunkText"] == "hello"
        assert result["resultSetId"] == "rs1"

    def test_retrieve_memories_empty_space_ids_raises(
        self, client: GoodMemClient
    ) -> None:
        with pytest.raises(ValueError, match="At least one valid Space ID"):
            client.retrieve_memories(query="q", space_ids="", wait_for_indexing=False)

    def test_retrieve_memories_handles_data_prefix(self, client: GoodMemClient) -> None:
        ndjson = "data:" + json.dumps(
            {
                "retrievedItem": {
                    "chunk": {
                        "chunk": {
                            "chunkId": "c1",
                            "chunkText": "world",
                            "memoryId": "m1",
                        },
                        "relevanceScore": 0.8,
                        "memoryIndex": 0,
                    }
                }
            }
        )
        mock_resp = _mock_response(text=ndjson)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            result = client.retrieve_memories(
                query="world",
                space_ids="s1",
                wait_for_indexing=False,
            )
        assert result["totalResults"] == 1
