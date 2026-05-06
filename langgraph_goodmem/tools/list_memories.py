"""GoodMem List Memories tool."""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langgraph_goodmem._client import GoodMemClient


class ListMemoriesInput(BaseModel):
    """Input schema for the GoodMem List Memories tool."""

    space_id: str = Field(
        description="The UUID of the space whose memories to list.",
    )
    max_results: int | None = Field(
        default=None,
        description=(
            "Maximum number of memories to return per page. Server clamps "
            "to a sensible range; omit to use the server default."
        ),
    )
    next_token: str | None = Field(
        default=None,
        description=(
            "Opaque pagination token from a previous list_memories response. "
            "Pass it back to fetch the next page."
        ),
    )
    status_filter: str | None = Field(
        default=None,
        description=(
            "Optional filter by processing status: PENDING, PROCESSING, "
            "COMPLETED, or FAILED."
        ),
    )
    include_content: bool = Field(
        default=False,
        description="Whether to include the original content for each memory.",
    )
    filter_expression: str | None = Field(
        default=None,
        description=(
            "Optional metadata filter expression using the GoodMem filter "
            "syntax (e.g. \"val('$.source') = 'email'\")."
        ),
    )


class GoodMemListMemories(BaseTool):
    """List memories within a GoodMem space.

    Supports pagination via `next_token` and filtering by processing status
    or metadata expression.
    """

    name: str = "goodmem_list_memories"
    description: str = (
        "List memories in a GoodMem space, with optional pagination, "
        "status filtering, and metadata filter expressions."
    )
    args_schema: type[BaseModel] = ListMemoriesInput

    goodmem_base_url: str = Field(description="GoodMem API base URL.")
    goodmem_api_key: str = Field(description="GoodMem API key.")
    goodmem_verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates."
    )

    def _run(
        self,
        space_id: str,
        max_results: int | None = None,
        next_token: str | None = None,
        status_filter: str | None = None,
        include_content: bool = False,
        filter_expression: str | None = None,
        **kwargs: Any,
    ) -> str:
        """List memories in a space.

        Args:
            space_id: The space UUID.
            max_results: Page size.
            next_token: Pagination token.
            status_filter: Processing status filter.
            include_content: Whether to include original content.
            filter_expression: Metadata filter expression.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            JSON string with the memories list and `nextToken`.
        """
        client = GoodMemClient(
            base_url=self.goodmem_base_url,
            api_key=self.goodmem_api_key,
            verify_ssl=self.goodmem_verify_ssl,
        )
        try:
            result = client.list_memories(
                space_id=space_id,
                max_results=max_results,
                next_token=next_token,
                status_filter=status_filter,
                include_content=include_content,
                filter_expression=filter_expression,
            )
        except Exception as e:
            result = {"success": False, "error": str(e)}
        return json.dumps(result)
