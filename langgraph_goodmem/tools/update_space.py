"""GoodMem Update Space tool."""

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langgraph_goodmem._client import GoodMemClient


class UpdateSpaceInput(BaseModel):
    """Input schema for the GoodMem Update Space tool."""

    space_id: str = Field(description="The UUID of the space to update.")
    name: str | None = Field(
        default=None,
        description="New name for the space (must be unique per owner).",
    )
    public_read: bool | None = Field(
        default=None,
        description="Whether the space should be readable by anyone.",
    )
    replace_labels: dict[str, str] | None = Field(
        default=None,
        description=("If provided, replaces the entire label set with this mapping."),
    )
    merge_labels: dict[str, str] | None = Field(
        default=None,
        description=(
            "If provided, merges these labels into the existing label set "
            "without removing other labels."
        ),
    )


class GoodMemUpdateSpace(BaseTool):
    """Update mutable fields on a GoodMem space.

    Only `name`, `publicRead`, and labels are mutable. Embedders and chunking
    config are immutable after creation.
    """

    name: str = "goodmem_update_space"
    description: str = (
        "Update mutable fields on a GoodMem space (name, publicRead, labels). "
        "Embedders and chunking config cannot be changed after creation."
    )
    args_schema: type[BaseModel] = UpdateSpaceInput

    goodmem_base_url: str = Field(description="GoodMem API base URL.")
    goodmem_api_key: str = Field(description="GoodMem API key.")
    goodmem_verify_ssl: bool = Field(
        default=True, description="Whether to verify SSL certificates."
    )

    def _run(
        self,
        space_id: str,
        name: str | None = None,
        public_read: bool | None = None,
        replace_labels: dict[str, str] | None = None,
        merge_labels: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Update a space.

        Args:
            space_id: The space UUID.
            name: New space name.
            public_read: New publicRead value.
            replace_labels: Replacement label set.
            merge_labels: Labels to merge into the existing set.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            JSON string with the updated space.
        """
        client = GoodMemClient(
            base_url=self.goodmem_base_url,
            api_key=self.goodmem_api_key,
            verify_ssl=self.goodmem_verify_ssl,
        )
        try:
            space = client.update_space(
                space_id=space_id,
                name=name,
                public_read=public_read,
                replace_labels=replace_labels,
                merge_labels=merge_labels,
            )
            result: dict[str, Any] = {
                "success": True,
                "spaceId": space.get("spaceId", space_id),
                "space": space,
                "message": "Space updated successfully",
            }
        except Exception as e:
            result = {"success": False, "error": str(e)}
        return json.dumps(result)
