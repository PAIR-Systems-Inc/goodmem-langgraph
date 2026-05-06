"""Test that all public symbols can be imported."""

from langgraph_goodmem import (
    GoodMemClient,
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteMemory,
    GoodMemGetMemory,
    GoodMemListEmbedders,
    GoodMemListSpaces,
    GoodMemRetrieveMemories,
)


def test_all_imports() -> None:
    assert GoodMemClient is not None
    assert GoodMemCreateSpace is not None
    assert GoodMemCreateMemory is not None
    assert GoodMemRetrieveMemories is not None
    assert GoodMemGetMemory is not None
    assert GoodMemDeleteMemory is not None
    assert GoodMemListEmbedders is not None
    assert GoodMemListSpaces is not None
