"""The two entry points must share implementations and future fixes."""

import goodmem_langchain
import goodmem_langchain.tools

import goodmem_langgraph
import goodmem_langgraph.tools


def test_public_exports_are_shared_implementations() -> None:
    assert set(goodmem_langgraph.__all__) <= set(goodmem_langchain.__all__)
    for name in goodmem_langgraph.__all__:
        assert getattr(goodmem_langgraph, name) is getattr(goodmem_langchain, name)
    assert set(goodmem_langgraph.tools.__all__) <= set(goodmem_langchain.tools.__all__)
    for name in goodmem_langgraph.tools.__all__:
        assert getattr(goodmem_langgraph.tools, name) is getattr(
            goodmem_langchain.tools, name
        )
