"""The two entry points must share implementations and future fixes."""

import langchain_goodmem
import langchain_goodmem.tools

import langgraph_goodmem
import langgraph_goodmem.tools


def test_public_exports_are_shared_implementations() -> None:
    assert set(langgraph_goodmem.__all__) <= set(langchain_goodmem.__all__)
    for name in langgraph_goodmem.__all__:
        assert getattr(langgraph_goodmem, name) is getattr(langchain_goodmem, name)
    assert set(langgraph_goodmem.tools.__all__) <= set(langchain_goodmem.tools.__all__)
    for name in langgraph_goodmem.tools.__all__:
        assert getattr(langgraph_goodmem.tools, name) is getattr(
            langchain_goodmem.tools, name
        )
