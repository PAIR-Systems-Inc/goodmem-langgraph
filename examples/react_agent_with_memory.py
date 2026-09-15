"""An agent that searches a configured GoodMem space and cites its sources.

Install `langgraph-goodmem[agents]` and your model provider's LangChain package.
Set GOODMEM_BASE_URL, GOODMEM_API_KEY, GOODMEM_SPACE_ID, the provider's credentials,
and GOODMEM_CHAT_MODEL to a provider:model identifier. For example, install
langchain-anthropic, set ANTHROPIC_API_KEY and choose an anthropic:model identifier.
Optional GOODMEM_RERANKER_ID enables reranking.
"""

import os
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool

from langgraph_goodmem import GoodMemRetriever


def build_agent(model: str | BaseChatModel, retriever: GoodMemRetriever) -> Any:
    """Give an agent a query-only tool with a developer-configured search scope."""
    search = create_retriever_tool(
        retriever,
        "search_policies",
        "Search company policies for questions about refunds and returns.",
        document_prompt=PromptTemplate.from_template(
            "Source: {source}\n{page_content}"
        ),
        response_format="content_and_artifact",
    )
    return create_agent(
        model=model,
        tools=[search],
        system_prompt=(
            "Search the company policies before answering. Cite the source URLs "
            "returned by the tool. If there is no evidence or the search fails, say so."
        ),
    )


def main() -> None:
    retriever = GoodMemRetriever(
        space_ids=[os.environ["GOODMEM_SPACE_ID"]],
        reranker_id=os.getenv("GOODMEM_RERANKER_ID"),
        k=5,
    )
    agent = build_agent(os.environ["GOODMEM_CHAT_MODEL"], retriever)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the refund policy?"}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
