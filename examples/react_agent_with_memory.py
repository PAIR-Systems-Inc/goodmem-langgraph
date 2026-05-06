"""ReAct agent with GoodMem long-term memory.

This example builds a LangGraph ReAct agent that can store and retrieve
information using GoodMem as its long-term memory backend. The agent can:

  - List available embedder models
  - Create a memory space (or reuse an existing one)
  - Store text as a memory
  - Semantically search stored memories

Setup (run from the repo root):
    pip install -e libs/partners/goodmem/ langchain-openai

Environment variables:
    GOODMEM_BASE_URL    GoodMem server URL (e.g. https://localhost:8080)
    GOODMEM_API_KEY     GoodMem API key
    GOODMEM_VERIFY_SSL  Set to "false" for self-signed certs (default: true)
    OPENAI_API_KEY      OpenAI API key for the agent's LLM
"""

import json
import os
import sys
from typing import Any

from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
from langgraph.prebuilt import create_react_agent  # type: ignore[import-not-found]

from langgraph_goodmem import (
    GoodMemCreateMemory,
    GoodMemCreateSpace,
    GoodMemDeleteMemory,
    GoodMemDeleteSpace,
    GoodMemGetMemory,
    GoodMemGetSpace,
    GoodMemListEmbedders,
    GoodMemListMemories,
    GoodMemListSpaces,
    GoodMemRetrieveMemories,
    GoodMemUpdateSpace,
)

# -- Configuration -----------------------------------------------------------

GOODMEM_BASE_URL = os.environ["GOODMEM_BASE_URL"]
GOODMEM_API_KEY = os.environ["GOODMEM_API_KEY"]
GOODMEM_VERIFY_SSL = os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() == "true"

goodmem_kwargs: dict[str, Any] = {
    "goodmem_base_url": GOODMEM_BASE_URL,
    "goodmem_api_key": GOODMEM_API_KEY,
    "goodmem_verify_ssl": GOODMEM_VERIFY_SSL,
}

# -- Tools -------------------------------------------------------------------

tools = [
    GoodMemListEmbedders(**goodmem_kwargs),
    GoodMemListSpaces(**goodmem_kwargs),
    GoodMemGetSpace(**goodmem_kwargs),
    GoodMemCreateSpace(**goodmem_kwargs),
    GoodMemUpdateSpace(**goodmem_kwargs),
    GoodMemDeleteSpace(**goodmem_kwargs),
    GoodMemCreateMemory(**goodmem_kwargs),
    GoodMemListMemories(**goodmem_kwargs),
    GoodMemGetMemory(**goodmem_kwargs),
    GoodMemRetrieveMemories(**goodmem_kwargs),
    GoodMemDeleteMemory(**goodmem_kwargs),
]

# -- Agent -------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a research assistant with long-term memory powered by GoodMem.

When the user asks you to remember something:
1. If no space exists yet, list the available embedders, pick one, and create
   a space called "research-notes".
2. Store the information as a memory in that space.

When the user asks a question that might be answered by prior memories:
1. Search the space with a semantic query.
2. Use the retrieved chunks to answer, citing the relevant text.

Always tell the user what you stored or found.
"""

llm = ChatOpenAI(model="gpt-4o-mini")

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)

# -- Run ---------------------------------------------------------------------


def verify_connection() -> None:
    """Check GoodMem connectivity before starting the agent."""
    tool = GoodMemListEmbedders(**goodmem_kwargs)
    raw = tool.invoke({})
    result = json.loads(raw)
    if not result.get("success"):
        print(f"ERROR: Cannot connect to GoodMem at {GOODMEM_BASE_URL}")
        print(f"  GOODMEM_VERIFY_SSL={GOODMEM_VERIFY_SSL}")
        print(f"  Error: {result.get('error', 'unknown')}")
        print("\nVerify your environment variables are set correctly.")
        sys.exit(1)
    count = result.get("totalEmbedders", 0)
    print(f"Connected to GoodMem at {GOODMEM_BASE_URL} ({count} embedders available)")


if __name__ == "__main__":
    verify_connection()
    print("GoodMem ReAct Agent (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        result = agent.invoke({"messages": [("user", user_input)]})

        # The last AI message contains the agent's final response
        ai_message = result["messages"][-1]
        print(f"\nAssistant: {ai_message.content}\n")
