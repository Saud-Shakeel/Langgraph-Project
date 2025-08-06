from backend.schemas.state_schema import State
from backend.core.config import llm
from backend.tools.tools import tool_list
from langchain.schema import SystemMessage, AIMessage


# Logical node
def logical_agent(state: State) -> dict:
    llm_with_tools = llm.bind_tools(tool_list)

    preview = llm_with_tools.invoke([
        SystemMessage(content="You are a logical assistant. Use tools when possible. Otherwise, answer briefly."),
        *state["messages"]
    ])

    tool_suggestion = None
    if preview.tool_calls:
        tool_suggestion = preview.tool_calls[0].get("name")

    return {
        "messages": [preview],
        "tool_suggestion": tool_suggestion
    }
