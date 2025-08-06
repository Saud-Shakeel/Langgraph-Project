from langgraph.graph import StateGraph, START, END
from backend.core.config import memory
from langgraph.prebuilt import tools_condition, ToolNode
from backend.schemas.state_schema import State
from backend.agents.logical_agent import logical_agent
from backend.agents.therpaist_agent import therapist_agent
from backend.agents.routers import router
from backend.models.message_classifier import classify_message
from backend.tools.tools import tool_list


# LangGraph Build
builder = StateGraph(State)
builder.add_node("classifier", classify_message)
builder.add_node("routers", router)
builder.add_node("logical", logical_agent)
builder.add_node("therapist", therapist_agent)
builder.add_node("tools", ToolNode(tool_list))

builder.add_edge(START, "classifier")
builder.add_edge("classifier", "routers")
builder.add_conditional_edges("routers", lambda s: s["next"], {
    "logical": "logical",
    "therapist": "therapist"
})
builder.add_conditional_edges("logical", tools_condition)
builder.add_edge("tools", "logical")
builder.add_edge("therapist", END)

graph = builder.compile(checkpointer=memory)
