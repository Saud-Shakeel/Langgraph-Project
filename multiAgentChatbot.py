from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langsmith import  traceable

load_dotenv(override=True)

# Initialize the LLM
llm = init_chat_model(model="gpt-4o-mini")

# Classification schema
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description= "Classify if the message requires an emotional or a logical response."
    )

# State type
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    message_type: str | None
    next: str | None

# Memory
memory = MemorySaver()
configuration = {"configurable": {"thread_id": "1"}}

# Tools
@tool("get_ticket_price", description="Return a mock ticket price for a DESTINATION CITY.")
def get_ticket_price(destination_city: str) -> str:
    prices = {"dubai": "$456.9", "islamabad": "$100.0", "tokyo": "$561.2", "mumbai": "$200.2"}
    return prices.get(destination_city.lower(), "ticket price not available")

@tool("get_stock_price", description="Return a mock stock price for the given COMPANY NAME.")
def get_stock_price(company_name: str) -> str:
    stocks = {"Microsoft": "250.2", "Apple": "350.5", "Google": "500.0", "Amazon": "400.7"}
    return stocks.get(company_name, "stock price not available")

tool_list = [get_stock_price, get_ticket_price]

# Classifier node
def classify_message(state: State) -> dict:
    last = state["messages"][-1].content
    classifier = llm.with_structured_output(MessageClassifier)
    result = classifier.invoke([
        SystemMessage(content="""
            Classify this user message as 'logical' or 'emotional'.
        """),
        HumanMessage(content=last)
    ])
    return {"message_type": result.message_type}

# Router
def router(state: State) -> dict:
    return {"next": "therapist" if state["message_type"] == "emotional" else "logical"}


# Logical node
def logical_agent(state: State) -> dict:
    llm_with_tools = llm.bind_tools(tool_list)

    # Try to predict whether a tool would be used
    tool_call_preview = llm_with_tools.invoke([
        SystemMessage(content="""
            You are a logical assistant. Use tools when possible. Otherwise, answer briefly.
        """),
        *state["messages"]
    ])

    if tool_call_preview.tool_calls:
        tool_name = tool_call_preview.tool_calls[0].get("name")
        print(f"Assistant: I can use the tool '{tool_name}' to help with this. Do you want me to use it? (yes/no)")
        user_input = input("User: ").strip().lower()

        if user_input not in ("yes"):
            return {
                "messages": [
                    AIMessage(content="I don’t have access to real-time data. If you want help in any other thing, do let me know.")
                ]
            }

    # Proceed with actual invocation if tool is allowed or not needed
    resp = llm_with_tools.invoke([
        SystemMessage(content="""
            You are a logical assistant. Use tools when possible. Otherwise, answer briefly.
        """),
        *state["messages"]
    ])
    return {"messages": [resp]}


# Therapist node
def therapist_agent(state: State) -> dict:
    resp = llm.invoke([
        SystemMessage(content="""
            You are a kind, supportive assistant. Respond briefly with warmth and empathy.
        """),
        state["messages"][-1]
    ])
    return {"messages": [resp]}

# LangGraph Build
builder = StateGraph(State)
builder.add_node("classifier", classify_message)
builder.add_node("router", router)
builder.add_node("logical", logical_agent)
builder.add_node("therapist", therapist_agent)
builder.add_node("tools", ToolNode(tool_list))

builder.add_edge(START, "classifier")
builder.add_edge("classifier", "router")
builder.add_conditional_edges("router", lambda s: s["next"], {
    "logical": "logical",
    "therapist": "therapist"
})
builder.add_conditional_edges("logical", tools_condition)
builder.add_edge("tools", "logical")
builder.add_edge("therapist", END)

graph = builder.compile(checkpointer=memory)

# CLI runner
@traceable
def run_chatbot():
    state = {"messages": [], "message_type": None, "next": None}
    while True:
        user_input = input("User: ")
        if user_input.lower().strip() in ("exit", "quit"):
            print("Assistant: Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = graph.invoke(state, config=configuration)

        print(f"Assistant: {state['messages'][-1].content}")


def draw_graph():
    with open("langgraph_diagram.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

if __name__=="__main__":
    run_chatbot()


