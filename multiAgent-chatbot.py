from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage
from langchain.chat_models import init_chat_model
from langchain.schema import SystemMessage, HumanMessage
from typing import Annotated, Literal
from pydantic_test import BaseModel, Field
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import display, Image

load_dotenv(override=True)

# Initialize the LLM
model_llm = "gpt-4o-mini"
llm = init_chat_model(model=model_llm)

# Pydantic model for classification output
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ..., description="Classify if the message requires an emotional or a logical response."
    )

# Define the shared state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    message_type: str | None
    next: str | None

# Ticket price tool
@tool("get_ticket_price", description="Return a mock ticket price for a DESTINATION CITY.")
def get_ticket_price(destination_city: str)->str:
    ticket_prices = {"Dubai": "$456.9", "Islamabad": "$100.0", "Tokyo": "$561.2",
                     "Mumbai": "$200.2"}
    return ticket_prices.get(destination_city.lower(), "0.0")

# Stock price tool
@tool("get_stock_price", description="Return a mock stock price for the given COMPANY NAME.")
def get_stock_price(company_name: str)->float:
    stock_symbols = {"Microsoft": 250.2, "Apple": 350.5, "Google": 500.0, "Amazon": 400.7}

    return stock_symbols.get(company_name, 0.0)

tool_list = [get_stock_price, get_ticket_price]

# Classify the user's message
def classify_message(state: State):
    last_message = state["messages"][-1].content
    llm_classifier = llm.with_structured_output(MessageClassifier)

    result = llm_classifier.invoke([
        {
            "role": "system",
            "content": """
                Classify the user message as either:
                - 'logical': if it asks for facts, logical analysis, reasoning, information, or practical implications.
                - 'emotional': if it asks for emotional support, therapy, or deals with feelings, etc.
            """
        },
        {
            "role": "user",
            "content": last_message
        }
    ])
    return {"message_type": result.message_type}

# Route based on classification
def router(state: State):
    message_type = state.get("message_type")
    return {"next": "therapist" if message_type == "emotional" else "logical"}

# Logical agent response
def logical_agent(state: State):
    llm_with_tools = llm.bind_tools(tool_list)
    response = llm_with_tools.invoke([
        SystemMessage(content=
        """You are a logical, fact‑driven assistant.
        - Provide step‑by‑step reasoning.
        - If a tool returns a more accurate answer, call it first (use JSON args).
        - Otherwise reply concisely."""
        ),
        *state["messages"]
    ])
    return {"messages": [response]}

# Therapist (emotional) agent response
def therapist_agent(state: State):
    response = llm.invoke([
        {
            "role": "system",
            "content": """
                You are an emotionally intelligent assistant. Respond with empathy, warmth, and emotional awareness.
                Adapt your tone based on the user's mood. Prioritize emotional support, encouragement, and understanding.
            """
        },
        state["messages"][-1]
    ])
    return {"messages": [response]}


# Build the LangGraph
builder = StateGraph(State)
builder.add_node("classifier", classify_message)
builder.add_node("router", router)
builder.add_node("logical", logical_agent)
builder.add_node("therapist", therapist_agent)
builder.add_node("tools", ToolNode(tool_list))

builder.add_edge(START, "classifier")
builder.add_edge("classifier", "router")
builder.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {
        "logical": "logical",
        "therapist": "therapist"
    }
)
builder.add_conditional_edges(
    "logical",
    tools_condition
)
builder.add_edge("tools", "logical")
builder.add_edge("logical", END)
builder.add_edge("therapist", END)

graph = builder.compile()

# CLI loop
def run_chatbot():
    state = {"messages": [], "message_type": None, "next": None}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye! 👋")
            break

        # Add user message
        state["messages"].append(HumanMessage(content=user_input))

        # Run graph
        state = graph.invoke(state)

        # Print assistant reply
        assistant_msg = state["messages"][-1].content
        print("Assistant:", assistant_msg)

def draw_graph():
    with open("langgraph_diagram.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

if __name__ == "__main__":
    run_chatbot()
    draw_graph()

