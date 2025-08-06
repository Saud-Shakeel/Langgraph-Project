from backend.graphs.builder import graph
from langsmith import traceable
from backend.core.config import configuration
from langchain.schema.messages import HumanMessage

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