from backend.schemas.state_schema import State
from backend.core.config import llm
from langchain.schema import SystemMessage

# Therapist node
def therapist_agent(state: State) -> dict:
    resp = llm.invoke([
        SystemMessage(content="""
            You are a kind, supportive assistant. Respond briefly with warmth and empathy.
        """),
        state["messages"][-1]
    ])
    return {"messages": [resp]}