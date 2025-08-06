from backend.schemas.state_schema import State, MessageClassifier
from backend.core.config import llm
from langchain.schema.messages import SystemMessage, HumanMessage

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