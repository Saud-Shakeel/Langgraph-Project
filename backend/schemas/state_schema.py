from pydantic import BaseModel, Field
from typing import Literal, TypedDict, Annotated
from langchain.schema.messages import AnyMessage
from langgraph.graph import add_messages

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
