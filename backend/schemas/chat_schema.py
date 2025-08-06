from pydantic import BaseModel, Field
from typing import Optional

class chatRequest(BaseModel):
    message: str = Field(..., description="Query that the user enters to the LLM.")
    tool_approval: Optional[str] = Field(default=None, description="Returns the tool message based on the user's approval.")

class chatResponse(BaseModel):
    reply: str = Field(..., description="Response that the model returns to the user.")
