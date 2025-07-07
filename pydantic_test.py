from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class Person(BaseModel):
    id: int = Field(ge=0)
    name: str
    salary: Optional[float] = None

p = Person(id="1", name="saud", salary=1800)

