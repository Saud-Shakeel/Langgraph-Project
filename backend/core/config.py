from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(override=True)

# Initialize the LLM
llm = init_chat_model(model="gpt-4o-mini")

# Memory
memory = MemorySaver()
configuration = {"configurable": {"thread_id": "1"}}