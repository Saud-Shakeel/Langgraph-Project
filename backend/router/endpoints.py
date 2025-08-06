from backend.agents.therpaist_agent import therapist_agent
from backend.schemas.chat_schema import chatRequest, chatResponse
from fastapi import APIRouter
from backend.agents.logical_agent import logical_agent
from langchain.schema.messages import HumanMessage
from backend.models.message_classifier import classify_message
from backend.agents.routers import router
from backend.graphs.builder import graph
from backend.core.config import configuration

route = APIRouter()

@route.post("/chat", response_model=chatResponse)
def send_message(request: chatRequest):
    user_message = request.message
    user_tool_approval = request.tool_approval

    state = {
        "messages": [HumanMessage(content=user_message)],
        "message_type": None,
        "next": None
    }

    message_type = classify_message(state)
    state["message_type"] = message_type["message_type"]
    next_agent = router(state)
    state["next"] = next_agent["next"]

    if state["next"] == "logical":
        logical_agent_call = logical_agent(state)
        tool_name = logical_agent_call.get("tool_suggestion")

        if tool_name:
            print(f"I can use the {tool_name} for this task. Do you want to use it (yes/no)?")
            if user_tool_approval and user_tool_approval.lower() == "yes":
                tool_response = graph.invoke(state, config=configuration)
                message = tool_response.get("messages")
                if message:
                    reply = message[-1].content
                else:
                    reply = "The tool didn't respond correctly."
            else:
                reply = "I don’t have access to real-time data. If you want help in any other thing, do let me know."

            return chatResponse(reply=reply)

        # No tool needed, return agent reply directly
        return chatResponse(reply=logical_agent_call["messages"][-1].content)

    elif state["next"] == "therapist":
        therapist_agent_call = therapist_agent(state)
        return chatResponse(reply=therapist_agent_call["messages"][-1].content)

    else:
        return chatResponse(reply="Something went wrong while processing your request.")
