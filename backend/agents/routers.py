from backend.schemas.state_schema import State

# Router
def router(state: State) -> dict:
    return {"next": "therapist" if state["message_type"] == "emotional" else "logical"}