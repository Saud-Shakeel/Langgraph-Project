from fastapi import APIRouter
from multiAgentChatbot import run_chatbot

router = APIRouter()

router.add_api_route("/therapist-logical-MultiAgent", run_chatbot, methods=["POST"])