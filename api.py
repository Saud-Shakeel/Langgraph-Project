from fastapi import FastAPI
import uvicorn
from router import router

app = FastAPI(title="Multi Agent Chatbot")
app.include_router(router=router)

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000)
