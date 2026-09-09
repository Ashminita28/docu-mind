from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique identifier for the user's chat session to maintain memory.")
    message: str = Field(..., description="The message sent by the user.")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The agent's text response.")
