import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.documind.api.schemas import ChatRequest, ChatResponse
from src.documind.agent.graph import agent_executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuMind API",
    description="API for the DocuMind RAG Agent",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Send a message to the DocuMind Agent.
    Maintains memory across requests using the session_id.
    """
    logger.info(f"Received message for session {request.session_id}")
    
    try:
        config = {"configurable": {"thread_id": request.session_id}}
        
        result = agent_executor.invoke(
            {"messages": [("user", request.message)]}, 
            config=config
        )
        
        raw_content = result["messages"][-1].content
        
        if isinstance(raw_content, list):
            final_message = "\n".join(block.get("text", "") for block in raw_content if isinstance(block, dict))
        else:
            final_message = str(raw_content)
            
        return ChatResponse(response=final_message)
        
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting DocuMind API server on http://localhost:8000")
    uvicorn.run("src.documind.api.server:app", host="0.0.0.0", port=8000, reload=True)
