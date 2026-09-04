from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from src.documind.config import GEMINI_API_KEY
from src.documind.agent.prompts import SYSTEM_PROMPT
from src.documind.agent.tools import search_documents

llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash",api_key=GEMINI_API_KEY)

tools=[search_documents]
memory=MemorySaver()

agent_executor = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    system_prompt=SYSTEM_PROMPT,
)

if __name__ == "__main__":
    
    config = {"configurable": {"thread_id": "test_user_1"}}
    
    print("--- TURN 1: NORMAL CHAT ---")
    response1 = agent_executor.invoke({"messages": [("user", "Hi, my name is Alex.")]}, config)
    print(response1["messages"][-1].content)
    
    print("\n--- TURN 2: FORCING A TOOL CALL ---")
    response2 = agent_executor.invoke({"messages": [("user", "Search my documents and tell me about the Q3 profits.")]}, config)
    print(response2["messages"][-1].content)



