from langchain_core.tools import tool

@tool
def search_documents(query:str)-> str:
    """Use this tool to search the knowledge base when the user asks a question about their 
    documents."""
    print(f"\n[AGENT DECISION] The LLM decided to use 'search_documents' for query: {query}\n")

    return "Dummy document snippet: The Q3 profits were up 15% due to new software sales."

