from langchain_core.tools import tool
from src.documind.retrieval.retriever import DocumentRetriever

# We instantiate the retriever outside the function so it only connects to the database ONCE.
# This makes the tool lightning fast when the LLM calls it.
doc_retriever = DocumentRetriever()

@tool
def search_documents(query: str) -> str:
    """Use this tool to search the knowledge base when the user asks a question about their documents."""
    print(f"\n[AGENT DECISION] The LLM decided to use 'search_documents' for query: {query}\n")
    
    # Actually run the semantic search against our ChromaDB!
    results = doc_retriever.retrieve_context(query, top_k=2)
    return results
