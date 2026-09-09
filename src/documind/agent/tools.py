import logging
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from src.documind.retrieval.retriever import DocumentRetriever
from src.documind.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)


doc_retriever = DocumentRetriever()


expander_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash", 
    api_key=GEMINI_API_KEY, 
    temperature=0.0
)


class QueryVariations(BaseModel):
    queries: list[str] = Field(..., description="A list of 3 distinct, highly descriptive search variations of the original user query.")

@tool
def search_documents(query: str) -> str:
    """Use this tool to search the knowledge base when the user asks a question about their documents."""
    print(f"\n[AGENT DECISION] Multi-Query Search initiated for: '{query}'")
    
    try:
        
        prompt = f"You are a search query optimizer. The user asked: '{query}'. Generate 3 distinct, highly descriptive variations of this query that would be better suited for a vector database search. Replace pronouns with specific nouns, and add relevant corporate synonyms."
        
        structured_expander = expander_llm.with_structured_output(QueryVariations)
        variations = structured_expander.invoke(prompt)
        
    
        all_queries = [query] + variations.queries
        print(f"   [Query Expansion] Searching variations: {variations.queries}\n")
        
        all_chunks = set() 
        
        for q in all_queries:
          
            results_string = doc_retriever.retrieve_context(q, top_k=2)
            chunks = results_string.split("\n\n---\n\n")
            for chunk in chunks:
                if chunk.strip(): 
                    all_chunks.add(chunk.strip())
                    
        combined_context = "\n\n---\n\n".join(list(all_chunks))
        return combined_context

    except Exception as e:
        logger.error(f"Error during Multi-Query search: {e}")
        return doc_retriever.retrieve_context(query, top_k=2)
