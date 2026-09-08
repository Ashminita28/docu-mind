import logging

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from src.documind.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Service class to handle vector database retrieval operations."""
    
    def __init__(self, db_path: str = "./chroma_data", collection_name: str = "documind_kb"):
       
        self.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-2", api_key=GEMINI_API_KEY)
        
        
        try:
            db_client = chromadb.PersistentClient(path=db_path)
            chroma_collection = db_client.get_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model
            )
            logger.info("Successfully connected to ChromaDB Vector Store.")
        except Exception as e:
            logger.error(f"Failed to connect to Vector Store: {e}")
            raise

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieves the top_k most relevant chunks for a given query."""
        logger.info(f"Retrieving context for query: '{query}'")
        
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        
        if not nodes:
            return "No relevant documents found."

        
        compiled_context = "\n\n---\n\n".join(
            [f"Source (Score: {node.get_score()}):\n{node.text}" for node in nodes]
        )
        return compiled_context
