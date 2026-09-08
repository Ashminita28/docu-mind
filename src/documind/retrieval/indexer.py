import logging
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from src.documind.config import GEMINI_API_KEY


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self, data_dir: str = "data/raw", db_path: str = "./chroma_data"):
        self.data_dir = Path(data_dir)
        self.db_path = db_path
        
        
        self.embed_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-2", api_key=GEMINI_API_KEY)
        
        
        self.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    def build_index(self) -> VectorStoreIndex | None:
        if not self.data_dir.exists():
            logger.error(f"Data directory {self.data_dir} does not exist.")
            return None

        logger.info(f"Loading documents from {self.data_dir}...")
        documents = SimpleDirectoryReader(input_dir=str(self.data_dir)).load_data()
        
        if not documents:
            logger.warning("No documents found to index.")
            return None

       
        db_client = chromadb.PersistentClient(path=self.db_path)
        chroma_collection = db_client.get_or_create_collection("documind_kb")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        logger.info(f"Building VectorStoreIndex for {len(documents)} documents...")
        
       
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            transformations=[self.node_parser]
        )
        
        logger.info("Indexing complete.")
        return index

if __name__ == "__main__":
    indexer = DocumentIndexer()
    indexer.build_index()
