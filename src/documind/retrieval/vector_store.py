import os
import chromadb

client=chromadb.PersistentClient(path="./chroma_data")
collection=client.get_or_create_collection(name="documind_kb")

def ingest_documents():
    """Reads the files in data/raw and saves them to the database."""
    print("Reading documents from data/raw/...")
    raw_dir="data/raw"
    documents=[]
    ids=[]
    for filename in os.listdir(raw_dir):
        if filename.endswith(".txt"):
            filepath=os.path.join(raw_dir,filename)
            with open(filepath,"r",encoding="utf-8") as f:
                content=f.read()
                documents.append(content)
                ids.append(filename)

    collection.add(documents=documents,ids=ids)
    print(f"Successfully ingested {len(documents)} documents into ChromaDB!")

def search_knowledge_base(query:str):
    """Searches the databse by meaning , not exact keywords."""
    print(f"\nSearching for : '{query}'")
    results=collection.query(query_texts=[query],n_results=1)
    print("---MOST RELEVANT DOCUMENT FOUND ---")
    print(results["documents"][0][0])

if __name__=="__main__":
    ingest_documents()
    search_knowledge_base("How much money did we make this year?")
    