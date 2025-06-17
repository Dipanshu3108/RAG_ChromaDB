import pandas as pd
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
import os
# this uses the chroma db.
# APIS
os.environ["LANGSMITH_TRACING"] = "true"


DATA_PATH = "filtered_urban_data.csv" 
CHROMA_DB_PATH = "./urban_dict_chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE =  2500 # smaller batch size to avoid memory issues, chroma db exceeds limit


def create_documents_from_df_batch(df_batch):
    """Converts a batch from a DataFrame to a list of LangChain Document objects."""
    documents = []
    for _, row in df_batch.iterrows():
        content = f"Word: {row['word']}\nDefinition: {row['definition']}"
        metadata = {
            "word": row['word'],
            # Test 1: without definition in metadata, Test 2: with definition in metadata
            # Note: Storing the full definition in metadata is redundant but can be useful.
            # "definition": row['definition'], 
            "source": "urban_dictionary",
            "row_id": str(row.name) # Use the DataFrame index as a string ID
        }
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    return documents


def main():
    # --- 1. Load Data ---
    print(f"Loading data from {DATA_PATH}...")
    # Using an iterator to read the CSV in chunks to manage memory
    df_iterator = pd.read_csv(DATA_PATH, chunksize=BATCH_SIZE, nrows=100000)
    total_rows = 100000 #currely hardcoded for testing; can be adjusted or calculated dynamically
    # total_rows = sum(1 for row in open(DATA_PATH, 'r', encoding='utf-8')) -1 # get total rows for tqdm
    print(f"Data load complete")
    
    # --- 2. Setup Embeddings and Text Splitter ---
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Use 'cuda' for GPU
        encode_kwargs={'normalize_embeddings': False}
    )
    print("Embedding model loaded successfully.")

    print("Setting up text splitter...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    print("Text splitter initialized.")
    # --- 3. Initialize ChromaDB ---
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Chroma DB already exists at {CHROMA_DB_PATH}. Skipping build process.")
        print("To rebuild, delete the directory and run this script again.")
        return

    print(f"Initializing new Chroma DB at: {CHROMA_DB_PATH}")
    # Initialize with the first document to create the collection
    vector_store = None

    # --- 4. Process data in batches and populate DB ---
    print("Processing data in batches and building vector store...")
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        for i, df_batch in enumerate(df_iterator):
            # Create LangChain documents
            docs_batch = create_documents_from_df_batch(df_batch)
            
            # Split documents 
            split_docs = text_splitter.split_documents(docs_batch)
            
            if not split_docs:
                pbar.update(len(df_batch))
                continue

            # Add to Chroma
            if vector_store is None:
                # First batch: create the database
                vector_store = Chroma.from_documents(
                    documents=split_docs,
                    embedding=embeddings,
                    persist_directory=CHROMA_DB_PATH
                )
            else:
                # Subsequent batches: add to the existing database
                vector_store.add_documents(split_docs)
            
            pbar.update(len(df_batch))
    
    print("\nVector store creation complete!")
    print(f"Total documents in store: {vector_store._collection.count()}")

if __name__ == "__main__":
    main()