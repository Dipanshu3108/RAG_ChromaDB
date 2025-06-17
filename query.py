import warnings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import pandas as pd

# Suppress a specific UserWarning from ChromaDB
warnings.filterwarnings("ignore", category=UserWarning, message="Given Chroma collection is not persistent.")

# --- Configuration ---
CHROMA_DB_PATH = "./urban_dict_chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "deepseek-r1:1.5b" 

def main():
    # --- 1. Load the Existing Vector Store ---
    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Use 'cuda' for GPU
    )
    
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    print("Vector store loaded successfully.")

    # --- 2. Setup Retriever ---
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3} # Retrieve top 3 most relevant definitions for now
    )

    # --- 3. Setup the Local LLM ( via Ollama) ---
    # Make sure the Ollama application is running with the model
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    
    # --- 4. Setup the RAG Chain ---
    prompt_template = """
    You are an expert on Urban Dictionary slang and informal language. Use the following Urban Dictionary definitions to answer the question.

    Context from Urban Dictionary:
    {context}

    Question: {question}

    Instructions:
    - Provide a clear, accurate answer based on the Urban Dictionary definitions provided.
    - Do not show your thinking process, reasoning steps, or any text surrounded by <think> tags.
    - If the context contains multiple definitions for a term, try to synthesize them into one coherent answer.
    - If the context doesn't contain relevant information, just say "I couldn't find a definition for that in my database."

    Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    # --- 5. Interactive Query Loop ---
    print("\n--- Urban Dictionary RAG is Ready ---")
    print("Ask a question, or type 'exit' to quit.")
    while True:
        question = input("\n> ")
        if question.lower() == 'exit':
            break

        print("Thinking...")
        result = qa_chain.invoke({"query": question})
        
        print("\n--- Answer ---")
        print(result["result"])
        
        print("\n--- Sources ---")
        for doc in result["source_documents"]:
            print(f"- Word: {doc.metadata.get('word', 'N/A')}")
            print(f"  Content: {doc.page_content[:150]}...") # Print a snippet
        print("\n" + "-"*50)


if __name__ == "__main__":
    
    # listing a list of some example words for users
    # Show 25 random words
    sample_data = pd.read_csv('filtered_urban_data.csv', nrows=100000)  # for 100k rows for user to select from
    random_words = sample_data['word'].sample(25)
    
    print("\n25 Random Words from Dataset:")
    print(", ".join(random_words))
    print("\n" + "-"*50)
    main()