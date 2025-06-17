# Urban Dictionary RAG System

A Retrieval-Augmented Generation (RAG) system that allows you to query Urban Dictionary definitions using a local vector database and language model.

## Overview

This project creates a searchable database of Urban Dictionary definitions only on fist 100k rows of the dataset and provides an interactive query interface. It uses ChromaDB for vector storage, HuggingFace embeddings for semantic search, and Ollama for local language model inference.
If you wish to use the complete dataset (~2.5 million records) here is he link https://drive.google.com/file/d/199oaSAHtmvqxo_3HTO2i78nWIgCqrpvP/view?usp=sharing 

## Features

- **Vector Database**: Build a searchable ChromaDB database from Urban Dictionary CSV data
- **Semantic Search**: Find relevant definitions using sentence transformers
- **Local LLM Integration**: Query definitions using Ollama's DeepSeek model (example models can have qwen, llama, mistral etc.)
- **Interactive Interface**: Command-line chat interface for asking questions
- **Batch Processing**: Efficiently handles large datasets 
- **Sample Discovery**: Browse words from the dataset 

## Prerequisites

### Required Software
- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Git (for cloning)

### Required Python Packages
```bash
pip install pandas langchain langchain-community langchain-huggingface 
pip install chromadb sentence-transformers tqdm huggingface-hub
```

### Required Models
- **Ollama Model**: `deepseek-r1:1.5b`
  ```bash
  ollama run deepseek-r1:1.5b
  ```
- **HuggingFace Model**: `sentence-transformers/all-mpnet-base-v2` (auto-downloaded)

## Setup

### 1. Data Preparation
You need the Urban Dictionary dataset as a CSV file named `filtered_urban_data.csv`. The CSV should have at least these columns:
- `word`: The slang term
- `definition`: The definition text

### 2. Environment Configuration
The system uses these APIs (update the keys in `build_db_chroma.py`):
- **LangSmith API**: For tracing and monitoring
- **HuggingFace API**: For model access

### 3. Build the Vector Database
```bash
python build_db_chroma.py
```

This will:
- Load the Urban Dictionary CSV data
- Create embeddings for all definitions
- Build a ChromaDB vector store at `./urban_dict_chroma_db`
- Process up to 100,000 rows in batches of 2,500

**Note**: If the database already exists, the script will skip rebuilding. Delete the `urban_dict_chroma_db` directory to rebuild.

### 4. Start Ollama
Ensure Ollama is running with the required model:
```bash
ollama run deepseek-r1:1.5b
```

## Usage

### Query the System
```bash
python query.py
```

This will:
1. Display 25 random words from the dataset for inspiration on first 100k rows
2. Load the vector database and language model
3. Start an interactive chat interface

### Example Queries
- "What does 'simp' mean?"
- "Define 'ghosting'"
- "What is 'rizz'?"
- "Explain 'sus'"

Type `exit` to quit the interface.


## Configuration

### Database Building (`build_db_chroma.py`)
- `DATA_PATH`: Path to the CSV file
- `CHROMA_DB_PATH`: Vector database storage location
- `EMBEDDING_MODEL_NAME`: HuggingFace embedding model
- `BATCH_SIZE`: Processing batch size (adjust for memory)

### Querying (`query.py`)
- `CHROMA_DB_PATH`: Vector database location
- `EMBEDDING_MODEL_NAME`: Must match the building script
- `LLM_MODEL`: Ollama model name
- `search_kwargs["k"]`: Number of definitions to retrieve per query

## System Requirements

### Memory Usage
- **Building**: Processes data in batches to manage memory
- **Querying**: Loads embeddings and model into memory
- **Recommended**: 8GB+ RAM for smooth operation

### Storage
- **Vector Database**: Size depends on dataset (expect 1-5GB for 100k entries)
- **Models**: ~1-2GB for embedding and LLM models

## Troubleshooting

### Common Issues

**"Chroma collection is not persistent" warning**
- This is suppressed and doesn't affect functionality

**Ollama connection errors**
- Ensure Ollama is running: `ollama serve`
- Verify model is available: `ollama list`

**Memory issues during building**
- Reduce `BATCH_SIZE` in `build_db_chroma.py`
- Process fewer rows by adjusting `nrows` parameter

**No results found**
- Check if the vector database was built successfully
- Verify the CSV data format matches expected columns

### Performance Optimization

**For faster building:**
- Use GPU by changing `'device': 'cpu'` to `'device': 'cuda'`
- Increase batch size if you have more memory

**For better query results:**
- Increase `k` value to retrieve more context
- Adjust the prompt template for different response styles


## Acknowledgments

- Urban Dictionary for the data source
- LangChain for the RAG framework
- ChromaDB for vector storage
- HuggingFace for embeddings
- Ollama for local LLM inference
