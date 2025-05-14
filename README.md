# Lifesight RAG Agent

A Retrieval-Augmented Generation (RAG) system for answering questions about Lifesight marketing measurement documentation. It uses Qdrant for vector storage, Sentence Transformers for embeddings, and OpenAI's GPT models for answer generation. A Streamlit UI is provided for interactive Q&A.

## Features
- Ingests documentation into Qdrant vector database
- Retrieves relevant document chunks using semantic search
- Generates answers using OpenAI GPT models, grounded in documentation
- Simple Streamlit web interface for user interaction

## Requirements
- Python 3.8+
- Qdrant running locally (default: `localhost:6333`)
- OpenAI API key

Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup
1. **Start Qdrant**
   - You can run Qdrant locally using Docker:
     ```bash
     docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
     ```
2. **Set OpenAI API Key**
   - Export your OpenAI API key as an environment variable:
     ```bash
     export OPENAI_API_KEY=your-api-key-here
     ```

## Data Ingestion
To ingest documentation into Qdrant:
```bash
python scripts/ingest_to_qdrant.py
```
- By default, it looks for markdown files in `data/readme_lifesight/docs/METHODOLOGIES`.
- You can modify the `DATA_DIR` in `scripts/ingest_to_qdrant.py` to point to your documentation.

## Running the UI
To launch the Streamlit web app:
```bash
streamlit run agent/ui_app.py
```
- Open your browser at the provided local URL to interact with the Q&A system.

## Project Structure
```
rag_agent/
├── agent/
│   ├── agentic_system.py      # Main RAG agent logic
│   ├── rag_retriever.py       # Qdrant-based retriever
│   └── ui_app.py              # Streamlit UI
├── scripts/
│   └── ingest_to_qdrant.py    # Data ingestion script
├── data/
│   └── readme_lifesight/      # Documentation data (git submodule)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Notes
- The `data/readme_lifesight` directory is a git submodule. To clone with submodules:
  ```bash
  git clone --recurse-submodules <repo-url>
  ```
- If you already cloned without submodules, run:
  ```bash
  git submodule update --init --recursive
  ```
- Ensure Qdrant is running before ingesting data or running the UI.
- The OpenAI API key is required for answer generation.

## Contributing
Pull requests and issues are welcome!

## License
[MIT](LICENSE)
