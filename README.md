# Investment Banking RAG on Local Model

A local Retrieval-Augmented Generation chatbot for investment banking and technical questions using Intel Neural Chat 7B, BGE Large embeddings, ChromaDB, LangChain, and FastAPI.

## Architecture

```text
PDF Documents
      |
      v
PyPDFLoader
      |
      v
Text Splitting
      |
      v
BGE Large Embeddings
      |
      v
ChromaDB
      |
      v
Similarity Search
      |
      v
Relevance Check
     / \
    /   \
Relevant  Not Relevant
   |          |
   v          v
 RAG      Neural Chat 7B
   |          |
   +----+-----+
        |
        v
     FastAPI
        |
        v
  Web Interface
```

## How It Works

The application uses a hybrid approach:

- Relevant questions are answered using the information retrieved from the PDF documents through RAG.
- Questions without sufficiently relevant document context are answered directly by the local Neural Chat 7B model.

## Tech Stack

- Python
- FastAPI
- LangChain
- ChromaDB
- BAAI BGE Large Embeddings
- Intel Neural Chat 7B
- CTransformers
- PyPDF
- HTML
- CSS
- JavaScript
- Bootstrap

## Project Structure

```text
Investment-Banker-Chatbot/
│
├── Data/
│   ├── 400 Questions & Technicals.pdf
│   └── WSP_RedBook_Sample.pdf
│
├── templates/
│   └── index.html
│
├── fastapi_app.py
├── ingest.py
├── requirements.txt
├── ARCHITECTURE.md
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/vishald018/Investement-Banking-Rag-on-Local-Model-Intel-neural-chat-7b.git
cd Investement-Banking-Rag-on-Local-Model-Intel-neural-chat-7b
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Neural Chat Model

Download:

```text
neural-chat-7b-v3-1.Q4_K_M.gguf
```

Place it in the project root directory. The model file is excluded from GitHub because of its large size.

### 5. Create the Vector Store

Run:

```bash
python ingest.py
```

This loads the PDFs from `Data/`, creates embeddings using `BAAI/bge-large-en`, and stores them in ChromaDB.

### 6. Run the Application

```bash
python -m uvicorn fastapi_app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Models

**Embedding Model**

```text
BAAI/bge-large-en
```

**Language Model**

```text
Intel Neural Chat 7B v3.1
neural-chat-7b-v3-1.Q4_K_M.gguf
```

The language model runs locally through CTransformers.

## Data

The current knowledge base contains:

- `400 Questions & Technicals.pdf`
- `WSP_RedBook_Sample.pdf`

Additional PDF documents can be added to the `Data/` directory and indexed using `ingest.py`.

## Author

Vishal Dixit
