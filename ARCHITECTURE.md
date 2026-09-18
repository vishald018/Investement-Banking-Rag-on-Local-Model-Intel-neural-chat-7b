# Architecture

## Investment Banker Chatbot — RAG Architecture

This document explains the architecture, data flow, components, and request lifecycle of the Investment Banker Chatbot.

The application is a **Retrieval-Augmented Generation (RAG)** based chatbot that uses financial documents as a knowledge source and a locally hosted Large Language Model (LLM) to generate answers.

---

## 1. Project Overview

The Investment Banker Chatbot combines:

- Document ingestion
- Text chunking
- Semantic embeddings
- Vector database storage
- Similarity-based retrieval
- Prompt construction
- Local LLM inference
- FastAPI backend
- HTML/Jinja2 frontend

The primary goal is to allow users to ask questions about the information contained in the provided financial documents without requiring the entire documents to be passed to the LLM for every query.

### Core RAG Flow

```text
Documents
    ↓
Document Loading
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
ChromaDB Vector Store
    ↓
User Query
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Relevant Context
    ↓
Prompt Construction
    ↓
Local LLM
    ↓
Generated Answer


                    ┌──────────────────┐
                    │   User Question  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     FastAPI      │
                    │ /get_response    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ BGE-Large        │
                    │ Embeddings       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    ChromaDB      │
                    │ Vector Search    │
                    │      k = 1       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Retrieved Chunk  │
                    └────────┬─────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Prompt + Context + Question│
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Neural Chat 7B   │
                    │ Local GGUF Model │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Answer + Source  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Frontend      │
                    └──────────────────┘