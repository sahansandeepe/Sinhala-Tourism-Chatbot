
# Sinhala Tourism Chatbot

A Sinhala-language tourism chatbot developed using SinLlama, Retrieval-Augmented Generation (RAG), and a Neo4j knowledge graph.

## Project Overview

This project focuses on providing Sinhala-language answers to questions about tourist destinations in Sri Lanka.

The chatbot retrieves relevant tourism information from a Neo4j knowledge graph and uses a locally running SinLlama language model to generate Sinhala responses.

### Supported Destinations

- Kandy
- Ella
- Sigiriya

## Technologies Used

- Python
- SinLlama (Llama 3 8B GGUF)
- llama-cpp-python
- Neo4j Knowledge Graph
- Retrieval-Augmented Generation (RAG)
- Visual Studio Code

## System Architecture

1. Sinhala User Question
2. Destination Identification
3. Neo4j Graph Retrieval
4. Context Validation
5. SinLlama Answer Generation
6. Final Sinhala Answer

## Project Structure

```text
Sinhala_Tourism_Chatbot/
├── final_chatbot.py
├── sinllama_integration.py
├── tourism_graph_retrieval.py
├── source_graphrag/
├── models/              # Local model (not uploaded)
├── .env                 # Private configuration
├── .gitignore
└── README.md
```

## Installation

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the main dependencies:

```bash
pip install llama-cpp-python neo4j python-dotenv truststore
```

## Configuration

Create a `.env` file in the project root:

```env
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

Place the locally obtained SinLlama GGUF model at:

`models/SinLlama.Q2_K.gguf`

The model and credentials are not included in this repository.

The Neo4j database must also contain the required tourism data and relationships.

## Run the Chatbot

```bash
python final_chatbot.py --question "සීගිරිය ඉදිකරන ලද්දේ කුමන රජු විසින්ද?"
```

## Hybrid GraphRAG Research

The `source_graphrag/` directory contains experimental files related to Sigiriya tourism information retrieval, answer refinement, and evaluation.

These files represent research and development work and are not presented as a fully integrated production system.

## Future Improvements

- Improve retrieval accuracy
- Expand tourism information
- Improve Sinhala answer quality
- Further evaluate Hybrid GraphRAG integration

## Author

Sahan Sandeepa

BSc (Hons) in Data Science – SLTC

Internship Project – SLT-MOBITEL

## GitHub Repository

https://github.com/sahansandeepe/Sinhala-Tourism-Chatbot
