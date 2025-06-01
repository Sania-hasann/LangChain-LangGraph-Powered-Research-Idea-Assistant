# ResearchGap Explorer: LangChain & LangGraph Powered Research Idea Assistant

## Overview
ResearchGap Explorer is an AI-powered research assistant that leverages LangChain and LangGraph to identify research gaps and generate novel research ideas. It uses advanced natural language processing, knowledge graphs, and machine learning techniques to analyze academic papers and suggest potential research directions. The tool combines LangChain's powerful document processing capabilities with LangGraph's graph-based reasoning to provide comprehensive research insights.

## Features
- **LangChain Integration**:
  - Document processing and embedding
  - Vector store management
  - Chain-based reasoning
- **LangGraph Analysis**:
  - Graph-based knowledge representation
  - Semantic relationship mapping
  - Research gap identification
- **Literature Analysis**: Automatically retrieves and analyzes research papers from ArXiv
- **Knowledge Graph Construction**: Builds a semantic knowledge graph of research concepts, methods, and applications
- **Gap Detection**: Identifies research gaps using multiple approaches:
  - Graph-based analysis using LangGraph
  - Topic modeling
  - Concept co-occurrence
  - Semantic similarity
- **Idea Generation**: Generates novel research ideas based on identified gaps
- **Domain-Specific Analysis**: Supports analysis across different research domains
- **Interactive Interface**: User-friendly command-line interface for easy interaction

## Folder Structure
```
research-idea-generator/
├── src/
│   ├── literature.py      
│   ├── app.py            
|   ├── gap_detection.py
|   ├── idea_generation.py
|   ├── knowledge_extraction.py
|   ├── llm.py
|   ├── paper_storage.py 
│   └── __pycache__/      # Python cache files
├── papers/               # Directory for storing downloaded papers
├── saved_papers/         # Directory for cached paper data
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Technology Used
- **Core Frameworks**:
  - LangChain for document processing and chain-based reasoning
  - LangGraph for graph-based analysis and relationship mapping
- **Natural Language Processing**:
  - HuggingFace Transformers for embeddings
  - NLTK for text processing
- **Machine Learning**:
  - Gensim for topic modeling
  - FAISS for vector similarity search
- **Graph Technologies**:
  - Neo4j for knowledge graph storage (optional)
  - NetworkX for graph analysis
- **Other Tools**:
  - ArXiv API for paper retrieval
  - Streamlit for web interface (optional)

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/Sania-hasann/LangChain-LangGraph-Powered-Research-Idea-Assistant.git
cd LangChain-LangGraph-Powered-Research-Idea-Assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How It Works

1. **Input**: Provide a research topic and domain of interest
2. **Paper Retrieval**: System fetches relevant papers from ArXiv
3. **Analysis Pipeline**:
   - Text processing and embedding
   - Topic modeling
   - Knowledge graph construction
   - Gap detection
4. **Output**: Generates research ideas based on identified gaps

## Use Cases

1. **Academic Research**:
   - Identifying new research directions
   - Finding gaps in existing literature
   - Generating novel research proposals

2. **Industry R&D**:
   - Technology trend analysis
   - Innovation opportunity identification
   - Competitive research analysis

3. **Research Funding**:
   - Proposal development
   - Research gap analysis
   - Impact assessment

4. **Education**:
   - Teaching research methodology
   - Student project ideation
   - Literature review assistance

## Contributions
Contributions are welcome! Feel free to fork the repo, make improvements, and submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments

- ArXiv for providing access to research papers
- LangChain community for their excellent documentation
- HuggingFace for their transformer models
- Neo4j for graph database technology 
