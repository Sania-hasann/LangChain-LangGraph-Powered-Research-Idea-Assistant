import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'

from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Dict, Any
import arxiv
import time
import re

def construct_arxiv_query(topic: str, domain: str = None) -> str:
    """Construct an ArXiv query using the topic and domain"""
    # Clean and prepare search terms
    topic = topic.strip().lower()
    domain = domain.strip().lower() if domain else None
    
    # Build the query
    if domain:
        query = f'"{topic}" AND "{domain}"'
    else:
        query = f'"{topic}"'
    
    return query

def retrieve_literature(topic: str, domain: str = None, max_results: int = 10) -> List[Document]:
    """Retrieve relevant papers from ArXiv using the arxiv package"""
    try:
        # Construct the query
        query = construct_arxiv_query(topic, domain)
        print(f"\nSearching ArXiv with query: {query}")
        
        # Create the search
        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,  # Get more papers initially
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        # Collect papers
        papers = []
        for result in search.results():
            try:
                # Create document from paper
                doc = Document(
                    page_content=result.summary,
                    metadata={
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "published": result.published.strftime("%Y-%m-%d"),
                        "id": result.entry_id.split('/')[-1],
                        "url": result.entry_id,
                        "pdf_url": result.pdf_url,
                        "primary_category": result.primary_category,
                        "categories": result.categories
                    }
                )
                papers.append(doc)
                print(f"Retrieved: {result.title}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing paper: {e}")
                continue
        
        if not papers:
            print("No papers found. Please try different search terms.")
            return []
        
        print(f"\nSuccessfully retrieved {len(papers)} papers")
        return papers
        
    except Exception as e:
        print(f"Error retrieving papers from ArXiv: {e}")
        return []

def embed_documents(docs: List[Document]):
    """Create embeddings and vector store for the documents"""
    try:
        # Split documents into smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = splitter.split_documents(docs)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore, embeddings
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None, None

def analyze_papers(docs: List[Document], vectorstore, embeddings) -> Dict[str, Any]:
    """Analyze papers to identify research gaps and trends"""
    analysis = {
        "topics": set(),
        "methods": set(),
        "applications": set(),
        "gaps": set()
    }
    
    # Extract key information from papers
    for doc in docs:
        content = doc.page_content.lower()
        
        # Extract topics (looking for common research terms)
        research_terms = re.findall(r'\b(?:novel|new|proposed|introduced|developed)\s+([a-z\s]+?)(?:method|approach|technique|framework|system|model)\b', content)
        analysis["topics"].update(term.strip() for term in research_terms)
        
        # Extract methods
        methods = re.findall(r'\b(?:using|based on|implemented with)\s+([a-z\s]+?)(?:method|approach|technique|framework|system|model)\b', content)
        analysis["methods"].update(method.strip() for method in methods)
        
        # Extract applications
        applications = re.findall(r'\b(?:applied to|used in|tested on)\s+([a-z\s]+?)(?:domain|field|area|application)\b', content)
        analysis["applications"].update(app.strip() for app in applications)
    
    # Identify potential gaps
    # 1. Find topics that appear in few papers
    topic_counts = {}
    for doc in docs:
        content = doc.page_content.lower()
        for topic in analysis["topics"]:
            if topic in content:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    # Add under-researched topics as gaps
    for topic, count in topic_counts.items():
        if count < 2:  # Topic appears in less than 2 papers
            analysis["gaps"].add(f"Limited research on {topic}")
    
    # 2. Find methods that could be applied to more domains
    for method in analysis["methods"]:
        method_domains = set()
        for doc in docs:
            content = doc.page_content.lower()
            if method in content:
                # Extract domain from the paper
                domain_match = re.search(r'\b(?:in|for|within)\s+([a-z\s]+?)(?:domain|field|area)\b', content)
                if domain_match:
                    method_domains.add(domain_match.group(1).strip())
        
        if len(method_domains) < 2:
            analysis["gaps"].add(f"Potential to apply {method} to more domains")
    
    return analysis

def generate_research_ideas(analysis: Dict[str, Any], domain: str) -> List[Dict[str, str]]:
    """Generate research ideas based on the analysis"""
    ideas = []
    
    # Generate ideas for each identified gap
    for gap in analysis["gaps"]:
        idea = {
            "gap": gap,
            "title": f"Novel Research in {domain}: {gap}",
            "description": f"Research focusing on {gap} within the {domain} domain",
            "potential_methods": list(analysis["methods"])[:3],  # Use top 3 methods
            "related_topics": list(analysis["topics"])[:3]  # Use top 3 topics
        }
        ideas.append(idea)
    
    return ideas 