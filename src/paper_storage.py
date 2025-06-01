import json
import os
from datetime import datetime
from typing import List, Dict, Any
from langchain.schema import Document
import markdown
import re

class PaperStorage:
    def __init__(self, base_dir: str = "saved_papers"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def _format_paper_content(self, content: str) -> str:
        """Format paper content for better readability"""
        # Split content into sections
        sections = content.split('\n\n')
        formatted_sections = []
        
        for section in sections:
            # Check if section is a header
            if section.strip().isupper() or section.strip().endswith(':'):
                formatted_sections.append(f"\n## {section.strip()}\n")
            else:
                formatted_sections.append(section.strip())
        
        return '\n'.join(formatted_sections)
    
    def save_papers(self, papers: List[Any], topic: str):
        """Save papers to topic-specific directory"""
        # Create topic directory
        topic_dir = os.path.join(self.base_dir, topic)
        os.makedirs(topic_dir, exist_ok=True)
        
        # Save each paper
        for paper in papers:
            paper_id = paper.metadata.get('id', 'unknown')
            paper_title = paper.metadata.get('title', 'Untitled')
            
            # Format content
            formatted_content = self._format_paper_content(paper.page_content)
            
            # Create paper data
            paper_data = {
                'id': paper_id,
                'title': paper_title,
                'content': formatted_content,
                'metadata': paper.metadata
            }
            
            # Save to file
            paper_file = os.path.join(topic_dir, f"{paper_id}.json")
            with open(paper_file, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, indent=2, ensure_ascii=False)
    
    def load_papers(self, topic: str) -> List[Dict]:
        """Load papers from topic directory"""
        topic_dir = os.path.join(self.base_dir, topic)
        papers = []
        
        if os.path.exists(topic_dir):
            for filename in os.listdir(topic_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(topic_dir, filename), 'r', encoding='utf-8') as f:
                        papers.append(json.load(f))
        
        return papers
    
    def list_saved_topics(self) -> Dict[str, int]:
        """List all saved topics with their paper counts"""
        topics = {}
        
        if os.path.exists(self.base_dir):
            for topic in os.listdir(self.base_dir):
                topic_dir = os.path.join(self.base_dir, topic)
                if os.path.isdir(topic_dir):
                    # Count JSON files in topic directory
                    paper_count = len([f for f in os.listdir(topic_dir) if f.endswith('.json')])
                    topics[topic] = paper_count
        
        return topics
    
    def get_paper_details(self, topic: str, paper_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific paper"""
        topic_dir = os.path.join(self.base_dir, topic)
        filepath = os.path.join(topic_dir, f"{paper_id}.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None 