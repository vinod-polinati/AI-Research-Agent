"""
ArXiv API integration for paper search and retrieval.
"""

from typing import List, Dict, Optional
import arxiv
from bs4 import BeautifulSoup
import pypdf
import requests
from dataclasses import dataclass

@dataclass
class Paper:
    """Represents a scientific paper from arXiv."""
    id: str
    title: str
    authors: List[str]
    abstract: str
    pdf_url: str
    published: str
    categories: List[str]

class ArXivTool:
    """Tool for interacting with arXiv API."""
    
    def __init__(self):
        """Initialize the ArXiv client."""
        self.client = arxiv.Client()
        
    def search_papers(self, 
                     query: str,
                     max_results: int = 5) -> List[Paper]:
        """
        Search for papers on arXiv.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of Paper objects
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for result in self.client.results(search):
            paper = Paper(
                id=result.entry_id.split('/')[-1],
                title=result.title,
                authors=[str(author) for author in result.authors],
                abstract=result.summary,
                pdf_url=result.pdf_url,
                published=result.published.strftime("%Y-%m-%d"),
                categories=result.categories
            )
            results.append(paper)
            
        return results
    
    def get_full_text(self, paper: Paper) -> str:
        """
        Download and extract full text from a paper's PDF.
        
        Args:
            paper: Paper object containing PDF URL
            
        Returns:
            Extracted text content
        """
        # Download PDF
        response = requests.get(paper.pdf_url)
        
        # Save temporarily and read with PyPDF
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
            
        text = ""
        with open("temp.pdf", "rb") as f:
            pdf = pypdf.PdfReader(f)
            for page in pdf.pages:
                text += page.extract_text()
                
        return text 