"""
Knowledge base management for storing and retrieving research findings.
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import tempfile
import os
import shutil

class KnowledgeBase:
    """Manages the storage and retrieval of research findings."""
    
    def __init__(self):
        """Initialize the knowledge base with ChromaDB."""
        # Create a temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize embedding function using SentenceTransformers
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        # Initialize ChromaDB with new configuration
        self.client = chromadb.PersistentClient(path=self.temp_dir)
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(
                name="research_findings",
                embedding_function=self.embedding_function
            )
        except NotFoundError:
            self.collection = self.client.create_collection(
                name="research_findings",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        
    def add_paper(self, 
                 paper_id: str,
                 title: str,
                 abstract: str,
                 findings: List[str],
                 metadata: Dict = None) -> None:
        """
        Add a paper and its findings to the knowledge base.
        
        Args:
            paper_id: Unique identifier for the paper
            title: Paper title
            abstract: Paper abstract
            findings: List of key findings
            metadata: Additional metadata about the paper
        """
        # Convert metadata values to supported types
        processed_metadata = {}
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, list):
                    processed_metadata[key] = ", ".join(str(v) for v in value)
                else:
                    processed_metadata[key] = value
        
        self.collection.add(
            documents=[abstract] + findings,
            metadatas=[{"type": "abstract", **processed_metadata}] + 
                     [{"type": "finding", **processed_metadata} for _ in findings],
            ids=[f"{paper_id}_abstract"] + 
                [f"{paper_id}_finding_{i}" for i in range(len(findings))]
        )
    
    def search(self, 
              query: str, 
              n_results: int = 5) -> List[Dict]:
        """
        Search the knowledge base for relevant findings.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant findings and their metadata
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return results
    
    def get_all_findings(self) -> List[Dict]:
        """Get all findings from the knowledge base."""
        return self.collection.get(
            where={"type": "finding"}
        )
    
    def __del__(self):
        """Cleanup temporary directory when the object is destroyed."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass 