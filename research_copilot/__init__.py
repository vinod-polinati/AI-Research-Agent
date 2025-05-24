"""
AI Research Copilot - An autonomous research agent for academic paper analysis.
"""

from typing import List, Dict
from research_copilot.workflows.research_workflow import create_research_workflow
from research_copilot.knowledge_base.base import KnowledgeBase

class ResearchCopilot:
    """Main class for the AI Research Copilot."""
    
    def __init__(self, model: str = "llama3-8b-8192"):
        """
        Initialize the research copilot.
        
        Args:
            model: The Groq model to use
        """
        self.knowledge_base = KnowledgeBase()
        self.workflow = create_research_workflow(model=model)
    
    def research(self, 
                research_goal: str, 
                max_papers: int = 5,
                include_full_text: bool = False) -> Dict:
        """
        Conduct research based on the provided goal.
        
        Args:
            research_goal: The research question or topic to investigate
            max_papers: Maximum number of papers to analyze
            include_full_text: Whether to analyze full papers or just abstracts
            
        Returns:
            Dictionary containing research summary and findings
        """
        # Execute the research workflow
        result = self.workflow.invoke({
            "research_goal": research_goal,
            "max_papers": max_papers,
            "include_full_text": include_full_text,
            "knowledge_base": self.knowledge_base
        })
        
        return result 