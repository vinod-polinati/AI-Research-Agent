"""
Researcher agent for paper search and analysis.
"""

from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from research_copilot.tools.arxiv import ArXivTool, Paper
from research_copilot.tools.llm_provider import LLMProvider
from research_copilot.knowledge_base.base import KnowledgeBase

PAPER_ANALYSIS_PROMPT = """
Analyze the following research paper and extract key findings and insights.

Title: {title}
Authors: {authors}
Abstract: {abstract}

{full_text}

Extract and summarize the main findings, methodology, and contributions.
Focus on novel insights and practical applications.
"""

class ResearcherAgent:
    """Agent responsible for searching and analyzing papers."""
    
    def __init__(self, model: str = "llama3-8b-8192"):
        """Initialize the researcher agent."""
        self.arxiv_tool = ArXivTool()
        self.llm = LLMProvider(model=model)
        
    def search_papers(self, research_goal: str, max_papers: int) -> List[Paper]:
        """Search for relevant papers based on research goal."""
        # Formulate search query
        search_query = self.llm.invoke([
            HumanMessage(content=f"""
            Convert this research goal into an effective arXiv search query:
            {research_goal}
            
            Return only the search query, no explanation.
            """)
        ]).content
        
        return self.arxiv_tool.search_papers(search_query, max_papers)
    
    def analyze_paper(self, 
                     paper: Paper,
                     include_full_text: bool = False) -> Dict[str, Any]:
        """
        Analyze a paper and extract key findings.
        
        Args:
            paper: Paper to analyze
            include_full_text: Whether to analyze full text or just abstract
            
        Returns:
            Dictionary containing analysis results
        """
        # Get paper content
        full_text = ""
        if include_full_text:
            full_text = self.arxiv_tool.get_full_text(paper)
            
        # Analyze paper
        analysis = self.llm.invoke([
            HumanMessage(content=PAPER_ANALYSIS_PROMPT.format(
                title=paper.title,
                authors=", ".join(paper.authors),
                abstract=paper.abstract,
                full_text=full_text
            ))
        ]).content
        
        return {
            "paper_id": paper.id,
            "title": paper.title,
            "analysis": analysis,
            "metadata": {
                "authors": paper.authors,
                "published": paper.published,
                "categories": paper.categories
            }
        }
        
    def research(self,
                research_goal: str,
                knowledge_base: KnowledgeBase,
                max_papers: int = 5,
                include_full_text: bool = False) -> Dict[str, Any]:
        """
        Conduct research on a given topic.
        
        Args:
            research_goal: Research question or topic
            knowledge_base: Knowledge base to store findings
            max_papers: Maximum number of papers to analyze
            include_full_text: Whether to analyze full papers
            
        Returns:
            Research findings and summary
        """
        # Search for papers
        papers = self.search_papers(research_goal, max_papers)
        
        # Analyze each paper
        findings = []
        for paper in papers:
            analysis = self.analyze_paper(paper, include_full_text)
            findings.append(analysis)
            
            # Add to knowledge base
            knowledge_base.add_paper(
                paper_id=paper.id,
                title=paper.title,
                abstract=paper.abstract,
                findings=[analysis["analysis"]],
                metadata=analysis["metadata"]
            )
            
        return {
            "papers_analyzed": len(findings),
            "findings": findings
        } 