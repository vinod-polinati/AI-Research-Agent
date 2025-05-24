"""
LangGraph workflow for orchestrating the research process.
"""

from typing import Dict, Any, TypedDict
from langgraph.graph import Graph, StateGraph
from research_copilot.agents.researcher import ResearcherAgent
from research_copilot.agents.synthesizer import SynthesizerAgent
from research_copilot.knowledge_base.base import KnowledgeBase

class ResearchState(TypedDict):
    """Type definition for research workflow state."""
    research_goal: str
    max_papers: int
    include_full_text: bool
    knowledge_base: KnowledgeBase
    findings: Dict[str, Any]
    summary: Dict[str, Any]

def create_research_workflow(model: str = "llama3-8b-8192") -> Graph:
    """
    Create the research workflow graph.
    
    Args:
        model: The Groq model to use
        
    Returns:
        LangGraph workflow for research
    """
    # Initialize workflow graph
    workflow = StateGraph(ResearchState)
    
    # Initialize agents
    researcher = ResearcherAgent(model=model)
    synthesizer = SynthesizerAgent(model=model)
    
    # Define research node
    def research(state: ResearchState) -> Dict[str, Any]:
        findings = researcher.research(
            research_goal=state["research_goal"],
            knowledge_base=state["knowledge_base"],
            max_papers=state["max_papers"],
            include_full_text=state["include_full_text"]
        )
        return {"findings": findings}
    
    # Define synthesis node
    def synthesize(state: ResearchState) -> Dict[str, Any]:
        summary = synthesizer.generate_summary(
            research_goal=state["research_goal"],
            findings=state["findings"]["findings"],
            knowledge_base=state["knowledge_base"]
        )
        return {"summary": summary}
    
    # Add nodes to graph
    workflow.add_node("research", research)
    workflow.add_node("synthesize", synthesize)
    
    # Define edges
    workflow.add_edge("research", "synthesize")
    workflow.set_entry_point("research")
    
    # Compile workflow
    return workflow.compile() 