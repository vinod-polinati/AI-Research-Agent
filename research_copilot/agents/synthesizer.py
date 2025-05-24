"""
Synthesizer agent for generating research summaries.
"""

from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from research_copilot.tools.llm_provider import LLMProvider
from research_copilot.knowledge_base.base import KnowledgeBase

SYNTHESIS_PROMPT = """
Generate a comprehensive research summary based on the following findings from multiple papers.
Focus on synthesizing insights, identifying patterns, and highlighting practical applications.

Research Goal: {research_goal}

Papers Analyzed:
{papers_summary}

Findings:
{findings}

Provide a well-structured summary that includes:
1. Overview of the research area
2. Key findings and insights
3. Common themes and patterns
4. Practical applications and implications
5. Gaps and future research directions
"""

class SynthesizerAgent:
    """Agent responsible for synthesizing research findings."""
    
    def __init__(self, model: str = "llama3-8b-8192"):
        """Initialize the synthesizer agent."""
        self.llm = LLMProvider(model=model)
        
    def generate_summary(self,
                        research_goal: str,
                        findings: List[Dict],
                        knowledge_base: KnowledgeBase) -> Dict[str, Any]:
        """
        Generate a research summary from findings.
        
        Args:
            research_goal: Original research question/goal
            findings: List of paper analyses
            knowledge_base: Knowledge base containing findings
            
        Returns:
            Dictionary containing the research summary
        """
        # Format papers summary
        papers_summary = "\n".join([
            f"- {finding['title']}" for finding in findings
        ])
        
        # Format findings
        findings_text = "\n\n".join([
            f"Paper: {finding['title']}\n{finding['analysis']}"
            for finding in findings
        ])
        
        # Generate synthesis
        synthesis = self.llm.invoke([
            HumanMessage(content=SYNTHESIS_PROMPT.format(
                research_goal=research_goal,
                papers_summary=papers_summary,
                findings=findings_text
            ))
        ]).content
        
        return {
            "research_goal": research_goal,
            "papers_analyzed": len(findings),
            "summary": synthesis
        } 