"""
LLM provider using Groq.
"""

from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

class LLMProvider:
    """Groq LLM provider."""
    
    def __init__(self, model: str = "llama3-8b-8192"):
        """
        Initialize Groq LLM provider.
        
        Args:
            model: The Groq model to use
        """
        self.llm = ChatGroq(model=model)
    
    def invoke(self, messages: List[HumanMessage], **kwargs):
        """
        Invoke Groq LLM.
        
        Args:
            messages: List of messages to send to LLM
            **kwargs: Additional arguments for LLM invocation
            
        Returns:
            LLM response
        """
        return self.llm.invoke(messages, **kwargs) 