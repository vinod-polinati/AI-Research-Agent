"""
Example usage of the AI Research Copilot.
"""

import os
from dotenv import load_dotenv
from research_copilot import ResearchCopilot

# Load environment variables
load_dotenv()

def main():
    # Initialize Research Copilot with Groq
    copilot = ResearchCopilot(
        model="llama3-8b-8192"  # Using Llama 3 8B model from Groq
    )
    
    # Define research goal
    research_goal = """
    What are the latest advances in large language model reasoning capabilities,
    particularly in areas like chain-of-thought prompting and self-reflection?
    Focus on papers from 2023-2024.
    """
    
    # Conduct research
    results = copilot.research(
        research_goal=research_goal,
        max_papers=5,
        include_full_text=False  # Set to True to analyze full papers
    )
    
    # Print summary
    print("\nResearch Summary:")
    print("=" * 80)
    print(f"\nResearch Goal: {research_goal}")
    
    if 'findings' in results and 'findings' in results['findings']:
        findings = results['findings']['findings']
        print(f"\nPapers Found: {len(findings)}")
        print("\nKey Findings:")
        for i, finding in enumerate(findings, 1):
            print(f"\n{i}. {finding['title']}")
            print(f"   Authors: {finding.get('authors', 'N/A')}")
            print(f"   Key Points: {finding.get('key_points', 'N/A')}")
    
    if 'summary' in results and 'summary' in results['summary']:
        print("\nOverall Summary:")
        print(results['summary']['summary'])

if __name__ == "__main__":
    main() 