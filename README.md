# AI Research Copilot

An autonomous research agent that helps with academic research by searching, reading, and summarizing scientific papers from arXiv.

## Features

- Search arXiv for relevant papers based on research goals
- Extract and analyze paper abstracts and content
- Build a knowledge base of research findings using sentence-transformers embeddings
- Generate concise research summaries
- Multi-agent workflow using LangGraph
- Powered by Groq's Llama 3 8B model for fast, free inference

## Setup

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
pip install sentence-transformers "numpy<2.0.0"
```
3. Create a `.env` file with your API key:
```
GROQ_API_KEY=your_key_here
```

## Usage

```python
from research_copilot import ResearchCopilot

# Initialize with default Llama model
copilot = ResearchCopilot()

# Or specify a different Groq model
copilot = ResearchCopilot(model="llama3-8b-8192")

# Conduct research
results = copilot.research(
    research_goal="Your research question or goal here",
    max_papers=5,
    include_full_text=False  # Set to True to analyze full papers
)

# Access findings and summary
if 'findings' in results and 'findings' in results['findings']:
    findings = results['findings']['findings']
    print(f"Papers Found: {len(findings)}")
    
if 'summary' in results and 'summary' in results['summary']:
    print("\nOverall Summary:")
    print(results['summary']['summary'])
```

## Components

### Groq Integration

The Research Copilot uses Groq's API for all language model operations:

- **Model**: Llama 3 8B
- **Features**:
  - Fast inference speed
  - Free API access
  - High-quality research analysis
  - 8,192 token context window

### Knowledge Base

The system uses a vector-based knowledge base powered by:
- **ChromaDB**: For efficient vector storage and retrieval
- **Sentence Transformers**: Local embedding model (all-MiniLM-L6-v2)
  - No additional API costs
  - Fast local embedding generation
  - Suitable for research paper analysis

## Project Structure

- `research_copilot/` - Main package directory
  - `agents/` - Individual agent implementations
  - `tools/` - Utility functions and API wrappers
  - `knowledge_base/` - Knowledge base management with ChromaDB
  - `workflows/` - LangGraph workflow definitions

## License

MIT 