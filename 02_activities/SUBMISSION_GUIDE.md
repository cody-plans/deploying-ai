# Assignment 2 Submission Guide

## Required Files for Submission

### Core Files (Required)
1. **`assignment_2.ipynb`** - Main demonstration notebook
2. **`probability_tool.py`** - Probability calculation module (used by pokemon_chat)

### Supporting Files (Required for notebook to run)
3. **`../05_src/pokemon_chat/main.py`** - Main pokemon_chat module
4. **`../05_src/pokemon_chat/prompts.py`** - System prompts
5. **`../05_src/pokemon_chat/__init__.py`** - Package init file
6. **`../05_src/utils/logger.py`** - Logger utility

### Configuration Files (Required)
7. **`../05_src/.secrets`** - API keys (OPENAI_API_KEY, RAPIDAPI_KEY)
8. **`../05_src/.env`** - Environment variables (optional)

### External Dependencies
- ChromaDB running on localhost:8000 (for rules/deck building tool)
- Python packages: langchain, langgraph, openai, chromadb, gradio, dotenv

## Submission Steps

### Option 1: Minimal Submission (Recommended)
Submit only the essential files that demonstrate the work:

```
assignment_2/
├── assignment_2.ipynb          # Main notebook
├── probability_tool.py         # Probability calculations
└── README.md                    # Setup instructions
```

### Option 2: Complete Submission
Include all supporting code:

```
assignment_2/
├── assignment_2.ipynb
├── probability_tool.py
├── pokemon_chat/
│   ├── __init__.py
│   ├── main.py
│   └── prompts.py
├── utils/
│   └── logger.py
└── README.md
```

## What the Notebook Demonstrates

1. **Card Information Tool**: Fetches Pokémon card data from API
2. **Rules & Deck Building Tool**: Answers questions using ChromaDB embeddings
3. **Probability Calculator**: Direct calculation examples (4 scenarios)
4. **Full Chat Interface**: Complete conversation with probability calculation

## Setup Instructions (for reviewer)

1. Install dependencies:
   ```bash
   pip install langchain langgraph openai chromadb gradio python-dotenv requests
   ```

2. Set up environment variables in `../05_src/.secrets`:
   ```
   OPENAI_API_KEY=your_key_here
   RAPIDAPI_KEY=your_key_here
   ```

3. Start ChromaDB (if using rules/deck building tool):
   ```bash
   docker compose -f ../05_src/deploying_ai_data/docker-compose.yml up -d chromadb
   ```

4. Run the notebook: Open `assignment_2.ipynb` in Jupyter

## Key Features Demonstrated

- ✅ LangGraph agent with tool orchestration
- ✅ Three integrated tools (card info, rules, probability)
- ✅ Direct probability calculation (no confirmation needed)
- ✅ Hypergeometric distribution for exact probabilities
- ✅ Vector database integration for semantic search

