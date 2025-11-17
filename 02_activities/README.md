# Assignment 2: Pokémon TCG Assistant

This assignment demonstrates a complete LangGraph-based assistant for Pokémon TCG with three integrated tools.

## Files Included

- **`assignment_2.ipynb`**: Main demonstration notebook showing all features
- **`probability_tool.py`**: Core probability calculation module using hypergeometric distribution

## Features Demonstrated

1. **Card Information Tool**: Real-time Pokémon card lookup via API
2. **Rules & Deck Building Tool**: Semantic search over official documentation using ChromaDB
3. **Draw Probability Calculator**: Exact mathematical calculations for card draw probabilities

## Quick Start

1. Ensure dependencies are installed (see requirements below)
2. Set up API keys in `../05_src/.secrets`
3. Start ChromaDB if using rules/deck building features
4. Open and run `assignment_2.ipynb`

## Requirements

- Python 3.11+
- langchain, langgraph, openai, chromadb, gradio, python-dotenv, requests
- ChromaDB (for rules/deck building tool)
- OpenAI API key
- RapidAPI key (for Pokémon TCG API)

## Probability Tool

The `probability_tool.py` implements:
- `p_at_least_n()`: Probability of drawing at least n copies of a card
- `p_a_or_b()`: Probability of drawing at least one from set A or B
- `p_a_and_b()`: Probability of drawing at least one A AND one B

All calculations use exact hypergeometric distribution formulas.

