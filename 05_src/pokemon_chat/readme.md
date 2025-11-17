# Pokémon TCG Assistant

This chat app provides a comprehensive Pokémon TCG assistant with three main capabilities:

1. **Card Information**: Look up specific Pokémon cards by name using the live Pokémon TCG API
2. **Rules & Deck Building**: Answer questions about Pokémon TCG rules and deck building strategies using vector embeddings from official documentation
3. **Draw Probability Calculator**: Calculate exact probabilities of drawing specific cards from a deck using hypergeometric distribution

## Features

- **LangGraph Agent**: Uses LangGraph for advanced conversational flow with sequential tool calls
- **Vector Database**: ChromaDB integration for semantic search over rulebook and deck building guides
- **Probability Calculations**: Exact mathematical calculations for card draw probabilities
- **Interactive Gradio Interface**: User-friendly chat interface

## Tools

- `pokemon_card_info_tool`: Fetches card information from the Pokémon TCG API
- `pokemon_rules_and_deck_building_tool`: Answers questions using embedded documentation
- `pokemon_draw_probability_tool`: Calculates drawing probabilities with confirmation workflow

## Setup

1. Ensure ChromaDB is running (default: http://localhost:8000)
2. Set environment variables in `.secrets`:
   - `OPENAI_API_KEY`
   - `RAPIDAPI_KEY`
3. Run the app: `python app.py`

## Usage

The assistant can help with:
- "Tell me about Pikachu"
- "What are the rules for evolution?"
- "What's the probability of drawing at least one Pikachu in my opening hand?"

