import os
import requests
import json

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage, ToolMessage
from typing import Literal, Optional
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from tqdm import tqdm

from pokemon_chat.prompts import return_instructions_root
from utils.logger import get_logger

# Import probability functions from local module
from pokemon_chat.probability_tool import p_at_least_n, p_a_or_b, p_a_and_b

_logs = get_logger(__name__)

# Load environment variables from 05_src directory (parent of pokemon_chat)
# When running as a module, __file__ is the path to this file
module_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(module_dir)  # Go up one level to 05_src
load_dotenv(os.path.join(src_dir, ".env"))
load_dotenv(os.path.join(src_dir, ".secrets"))

client = OpenAI()

BASE_URL = "https://pokemon-tcg-pocket1.p.rapidapi.com"
RAPIDAPI_HOST = "pokemon-tcg-pocket1.p.rapidapi.com"

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
if not RAPIDAPI_KEY:
    raise RuntimeError("Please set RAPIDAPI_KEY environment variable.")

HEADERS = {
    "x-rapidapi-host": RAPIDAPI_HOST,
    "x-rapidapi-key": RAPIDAPI_KEY,
}

def search_card_by_name(name: str):
    """Search for a Pokémon card by name using the RapidAPI."""
    url = f"{BASE_URL}/api/chosen_cards_name"
    params = {"name": name}
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    data = resp.json()
    
    if isinstance(data, list) and data:
        return data[0]
    return None

@tool
def pokemon_card_info_tool(query: str) -> str:
    """
    Given a user query mentioning a Pokémon card by name,
    fetch the first matching card from the live Pokémon TCG API
    and return key info (HP, type, stage, attacks, weaknesses, price).
    """
    card = search_card_by_name(query)
    
    if card is None:
        return f"I couldn't find a card matching '{query}'."

    name = card.get("0_Card_name")
    hp = card.get("5_HP")
    pokemon_type = card.get("4_pokemon_type")
    card_type = card.get("3_card_type")
    pack = card.get("1_pack")
    pack_number = card.get("2_pack_number")
    weakness = card.get("8_weakness")
    retreat_cost = card.get("9_retreat_cost")
    ability = card.get("10_ability")
    ability_effect = card.get("11_ability_effect")
    description = card.get("12_description")
    
    attacks = []
    attack_1 = card.get("6_Attack_1")
    attack_2 = card.get("7_Attack_1")
    if attack_1 and attack_1.get("name"):
        attacks.append(attack_1)
    if attack_2 and attack_2.get("name"):
        attacks.append(attack_2)

    summary = {
        "name": name,
        "hp": hp,
        "pokemon_type": pokemon_type,
        "card_type": card_type,
        "pack": pack,
        "pack_number": pack_number,
        "attacks": attacks,
        "weakness": weakness,
        "retreat_cost": retreat_cost,
        "ability": ability,
        "ability_effect": ability_effect,
        "description": description,
    }

    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    
    prompt = f"""Summarize the following Pokémon card information in a clear and concise way:

{json.dumps(summary, indent=2)}

Provide a natural language summary that includes:
- The card name and basic stats (HP, type, card type)
- Attack information if available
- Weakness and retreat cost
- Any special abilities
- The card's description

Keep it concise and easy to read."""

    messages = [
        SystemMessage(content="You are a helpful assistant that summarizes Pokémon card information."),
        HumanMessage(content=prompt)
    ]
    
    response = model.invoke(messages)
    return response.content

# Batch IDs from completed embeddings
BATCH_IDS = [
    "batch_691a6dd2db608190bc19f29d5ca6a691",
    "batch_691a6dd3907c81909cb177760006a745",
    "batch_691a6dd4132c81909b188ea52035eb83"
]

CHROMA_URL = "http://localhost:8000"
COLLECTION_NAME = "pokemon_documents"

def get_content_from_file(file_id: str):
    """Load JSONL content from OpenAI file"""
    file = client.files.content(file_id)
    text = file.text
    lines = text.split('\n')
    content_lines = [json.loads(line) for line in lines if line.strip()]
    return content_lines

def get_text_and_embeddings(batch_id: str):
    """Get text and embedding data from a completed batch"""
    batch = client.batches.retrieve(batch_id)
    embedding_lines = get_content_from_file(batch.output_file_id)
    text_lines = get_content_from_file(batch.input_file_id)
    return embedding_lines, text_lines

def create_chroma_inputs(embedding_lines, text_lines):
    """Create ChromaDB input format from batch results"""
    chroma_inputs = []
    text_dict = {item['custom_id']: item['body']['input'] for item in text_lines}
    for embed_item in embedding_lines:
        custom_id = embed_item['custom_id']
        text = text_dict.get(custom_id, "")
        chroma_input = {
            'id': custom_id,
            'embedding': embed_item['response']['body']['data'][0]['embedding'],
            'text': text
        }
        chroma_inputs.append(chroma_input)
    return chroma_inputs

def setup_collection(chroma_url: str = CHROMA_URL, collection_name: str = COLLECTION_NAME):
    """Setup ChromaDB collection, deleting if exists to ensure it's empty"""
    chroma_client = chromadb.HttpClient(host=chroma_url)
    collections = chroma_client.list_collections()
    
    # Delete collection if it exists to ensure it's empty
    if collection_name in [col.name for col in collections]:
        chroma_client.delete_collection(name=collection_name)
    
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
    )
    return collection

def load_embeddings_to_chromadb(batch_ids: list[str], chroma_url: str = CHROMA_URL, 
                                 collection_name: str = COLLECTION_NAME, batch_size: int = 100):
    """Load embeddings from completed batches into ChromaDB"""
    all_chroma_inputs = []
    
    # Process all batches
    for batch_id in tqdm(batch_ids, desc="Processing batches"):
        embedding_lines, text_lines = get_text_and_embeddings(batch_id)
        chroma_inputs = create_chroma_inputs(embedding_lines, text_lines)
        all_chroma_inputs.extend(chroma_inputs)
    
    # Setup collection (ensures it's empty)
    collection = setup_collection(chroma_url=chroma_url, collection_name=collection_name)
    
    # Load embeddings in batches
    for i in tqdm(range(0, len(all_chroma_inputs), batch_size), desc="Loading to ChromaDB"):
        batch = all_chroma_inputs[i:i + batch_size]
        collection.add(
            documents=[item['text'] for item in batch],
            embeddings=[item['embedding'] for item in batch],
            ids=[item['id'] for item in batch]
        )
    
    return collection

# Initialize ChromaDB collection (load embeddings on first import)
_chroma_collection = None

def get_chroma_collection():
    """Get or initialize ChromaDB collection"""
    global _chroma_collection
    if _chroma_collection is None:
        try:
            chroma_client = chromadb.HttpClient(host=CHROMA_URL)
            _chroma_collection = chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-3-small"
                )
            )
        except Exception:
            # Collection doesn't exist, load embeddings
            _chroma_collection = load_embeddings_to_chromadb(BATCH_IDS)
    return _chroma_collection

@tool
def pokemon_rules_and_deck_building_tool(query: str) -> str:
    """
    Answer questions about Pokémon TCG rules and deck building strategies.
    Uses embeddings to find relevant information from the official rulebook and deck building guide.
    """
    collection = get_chroma_collection()
    
    # Query ChromaDB for relevant context
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    if not results['documents'] or not results['documents'][0]:
        return "I couldn't find relevant information to answer your question."
    
    # Build context from retrieved documents
    context = "\n\n".join([f"Context {i+1}:\n{doc}" for i, doc in enumerate(results['documents'][0])])
    
    # Use LLM to generate answer based on context
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    
    prompt = f"""Answer the following question about Pokémon TCG rules or deck building using only the provided context.

Question: {query}

Context from documents:
{context}

Provide a clear, accurate answer based on the context. If the context doesn't contain enough information, say so."""

    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions about Pokémon TCG rules and deck building strategies based on official documentation."),
        HumanMessage(content=prompt)
    ]
    
    response = model.invoke(messages)
    return response.content

@tool
def pokemon_draw_probability_tool(
    calculation_type: str,
    deck_size: int = 60,
    draw_size: int = 7,
    count_a: Optional[int] = None,
    count_b: Optional[int] = None,
    min_copies: int = 1
) -> str:
    """
    Calculate the probability of drawing specific cards from a Pokémon TCG deck.
    
    This tool calculates exact probabilities using hypergeometric distribution (without replacement).
    Standard Pokémon TCG decks have 60 cards, and opening hands draw 7 cards.
    
    Args:
        calculation_type: Type of probability to calculate. Must be one of:
            - "at_least_n": Probability of drawing at least min_copies of a single card type
            - "or": Probability of drawing at least one from either card A or card B (or both)
            - "and": Probability of drawing at least one A AND at least one B
        deck_size: Total deck size (default: 60 for standard Pokémon TCG)
        draw_size: Number of cards to draw (default: 7 for opening hand)
        count_a: Number of copies of card A in deck (required for all calculation types)
        count_b: Number of copies of card B in deck (required for "or" and "and" calculations)
        min_copies: Minimum number of copies to draw (for "at_least_n", default: 1)
    
    Returns:
        A formatted string with the probability as both a decimal and percentage.
    """
    try:
        if calculation_type == "at_least_n":
            if count_a is None:
                return "Error: count_a is required for 'at_least_n' calculation."
            prob = p_at_least_n(deck_size, count_a, draw_size, min_copies)
            return f"Probability of drawing at least {min_copies} copy/copies of a card with {count_a} copies in deck: {prob:.4f} ({prob*100:.2f}%)"
        
        elif calculation_type == "or":
            if count_a is None or count_b is None:
                return "Error: Both count_a and count_b are required for 'or' calculation."
            prob = p_a_or_b(deck_size, count_a, count_b, draw_size)
            return f"Probability of drawing at least one card A ({count_a} copies) OR card B ({count_b} copies): {prob:.4f} ({prob*100:.2f}%)"
        
        elif calculation_type == "and":
            if count_a is None or count_b is None:
                return "Error: Both count_a and count_b are required for 'and' calculation."
            prob = p_a_and_b(deck_size, count_a, count_b, draw_size)
            return f"Probability of drawing at least one card A ({count_a} copies) AND at least one card B ({count_b} copies): {prob:.4f} ({prob*100:.2f}%)"
        
        else:
            return f"Error: Invalid calculation_type '{calculation_type}'. Must be one of: 'at_least_n', 'or', 'and'"
    
    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error calculating probability: {str(e)}"

# LangGraph State Definition
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def get_model_with_tools():
    """Get model with all tools"""
    model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.7)
    tools = [pokemon_card_info_tool, pokemon_rules_and_deck_building_tool, pokemon_draw_probability_tool]
    model_with_tools = model.bind_tools(tools)
    return model_with_tools

# LangGraph Nodes
def llm_call(state: MessagesState):
    """LLM node: decides whether to call a tool or provide final answer"""
    model_with_tools = get_model_with_tools()
    system_message = SystemMessage(content=return_instructions_root())
    
    return {
        "messages": [
            model_with_tools.invoke([system_message] + state["messages"])
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

def tool_node(state: MessagesState):
    """Tool node: executes tool calls and returns results"""
    tools = [pokemon_card_info_tool, pokemon_rules_and_deck_building_tool, pokemon_draw_probability_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    
    result = []
    last_message = state["messages"][-1]
    
    # Handle multiple tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool = tools_by_name.get(tool_call["name"])
            if tool:
                observation = tool.invoke(tool_call["args"])
                result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """
    Decision point: determines whether to continue using tools or stop.
    If the LLM made tool calls, continue to tool_node.
    Otherwise, stop and return final answer to user.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, continue to tool node
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    
    # Otherwise, stop and return the final answer
    return END

def get_pokemon_agent():
    """Build and compile the LangGraph agent"""
    # Build workflow
    agent_builder = StateGraph(MessagesState)
    
    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    
    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    
    # Conditional edge: decide whether to use tools or end
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            "tool_node": "tool_node",
            END: END
        }
    )
    
    # After tool execution, go back to LLM to process results
    agent_builder.add_edge("tool_node", "llm_call")
    
    return agent_builder.compile()

# Initialize agent
_agent = None

def get_agent():
    """Get or initialize the LangGraph agent"""
    global _agent
    if _agent is None:
        _agent = get_pokemon_agent()
    return _agent

def reset_agent():
    """Reset the cached agent (useful when prompts change)"""
    global _agent
    _agent = None

def pokemon_chat(message: str, history: list[dict] = []) -> str:
    """Gradio chat function using LangGraph agent"""
    _logs.info(f'User message: {message}')
    
    agent = get_agent()
    
    # Convert history to LangChain messages
    langchain_messages = []
    for msg in history:
        if msg['role'] == 'user':
            langchain_messages.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            langchain_messages.append(AIMessage(content=msg['content']))
    
    # Add current user message
    langchain_messages.append(HumanMessage(content=message))
    
    # Invoke agent with state
    state = {
        "messages": langchain_messages,
        "llm_calls": 0
    }
    
    result = agent.invoke(state)
    
    # Return the last message content (final answer)
    if result["messages"]:
        last_message = result["messages"][-1]
        if hasattr(last_message, 'content'):
            return last_message.content
        return str(last_message)
    
    return "I couldn't generate a response."

