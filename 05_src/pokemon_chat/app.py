import gradio as gr
import sys
import os

# Add 05_src directory to path so we can import pokemon_chat
# When running from pokemon_chat directory, we need to add the parent (05_src) to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_file_dir)  # This is 05_src
sys.path.insert(0, src_dir)

from pokemon_chat.main import pokemon_chat, get_chroma_collection
from dotenv import load_dotenv
from utils.logger import get_logger

_logs = get_logger(__name__)

# Load secrets from 05_src directory (src_dir was already calculated above)
load_dotenv(os.path.join(src_dir, '.secrets'))
load_dotenv(os.path.join(src_dir, '.env'))

chat = gr.ChatInterface(
    fn=pokemon_chat,
    type="messages",
    title="Pokémon TCG Assistant"
)

if __name__ == "__main__":
    # Ensure ChromaDB is running and load embeddings if needed
    _logs.info('Initializing ChromaDB collection...')
    get_chroma_collection()
    _logs.info('Ready!')
    chat.launch()

