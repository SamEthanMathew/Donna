"""
Core Assistant class that orchestrates LLM, memory, and conversation management.
"""

import sys
from datetime import datetime
from typing import Optional

# Import from unified config
import config

# Add project root to path and import llm as a package
_BASE_DIR = config.BASE_DIR
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

# Import from llm.core package
from llm.core.ollama_client import OllamaClient
from llm.core.memory_store import MemoryStore
from llm.core.memory_policy import extract_memory_suggestion, is_allowed_memory, categorize
from llm.core.conversation_summary import ConversationManager
from llm.core.utils import load_text_file


def format_memory_context(memories) -> str:
    """Format memories as bullet points for context."""
    if not memories:
        return ""
    lines = ["Relevant context:"]
    for mem in memories:
        lines.append(f"- {mem.key} = {mem.value}")
    return "\n".join(lines)


class Assistant:
    """
    Main assistant orchestrator that handles LLM interactions,
    memory management, and conversation state.
    """
    
    def __init__(self):
        """Initialize the assistant with LLM, memory, and conversation manager."""
        self.memory = MemoryStore(config.DB_PATH)
        self.llm = OllamaClient(config.OLLAMA_URL, config.MODEL_NAME)
        self.system_prompt = load_text_file(config.PROMPT_PATH)
        self.conversation = ConversationManager(
            self.llm,
            verbatim_turns=config.VERBATIM_TURNS,
            summary_threshold=config.SUMMARY_UPDATE_THRESHOLD
        )
        self.last_interaction = None
        self.vision_context = ""  # Current vision state (detected persons)
        
    def set_vision_context(self, context: str):
        """Update the current vision context (detected persons)."""
        self.vision_context = context
    
    def get_vision_context(self) -> str:
        """Get the current vision context."""
        return self.vision_context
    
    def process(self, user_input: str, stream_callback: Optional[callable] = None) -> str:
        """
        Process user input and return assistant response.
        
        Args:
            user_input: User's text input
            stream_callback: Optional callback function(token) for streaming tokens
            
        Returns:
            Full assistant response text
        """
        if not user_input or not user_input.strip():
            return ""
        
        # Truncate if too long
        if len(user_input) > config.MAX_USER_CHARS:
            user_input = user_input[:config.MAX_USER_CHARS] + "…"
        
        # Get relevant memories (3-8 max)
        relevant_memories = self.memory.get_relevant_memories(
            user_input,
            min_count=config.MIN_MEMORY_CONTEXT,
            max_count=config.MAX_MEMORY_CONTEXT
        )
        memory_context = format_memory_context(relevant_memories)
        
        # Get conversation context (summary + recent turns)
        conv_context = self.conversation.get_context()
        
        # Build messages: system prompt + memory context + conversation context + vision context + user message
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
        
        if conv_context:
            messages.append({"role": "system", "content": conv_context})
        
        # Add vision context if available
        if self.vision_context:
            messages.append({"role": "system", "content": f"Vision context: {self.vision_context}"})
        
        messages.append({"role": "user", "content": user_input})
        
        # Call model with streaming
        try:
            full_response = ""
            stream = self.llm.chat_stream(messages)
            
            # Collect tokens and optionally stream to callback
            for token in stream:
                if token.startswith("__FULL_RESPONSE__"):
                    # This is the final full response yield
                    full_response = token.replace("__FULL_RESPONSE__", "", 1)
                else:
                    # Individual token
                    full_response += token
                    if stream_callback:
                        stream_callback(token)
            
            # Extract memory suggestion (if any)
            assistant_text, suggestion = extract_memory_suggestion(full_response)
            
            if len(assistant_text) > config.MAX_ASSISTANT_CHARS:
                assistant_text = assistant_text[:config.MAX_ASSISTANT_CHARS] + "…"
            
            # Add to conversation
            self.conversation.add_turn(user_input, assistant_text)
            
            # Update summary if needed
            if self.conversation.should_update_summary():
                self.conversation.update_summary()
            
            # Decide whether to store memory
            if suggestion:
                if is_allowed_memory(suggestion):
                    cat = categorize(suggestion.key)
                    self.memory.upsert_memory(suggestion.key, suggestion.value, cat, confidence=0.78)
                    if config.LOG_INTERACTIONS:
                        print(f"[Saved memory] {suggestion.key} = {suggestion.value}")
                else:
                    if config.LOG_INTERACTIONS:
                        print("[Memory suggestion rejected by policy]")
            
            # Update last interaction timestamp
            self.last_interaction = datetime.now()
            
            return assistant_text
            
        except Exception as e:
            error_msg = f"Error calling Ollama: {e}"
            if config.LOG_INTERACTIONS:
                print(error_msg)
            raise RuntimeError(error_msg)
    
    def process_proactive(self, proactive_brief: str) -> str:
        """
        Process a proactive conversation trigger.
        
        Args:
            proactive_brief: Brief context for proactive conversation
            
        Returns:
            Assistant response text
        """
        # Use the proactive brief as user input
        return self.process(proactive_brief)
    
    def list_memories(self, limit: int = 50):
        """List stored memories."""
        return self.memory.list_memories(limit=limit)

