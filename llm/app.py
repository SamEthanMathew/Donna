import sys
from datetime import datetime

from config import (
    DB_PATH, PROMPT_PATH, OLLAMA_URL, MODEL_NAME,
    WORKING_MEMORY_TURNS, MAX_USER_CHARS, MAX_ASSISTANT_CHARS,
    VERBATIM_TURNS, MAX_MEMORY_CONTEXT, MIN_MEMORY_CONTEXT, SUMMARY_UPDATE_THRESHOLD
)
from core.ollama_client import OllamaClient
from core.memory_store import MemoryStore
from core.memory_policy import extract_memory_suggestion, is_allowed_memory, categorize
from core.conversation_summary import ConversationManager
from core.utils import load_text_file

def format_memory_context(memories) -> str:
    """Format memories as bullet points for context."""
    if not memories:
        return ""
    lines = ["Relevant context:"]
    for mem in memories:
        lines.append(f"- {mem.key} = {mem.value}")
    return "\n".join(lines)

def main():
    memory = MemoryStore(DB_PATH)
    llm = OllamaClient(OLLAMA_URL, MODEL_NAME)
    system_prompt = load_text_file(PROMPT_PATH)
    conversation = ConversationManager(llm, verbatim_turns=VERBATIM_TURNS, summary_threshold=SUMMARY_UPDATE_THRESHOLD)

    print(f"Donna (local) — model={MODEL_NAME}")
    print("Type /mem to view memory, /exit to quit.\n")

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/mem":
            items = memory.list_memories(limit=50)
            if not items:
                print("\n[No memories saved yet]\n")
                continue
            print("\n[Memories]")
            for m in items:
                print(f"- {m.key} = {m.value} ({m.category}, conf={m.confidence:.2f})")
            print()
            continue

        if len(user) > MAX_USER_CHARS:
            user = user[:MAX_USER_CHARS] + "…"

        # Get relevant memories (3-8 max)
        relevant_memories = memory.get_relevant_memories(user, min_count=MIN_MEMORY_CONTEXT, max_count=MAX_MEMORY_CONTEXT)
        memory_context = format_memory_context(relevant_memories)

        # Get conversation context (summary + recent turns)
        conv_context = conversation.get_context()

        # Build messages: system prompt + memory context + conversation context + user message
        messages = [{"role": "system", "content": system_prompt}]
        
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
        
        if conv_context:
            messages.append({"role": "system", "content": conv_context})
        
        messages.append({"role": "user", "content": user})

        # Call model with streaming
        try:
            print("\nDonna: ", end="", flush=True)
            full_response = ""
            stream = llm.chat_stream(messages)
            
            # Collect tokens and print as they arrive
            for token in stream:
                if token.startswith("__FULL_RESPONSE__"):
                    # This is the final full response yield
                    full_response = token.replace("__FULL_RESPONSE__", "", 1)
                else:
                    # Individual token
                    print(token, end="", flush=True)
                    full_response += token
            
            print("\n")  # Newline after streaming completes
            
            # Extract memory suggestion (if any)
            assistant_text, suggestion = extract_memory_suggestion(full_response)

            if len(assistant_text) > MAX_ASSISTANT_CHARS:
                assistant_text = assistant_text[:MAX_ASSISTANT_CHARS] + "…"

            # Add to conversation
            conversation.add_turn(user, assistant_text)

            # Update summary if needed
            if conversation.should_update_summary():
                conversation.update_summary()

            # Decide whether to store memory
            if suggestion:
                if is_allowed_memory(suggestion):
                    cat = categorize(suggestion.key)
                    memory.upsert_memory(suggestion.key, suggestion.value, cat, confidence=0.78)
                    print(f"[Saved memory] {suggestion.key} = {suggestion.value}\n")
                else:
                    print("[Memory suggestion rejected by policy]\n")

        except Exception as e:
            print(f"\nError calling Ollama: {e}")
            print("Make sure Ollama is running: ollama serve\n")
            continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)
