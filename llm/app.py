import sys
from datetime import datetime

from config import (
    DB_PATH, PROMPT_PATH, OLLAMA_URL, MODEL_NAME,
    WORKING_MEMORY_TURNS, MAX_USER_CHARS, MAX_ASSISTANT_CHARS
)
from core.ollama_client import OllamaClient
from core.memory_store import MemoryStore
from core.memory_policy import extract_memory_suggestion, is_allowed_memory, categorize
from core.utils import load_text_file

def main():
    memory = MemoryStore(DB_PATH)
    llm = OllamaClient(OLLAMA_URL, MODEL_NAME)
    system_prompt = load_text_file(PROMPT_PATH)

    print(f"Donna (local) — model={MODEL_NAME}")
    print("Type /mem to view memory, /exit to quit.\n")

    conversation_history = []

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

        # Build messages: system prompt + conversation history + new user message
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user})

        # Call model
        try:
            raw = llm.chat(messages)
        except Exception as e:
            print(f"\nError calling Ollama: {e}")
            print("Make sure Ollama is running: ollama serve\n")
            continue

        # Extract memory suggestion (if any)
        assistant_text, suggestion = extract_memory_suggestion(raw)

        if len(assistant_text) > MAX_ASSISTANT_CHARS:
            assistant_text = assistant_text[:MAX_ASSISTANT_CHARS] + "…"

        # Add to conversation history
        conversation_history.append({"role": "user", "content": user})
        conversation_history.append({"role": "assistant", "content": assistant_text})

        # Keep only last N turns (each turn is user+assistant pair, so 2*N messages)
        if len(conversation_history) > WORKING_MEMORY_TURNS * 2:
            conversation_history = conversation_history[-(WORKING_MEMORY_TURNS * 2):]

        # Print assistant
        print(f"\nDonna: {assistant_text}\n")

        # Decide whether to store memory
        if suggestion:
            if is_allowed_memory(suggestion):
                cat = categorize(suggestion.key)
                # default confidence; you can adjust based on reason or user confirmation later
                memory.upsert_memory(suggestion.key, suggestion.value, cat, confidence=0.78)
                print(f"[Saved memory] {suggestion.key} = {suggestion.value}\n")
            else:
                print("[Memory suggestion rejected by policy]\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)


