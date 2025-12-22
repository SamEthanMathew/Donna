#!/usr/bin/env python3
"""
Lightweight voice test - uses text input instead of STT to reduce memory.
"""

import sys
import os

# Fix for static TLS issue
os.environ['LD_PRELOAD'] = '/home/sam/jetson_agent/.venv/lib/python3.10/site-packages/scikit_learn.libs/libgomp-947d5fa1.so.1.0.0'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.tts import TextToSpeech
from llm import QwenModel


def main():
    print("=" * 60)
    print("Jetson Agent - Text-to-Voice Test (Lite)")
    print("=" * 60)
    print()
    
    print("Initializing components...")
    print()
    
    try:
        # Initialize Text-to-Speech
        print("[1/2] Loading Text-to-Speech...")
        tts = TextToSpeech(use_cuda=False)
        print()
        
        # Initialize LLM
        print("[2/2] Loading Language Model...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "llm/qwen2.5-7b-instruct-f16.gguf")
        llm = QwenModel(
            model_path=model_path,
            temperature=0.7,
            n_ctx=2048,
            n_gpu_layers=0
        )
        print()
        
        print("=" * 60)
        print("All systems ready! Text-to-voice interaction starting...")
        print("=" * 60)
        print()
        print("Instructions:")
        print("- Type your questions/messages")
        print("- The agent will respond via voice")
        print("- Type 'exit', 'quit', or 'stop' to end")
        print()
        
        # System prompt
        system_prompt = (
            "You are a helpful AI assistant running on a Jetson device. "
            "Keep your responses concise and conversational, ideally 1-2 sentences. "
            "Be friendly and helpful."
        )
        
        conversation_count = 0
        
        while True:
            try:
                print("-" * 60)
                print(f"\n[Turn {conversation_count + 1}]")
                print()
                
                # Get text input
                user_text = input("You: ").strip()
                
                if not user_text:
                    continue
                
                # Check for exit commands
                exit_words = ['exit', 'quit', 'stop', 'goodbye', 'bye']
                if user_text.lower() in exit_words:
                    farewell = "Goodbye! It was nice talking to you."
                    print(f"Agent: {farewell}")
                    tts.speak(farewell)
                    break
                
                # Generate LLM response
                print("Thinking...")
                if conversation_count == 0:
                    response = llm.chat(user_text, max_tokens=128, system_prompt=system_prompt)
                else:
                    response = llm.chat(user_text, max_tokens=128)
                
                print(f"Agent: {response}")
                print()
                
                # Speak the response
                print("Speaking...")
                tts.speak(response)
                
                conversation_count += 1
                print()
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"\nError during conversation: {e}")
                import traceback
                traceback.print_exc()
                print("\nTrying to continue...")
                continue
        
        print()
        print("=" * 60)
        print(f"Conversation ended. Total turns: {conversation_count}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

