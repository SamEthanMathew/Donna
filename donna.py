#!/usr/bin/env python3
"""
Donna - Integrated Voice Assistant
Main application that orchestrates STT, TTS, LLM, and Vision modules.
"""

import sys
import signal
import time
import argparse
import threading
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Import config
import config

# Import core modules
from core.assistant import Assistant
from core.vision_monitor import VisionMonitor

# Import STT and TTS
from stt.stt_whisper import detect_wake_word, listen_and_transcribe
from tts.core import say


class DonnaApp:
    """Main application class for Donna voice assistant."""
    
    def __init__(self):
        """Initialize the application."""
        self.assistant = None
        self.vision_monitor = None
        self.running = False
        
    def initialize(self):
        """Initialize all modules."""
        print("=" * 60)
        print(f"Initializing {config.ASSISTANT_NAME} Voice Assistant")
        print("=" * 60)
        
        # Initialize assistant
        print("\n[1/3] Initializing Assistant (LLM, Memory, Conversation)...")
        try:
            self.assistant = Assistant()
            print(f"✓ Assistant initialized (model: {config.MODEL_NAME})")
        except Exception as e:
            print(f"✗ Failed to initialize assistant: {e}")
            print("Make sure Ollama is running: ollama serve")
            return False
        
        # Initialize vision monitor (if enabled)
        if config.VISION_ENABLED:
            print("\n[2/3] Initializing Vision Monitor...")
            try:
                self.vision_monitor = VisionMonitor(self.assistant, enabled=True)
                self.vision_monitor.start()
                print("✓ Vision monitoring started")
            except Exception as e:
                print(f"⚠ Vision monitoring failed: {e}")
                print("Continuing without vision features...")
                self.vision_monitor = None
        else:
            print("\n[2/3] Vision monitoring disabled in config")
            self.vision_monitor = None
        
        # Test TTS
        print("\n[3/3] Testing TTS...")
        try:
            say("Initialization complete.")
            print("✓ TTS working")
        except Exception as e:
            print(f"⚠ TTS test failed: {e}")
            print("Continuing anyway...")
        
        print("\n" + "=" * 60)
        print(f"{config.ASSISTANT_NAME} is ready!")
        print("=" * 60)
        print(f"\nWake word: '{config.WAKE_WORD}'")
        print("Say the wake word to start a conversation.")
        print("Press Ctrl+C to exit.\n")
        
        return True
    
    def run_voice_loop(self):
        """Main voice interaction loop."""
        self.running = True
        
        while self.running:
            try:
                # Listen for wake word
                if config.LOG_LEVEL == "DEBUG":
                    print("[Listening for wake word...]")
                
                # This will block until wake word is detected
                if detect_wake_word():
                    print(f"\n✓ Wake word '{config.WAKE_WORD}' detected!")
                    
                    # Record and transcribe
                    try:
                        print("Recording...")
                        user_text = listen_and_transcribe()
                        
                        if not user_text or not user_text.strip():
                            print("No speech detected. Listening again...\n")
                            continue
                        
                        print(f"You: {user_text}")
                        
                        # Process with assistant
                        print(f"\n{config.ASSISTANT_NAME}: ", end="", flush=True)
                        response = self.assistant.process(
                            user_text,
                            stream_callback=lambda token: print(token, end="", flush=True)
                        )
                        print("\n")  # Newline after streaming
                        
                        # Speak response
                        say(response)
                        print()  # Blank line for readability
                        
                    except KeyboardInterrupt:
                        print("\nInterrupted by user")
                        break
                    except Exception as e:
                        error_msg = f"Error processing input: {e}"
                        print(f"\n{error_msg}")
                        say("Sorry, I encountered an error. Please try again.")
                        if config.LOG_LEVEL == "DEBUG":
                            import traceback
                            traceback.print_exc()
                
            except KeyboardInterrupt:
                print("\n\nShutting down...")
                break
            except Exception as e:
                print(f"\nError in voice loop: {e}")
                if config.LOG_LEVEL == "DEBUG":
                    import traceback
                    traceback.print_exc()
                time.sleep(1)  # Brief pause before retrying
    
    def shutdown(self):
        """Gracefully shutdown the application."""
        print("\nShutting down...")
        self.running = False
        
        if self.vision_monitor:
            self.vision_monitor.stop()
            self.vision_monitor.join(timeout=2.0)
        
        print("Goodbye!")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\nReceived interrupt signal")
    if 'app' in globals():
        app.shutdown()
    sys.exit(0)


def start_web_server():
    """Start the FastAPI web server."""
    import uvicorn
    from web.app import app
    
    print(f"\n[Web] Starting web server at http://{config.WEB_HOST}:{config.WEB_PORT}")
    print(f"[Web] Open http://{config.WEB_HOST}:{config.WEB_PORT} in your browser")
    
    uvicorn.run(
        app,
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        log_level=config.LOG_LEVEL.lower()
    )


def main():
    """Main entry point."""
    global app
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Donna Voice Assistant")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start web server instead of voice loop"
    )
    parser.add_argument(
        "--mode",
        choices=["voice", "web", "both"],
        default="voice",
        help="Operation mode: voice (default), web, or both"
    )
    
    args = parser.parse_args()
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Web-only mode
    if args.web or args.mode == "web":
        if not config.WEB_ENABLED:
            print("Web server is disabled in config. Set WEB_ENABLED=True to enable.")
            sys.exit(1)
        start_web_server()
        return
    
    # Both mode: start web server in background thread
    if args.mode == "both":
        if config.WEB_ENABLED:
            web_thread = threading.Thread(target=start_web_server, daemon=True)
            web_thread.start()
            print(f"\n[Web] Web server starting in background at http://{config.WEB_HOST}:{config.WEB_PORT}")
        else:
            print("[Web] Web server disabled in config")
    
    # Create and initialize app
    app = DonnaApp()
    if not app.initialize():
        print("\nInitialization failed. Exiting.")
        sys.exit(1)
    
    # Run main voice loop
    try:
        app.run_voice_loop()
    except Exception as e:
        print(f"\nFatal error: {e}")
        if config.LOG_LEVEL == "DEBUG":
            import traceback
            traceback.print_exc()
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()

