from llama_cpp import Llama
from typing import List, Dict, Optional
import os


class QwenModel:
    """
    Qwen 2.5 7B language model loader using llama-cpp-python.
    
    Supports chat-style conversations with context history.
    """
    
    def __init__(
        self,
        model_path: str = "llm/qwen2.5-7b-instruct-f16.gguf",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        temperature: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize Qwen model.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (tokens)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            temperature: Sampling temperature (0.0 to 1.0)
            verbose: Whether to print debug info
        """
        self.model_path = model_path
        self.temperature = temperature
        self.chat_history: List[Dict[str, str]] = []
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        print(f"Loading Qwen model from {model_path}...")
        print(f"Context size: {n_ctx}, GPU layers: {n_gpu_layers}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            n_threads=4,  # Adjust based on CPU cores
            chat_format="chatml"  # Qwen uses ChatML format
        )
        
        print("Qwen model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate a response from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses default if None)
            stop: Stop sequences
            
        Returns:
            Generated text response
        """
        if temperature is None:
            temperature = self.temperature
        
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or ["</s>", "<|endoftext|>", "<|im_end|>"],
            echo=False
        )
        
        return response["choices"][0]["text"].strip()
    
    def chat(
        self,
        message: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Chat with the model using conversation history.
        
        Args:
            message: User message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses default if None)
            system_prompt: Optional system prompt for this turn
            
        Returns:
            Assistant's response
        """
        if temperature is None:
            temperature = self.temperature
        
        # Add user message to history
        self.chat_history.append({
            "role": "user",
            "content": message
        })
        
        # Build messages list with optional system prompt
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.extend(self.chat_history)
        
        # Generate response
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "<|endoftext|>", "<|im_end|>"]
        )
        
        assistant_message = response["choices"][0]["message"]["content"].strip()
        
        # Add assistant response to history
        self.chat_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def reset_history(self):
        """Clear chat history."""
        self.chat_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get current chat history."""
        return self.chat_history.copy()


# Example usage
if __name__ == "__main__":
    import sys
    
    # Initialize model
    model = QwenModel()
    
    print("\n=== Qwen 2.5 7B Chat Test ===")
    print("Type 'exit' to quit, 'reset' to clear history\n")
    
    # Simple chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                model.reset_history()
                print("Chat history cleared.")
                continue
            
            # Generate response
            print("Agent: ", end="", flush=True)
            response = model.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


