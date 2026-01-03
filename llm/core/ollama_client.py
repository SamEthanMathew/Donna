import requests
import json
from typing import List, Dict, Any, Iterator

class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.6) -> str:
        """
        Uses Ollama chat API (non-streaming).
        """
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
            },
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]

    def chat_stream(self, messages: List[Dict[str, str]], temperature: float = 0.6) -> Iterator[str]:
        """
        Uses Ollama chat API with streaming.
        Yields tokens as they arrive. The final yield is the complete response string.
        """
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
            },
            "stream": True,
        }
        r = requests.post(url, json=payload, timeout=120, stream=True)
        r.raise_for_status()
        
        full_response = ""
        for line in r.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        token = data["message"]["content"]
                        full_response += token
                        yield token
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
        
        # Yield full response as final item (marked with special prefix for identification)
        yield f"__FULL_RESPONSE__{full_response}"


