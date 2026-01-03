from dataclasses import dataclass, field
from typing import List, Dict
from core.ollama_client import OllamaClient

@dataclass
class ConversationState:
    recent_turns: List[Dict[str, str]] = field(default_factory=list)  # Last 4 turns verbatim
    summary: str = ""  # 5-10 line rolling summary
    turn_count: int = 0  # Counter for when to update summary
    archived_turns: List[Dict[str, str]] = field(default_factory=list)  # Older turns to summarize

class ConversationManager:
    def __init__(self, llm: OllamaClient, verbatim_turns: int = 4, summary_threshold: int = 6):
        self.llm = llm
        self.verbatim_turns = verbatim_turns
        self.summary_threshold = summary_threshold
        self.state = ConversationState()

    def add_turn(self, user_message: str, assistant_message: str):
        """Add a new conversation turn."""
        self.state.recent_turns.append({"role": "user", "content": user_message})
        self.state.recent_turns.append({"role": "assistant", "content": assistant_message})
        self.state.turn_count += 1

        # Keep only last N turns verbatim (each turn = 2 messages: user + assistant)
        max_messages = self.verbatim_turns * 2
        if len(self.state.recent_turns) > max_messages:
            # Move oldest turns to archived
            excess = len(self.state.recent_turns) - max_messages
            self.state.archived_turns.extend(self.state.recent_turns[:excess])
            self.state.recent_turns = self.state.recent_turns[excess:]

    def update_summary(self) -> str:
        """
        Update the conversation summary using LLM.
        Summarizes archived_turns and existing summary.
        """
        if not self.state.archived_turns and not self.state.summary:
            return ""

        # Build summary prompt
        summary_prompt = """Summarize the following conversation history into 5-10 concise bullet points.
Focus on key topics, decisions, and important information.
Keep it brief and actionable.

Previous summary:
{existing_summary}

Conversation to summarize:
{conversation}

Provide a concise summary in 5-10 bullet points:""".format(
            existing_summary=self.state.summary if self.state.summary else "(none)",
            conversation=self._format_turns(self.state.archived_turns)
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates concise conversation summaries."},
            {"role": "user", "content": summary_prompt}
        ]

        try:
            new_summary = self.llm.chat(messages, temperature=0.3)  # Lower temp for more consistent summaries
            self.state.summary = new_summary.strip()
            self.state.archived_turns = []  # Clear archived after summarizing
            self.state.turn_count = 0  # Reset counter
            return self.state.summary
        except Exception as e:
            print(f"[Warning] Summary update failed: {e}")
            return self.state.summary  # Return existing summary on error

    def should_update_summary(self) -> bool:
        """Check if summary should be updated."""
        return self.state.turn_count >= self.summary_threshold

    def get_context(self) -> str:
        """
        Returns formatted context: summary + recent turns.
        """
        parts = []
        
        if self.state.summary:
            parts.append(f"Previous conversation summary:\n{self.state.summary}")
        
        if self.state.recent_turns:
            parts.append(f"\nRecent turns:\n{self._format_turns(self.state.recent_turns)}")
        
        return "\n".join(parts) if parts else ""

    def _format_turns(self, turns: List[Dict[str, str]]) -> str:
        """Format conversation turns for display."""
        lines = []
        for turn in turns:
            role = turn["role"]
            content = turn["content"]
            lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(lines)

