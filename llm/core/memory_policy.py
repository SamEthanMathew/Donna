import re
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MemorySuggestion:
    key: str
    value: str
    reason: str

MEMORY_BLOCK_RE = re.compile(
    r"MEMORY_SUGGESTION:\s*key:\s*\"(?P<key>.*?)\"\s*value:\s*\"(?P<value>.*?)\"\s*reason:\s*\"(?P<reason>.*?)\"",
    re.DOTALL
)

def extract_memory_suggestion(text: str) -> Tuple[str, Optional[MemorySuggestion]]:
    """
    Returns (assistant_text_without_memory_block, suggestion_or_none)
    """
    m = MEMORY_BLOCK_RE.search(text)
    if not m:
        return text.strip(), None

    suggestion = MemorySuggestion(
        key=m.group("key").strip(),
        value=m.group("value").strip(),
        reason=m.group("reason").strip()
    )

    cleaned = (text[:m.start()] + text[m.end():]).strip()
    return cleaned, suggestion

def is_allowed_memory(s: MemorySuggestion) -> bool:
    """
    Hard guardrails. We store only non-sensitive, stable-ish info.
    """
    key = s.key.lower()
    value = s.value.lower()

    # Disallow super sensitive categories (keep this strict early).
    banned_markers = [
        "ssn", "social security", "credit card", "password", "bank account",
        "medical", "diagnosis", "sexual", "political affiliation",
        "religion", "race", "ethnicity"
    ]
    if any(b in key or b in value for b in banned_markers):
        return False

    # Prevent huge blobs.
    if len(s.key) > 60 or len(s.value) > 200:
        return False

    # Basic sanity: key should look like namespace.something
    if "." not in s.key:
        return False

    return True

def categorize(key: str) -> str:
    k = key.lower()
    if k.startswith("preference."):
        return "preference"
    if k.startswith("routine."):
        return "routine"
    if k.startswith("goal."):
        return "goal"
    if k.startswith("constraint."):
        return "constraint"
    if k.startswith("project."):
        return "project"
    if k.startswith("person."):
        return "person"
    return "misc"


