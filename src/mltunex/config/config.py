from dataclasses import dataclass
from mltunex.ai_handler.prompt import LLMPrompts

@dataclass
class OpenAIConfig:
    model: str = "gpt-4o"
    temperature: float = 0
    SYSTEM_PROMPT: str = LLMPrompts.OpenAIPrompt