"""Frozen regex dictionary for Q3 (agent and model mentions). See PROTOCOL.md."""
import re

_F = re.IGNORECASE

AGENT_TOOLS = {
    "claude_code": r"\bclaude[\s-]?code\b",
    "codex": r"\bcodex\b",
    "cursor": r"\bcursor\s+(?:agent|ide|editor|composer)\b|\bcursor\.(?:sh|com)\b",
    "gemini_cli": r"\bgemini[\s-]cli\b",
    "aider": r"\baider\b",
    "openhands": r"\bopen[\s-]?hands\b",
    "devin": r"\bdevin\b",
    "copilot": r"\b(?:github\s+)?copilot\b",
    "cline": r"\bcline\b",
    "windsurf": r"\bwindsurf\b",
    "manus": r"\bmanus\b",
    "kimi_cli": r"\bkimi[\s-](?:cli|code)\b",
    "qwen_code": r"\bqwen[\s-]code\b",
    "opencode": r"\bopencode\b",
    "goose": r"\bgoose\b",
    "amp": r"\bamp(?:code)?\s+(?:agent|cli)\b|\bampcode\b",
}

MODEL_FAMILIES = {
    "claude": r"\bclaude\b(?![\s-]?code)|\b(?:opus|sonnet|haiku)\s*\d",
    "gpt": r"\bgpt[\s-]?(?:4o|4\.1|4|5(?:\.\d)?)\b|\bchatgpt\b",
    "openai_o": r"\bo(?:1|3|4)(?:-mini|-pro)?\b",
    "gemini": r"\bgemini\b(?![\s-]cli)",
    "glm": r"\bglm[\s-]?\d",
    "qwen": r"\bqwen\d*\b(?![\s-]code)",
    "deepseek": r"\bdeepseek\b",
    "kimi": r"\bkimi\b(?![\s-](?:cli|code))",
    "llama": r"\bllama[\s-]?\d",
    "mistral": r"\bmistral\b",
    "grok": r"\bgrok[\s-]?\d",
}

COMPILED = {("tool", k): re.compile(v, _F) for k, v in AGENT_TOOLS.items()}
COMPILED.update({("model", k): re.compile(v, _F) for k, v in MODEL_FAMILIES.items()})
