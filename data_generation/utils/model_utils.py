"""Model detection utilities."""


def is_thinking_model(model_name: str) -> bool:
    """Check if the model uses <think> tokens and requires enable_thinking=True.

    Matches: QwQ, DeepSeek-R1, R1-distill, *Thinking* models.
    """
    model_lower = model_name.lower()
    thinking_patterns = [
        "qwq",
        "deepseek-r1",
        "r1-distill",
        "thinking",
    ]
    return any(pattern in model_lower for pattern in thinking_patterns)
