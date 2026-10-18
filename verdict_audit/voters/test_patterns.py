from patterns import COMPILED


def hits(text):
    return {k for k, rx in COMPILED.items() if rx.search(text)}


def test_tool_vs_model_separation():
    assert ("tool", "claude_code") in hits("Produced with Claude Code (Opus 4.7).")
    assert ("model", "claude") not in hits("Produced with Claude Code.")
    assert ("model", "claude") in hits("Produced with Claude Code (Opus 4.7).")
    assert ("tool", "gemini_cli") in hits("ran via gemini-cli")
    assert ("model", "gemini") not in hits("ran via gemini-cli")


def test_no_false_hits_on_common_words():
    assert hits("We move the cursor and amplify the signal; o10 is not a model.") == set()
