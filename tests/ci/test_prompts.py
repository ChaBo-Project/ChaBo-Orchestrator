"""
Tests for the generator prompt builders.

*No network required (.env / services etc.)

Covers:
- BASE_PROMPT carries no built-in follow-up question policy
- optional instance_user_instructions block in the user message
- loading instance_user_instructions from INSTANCE_CONFIG_DIR/instance.yaml
"""

from components.generator.prompts import (
    BASE_PROMPT,
    INSTANCE_USER_INSTRUCTIONS_HEADER,
    LANGUAGE_NUDGE,
    build_messages,
    load_instance_user_instructions,
)


def _user_content(**kwargs):
    messages = build_messages("SYSTEM", "What is X?", "context text", **kwargs)
    return messages[1].content


def test_base_prompt_has_no_follow_up_policy():
    assert "FOLLOW-UP" not in BASE_PROMPT
    assert "You might also want to know" not in BASE_PROMPT


def test_no_instance_block_by_default():
    content = _user_content()
    assert INSTANCE_USER_INSTRUCTIONS_HEADER not in content
    assert content.endswith(LANGUAGE_NUDGE)


def test_blank_instructions_add_no_block():
    assert INSTANCE_USER_INSTRUCTIONS_HEADER not in _user_content(instance_user_instructions="  \n ")


def test_instance_block_sits_after_question_and_before_language_nudge():
    content = _user_content(instance_user_instructions="  End with one follow-up question.  ")
    question_pos = content.index("### USER QUESTION")
    block_pos = content.index(INSTANCE_USER_INSTRUCTIONS_HEADER)
    assert question_pos < block_pos
    assert f"{INSTANCE_USER_INSTRUCTIONS_HEADER}\nEnd with one follow-up question." in content
    assert content.endswith(LANGUAGE_NUDGE)


def test_system_prompt_is_unchanged_by_instance_block():
    messages = build_messages("SYSTEM", "q", "c", instance_user_instructions="extra")
    assert messages[0].content == "SYSTEM"


def test_loader_reads_key_from_instance_yaml(tmp_path, monkeypatch):
    (tmp_path / "instance.yaml").write_text("instance_user_instructions: |\n  Be brief.\n")
    monkeypatch.setenv("INSTANCE_CONFIG_DIR", str(tmp_path))
    assert load_instance_user_instructions() == "Be brief."


def test_loader_is_empty_without_instance_config(monkeypatch):
    monkeypatch.delenv("INSTANCE_CONFIG_DIR", raising=False)
    assert load_instance_user_instructions() == ""
