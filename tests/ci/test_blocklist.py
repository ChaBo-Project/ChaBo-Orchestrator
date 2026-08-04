"""
Tests output guard streaming blocklist filter   

*No network required (.env / services etc.)

Pure text logic - covers:
- clean text passing through
- blocked word stopping the stream
- term split across streamed chunks still being caught,
- whole-word matching, single-notice behaviour
- non-English stuff (CJK substring matching / Arabic text normalisation)
"""

from components.guardrails.output_guard import (
    compile_blocklist,
    normalize_arabic,
    StreamingBlocklistFilter,
)


def _filter(terms_by_lang, notice="[blocked]"):
    return StreamingBlocklistFilter(compile_blocklist(terms_by_lang), notice)


def _run(text_filter, chunks):
    """Feed every chunk, then flush. Returns (all_emitted_text, was_blocked)."""
    out = []
    for chunk in chunks:
        text, hit = text_filter.feed(chunk)
        out.append(text)
        if hit:
            return "".join(out), True
    text, hit = text_filter.flush_final()
    out.append(text)
    return "".join(out), hit


def test_clean_text_passes_through_unchanged():
    text, blocked = _run(_filter({"en": ["forbidden"]}), ["hello ", "world, all good here"])
    assert blocked is False
    assert text == "hello world, all good here"


def test_a_blocked_word_stops_the_stream():
    text, blocked = _run(_filter({"en": ["forbidden"]}), ["this is forbidden text"])
    assert blocked is True
    assert "[blocked]" in text


def test_a_term_split_across_chunks_is_still_caught():
    _, blocked = _run(_filter({"en": ["forbidden"]}), ["this is forbi", "dden text"])
    assert blocked is True


def test_whole_word_matching_does_not_flag_a_larger_word():
    # "ass" is matched as a whole word only, so it must not match inside "classic".
    _, blocked = _run(_filter({"en": ["ass"]}), ["this is a classic example"])
    assert blocked is False


def test_the_notice_is_returned_only_once():
    text_filter = _filter({"en": ["forbidden"]})
    first, hit1 = text_filter.feed("this is forbidden now")
    assert hit1 is True
    assert "[blocked]" in first
    again, hit2 = text_filter.feed(" and more")
    assert hit2 is True
    assert again == ""


def test_cjk_terms_match_as_a_substring():
    # Chinese has no word separators, so its terms are matched as substrings.
    _, blocked = _run(_filter({"zh": ["禁止"]}), ["这是禁止的内容"])
    assert blocked is True


def test_normalize_arabic_folds_letter_variants_and_strips_diacritics():
    # The alef variants all fold to a plain alef, so spellings match each other.
    assert normalize_arabic("أحمد") == normalize_arabic("احمد")
