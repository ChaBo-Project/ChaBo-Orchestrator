"""
Tests for the Excel test-set loader (testset.py).

*No network required (.env / services etc.)

Covers:
- a well-formed file loading into TestCase objects
- case-insensitive column matching
- the Instructions sheet being ignored
- validation: missing required column, duplicate #, empty Expected answer
- Should be answered defaulting to Yes when blank or the column is absent
- quotes/line breaks getting stripped from cell values
"""

import pandas as pd
import pytest

from testset import InvalidTestSetError, load_testset

QUESTIONS_SHEET = "Questions"


def _write_testset(path, rows, columns, instructions_sheet=False):
    df = pd.DataFrame(rows, columns=columns)
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name=QUESTIONS_SHEET, index=False)
        if instructions_sheet:
            pd.DataFrame({"Guidance": ["fill in your questions here"]}).to_excel(
                writer, sheet_name="Instructions", index=False
            )
    return path


FULL_COLUMNS = [
    "#",
    "Question",
    "Expected answer",
    "Should be answered",
    "Source document",
    "Notes",
]


# --- happy path ---

def test_loads_a_well_formed_file(tmp_path):
    path = _write_testset(
        tmp_path / "testset.xlsx",
        rows=[
            [1, "What is wheat?", "A cereal crop.", "Yes", "wheat.pdf", "basic"],
            [2, "Is the moon made of cheese?", None, "No", None, "abstention"],
        ],
        columns=FULL_COLUMNS,
        instructions_sheet=True,
    )
    cases = load_testset(str(path))

    assert len(cases) == 2
    assert cases[0].case_id == 1
    assert cases[0].question == "What is wheat?"
    assert cases[0].expected_answer == "A cereal crop."
    assert cases[0].should_be_answered is True
    assert cases[0].source_document == "wheat.pdf"

    assert cases[1].should_be_answered is False
    assert cases[1].expected_answer is None


def test_column_names_are_matched_case_insensitively(tmp_path):
    path = _write_testset(
        tmp_path / "testset.xlsx",
        rows=[[1, "What is wheat?", "A cereal crop."]],
        columns=["#", "question", "EXPECTED ANSWER"],
    )
    cases = load_testset(str(path))
    assert cases[0].question == "What is wheat?"
    assert cases[0].expected_answer == "A cereal crop."


def test_blank_should_be_answered_defaults_to_yes(tmp_path):
    path = _write_testset(
        tmp_path / "testset.xlsx",
        rows=[[1, "What is wheat?", "A cereal crop.", None]],
        columns=["#", "Question", "Expected answer", "Should be answered"],
    )
    assert load_testset(str(path))[0].should_be_answered is True


def test_missing_should_be_answered_column_defaults_to_yes(tmp_path):
    path = _write_testset(
        tmp_path / "testset.xlsx",
        rows=[[1, "What is wheat?", "A cereal crop."]],
        columns=["#", "Question", "Expected answer"],
    )
    assert load_testset(str(path))[0].should_be_answered is True


def test_quotes_and_linebreaks_are_stripped(tmp_path):
    path = _write_testset(
        tmp_path / "testset.xlsx",
        rows=[[1, '"What is\nwheat?"', "A cereal\r\ncrop."]],
        columns=["#", "Question", "Expected answer"],
    )
    cases = load_testset(str(path))
    assert cases[0].question == "What is wheat?"
    assert cases[0].expected_answer == "A cereal crop."


# --- validation failures ---

def test_missing_required_column_raises(tmp_path):
    path = _write_testset(
        tmp_path / "testset.xlsx",
        rows=[[1, "What is wheat?"]],
        columns=["#", "Question"],  # no Expected answer column at all
    )
    with pytest.raises(InvalidTestSetError, match="missing required column"):
        load_testset(str(path))


def test_duplicate_id_raises(tmp_path):
    path = _write_testset(
        tmp_path / "testset.xlsx",
        rows=[
            [1, "What is wheat?", "A cereal crop."],
            [1, "What is maize?", "A cereal crop."],
        ],
        columns=["#", "Question", "Expected answer"],
    )
    with pytest.raises(InvalidTestSetError, match="duplicate '#'"):
        load_testset(str(path))


def test_empty_expected_answer_raises_when_should_be_answered_is_yes(tmp_path):
    path = _write_testset(
        tmp_path / "testset.xlsx",
        rows=[[1, "What is wheat?", None, "Yes"]],
        columns=["#", "Question", "Expected answer", "Should be answered"],
    )
    with pytest.raises(InvalidTestSetError, match="empty 'Expected answer'"):
        load_testset(str(path))


def test_empty_question_raises(tmp_path):
    path = _write_testset(
        tmp_path / "testset.xlsx",
        rows=[[1, None, "A cereal crop."]],
        columns=["#", "Question", "Expected answer"],
    )
    with pytest.raises(InvalidTestSetError, match="empty 'Question'"):
        load_testset(str(path))
