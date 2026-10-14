"""Loads and validates the RAGAS eval test set from an Excel file.

The file has two sheets: `Questions` (the data this module reads) and `Instructions`
(guidance for content experts; ignored here). See the `Test template` section of the
eval plan for the full column reference and rules.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

QUESTIONS_SHEET = "Questions"
REQUIRED_COLUMNS = ["#", "Question", "Expected answer"]


class InvalidTestSetError(ValueError):
    """Raised when the test set file is missing a column or contains an invalid row."""


@dataclass
class TestCase:
    case_id: int
    question: str
    expected_answer: Optional[str]
    should_be_answered: bool
    source_document: Optional[str]
    notes: Optional[str]


def _clean_cell(value: object) -> Optional[str]:
    """Strip surrounding quotes/whitespace and collapse line breaks; blank/NaN -> None."""
    if pd.isna(value):
        return None
    text = str(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = text.strip().strip("\"'").strip()
    return text or None


def load_testset(path: str) -> List[TestCase]:
    """Load and validate the Questions sheet of an eval test-set Excel file.

    Raises InvalidTestSetError with a clear message on a missing required column, a duplicate
    `#`, or a missing Expected answer on a row where Should be answered is not No.
    """
    try:
        sheet = pd.read_excel(path, sheet_name=QUESTIONS_SHEET, dtype=object)
    except ValueError as exc:
        raise InvalidTestSetError(f"{path}: could not read sheet '{QUESTIONS_SHEET}': {exc}") from exc

    column_by_lower: Dict[str, str] = {str(c).strip().lower(): c for c in sheet.columns}

    missing = [name for name in REQUIRED_COLUMNS if name.lower() not in column_by_lower]
    if missing:
        raise InvalidTestSetError(
            f"{path}: missing required column(s) {missing} in sheet '{QUESTIONS_SHEET}'"
        )

    id_col = column_by_lower["#"]
    question_col = column_by_lower["question"]
    expected_answer_col = column_by_lower["expected answer"]
    should_be_answered_col = column_by_lower.get("should be answered")
    source_document_col = column_by_lower.get("source document")
    notes_col = column_by_lower.get("notes")

    cases: List[TestCase] = []
    seen_ids = set()

    for row_num, row in sheet.iterrows():
        excel_row = row_num + 2  # header is row 1; pandas rows are 0-indexed

        raw_id = row[id_col]
        if pd.isna(raw_id):
            raise InvalidTestSetError(f"{path}: row {excel_row} has an empty '#' (case ID)")
        try:
            case_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise InvalidTestSetError(
                f"{path}: row {excel_row} has a non-numeric '#': {raw_id!r}"
            ) from exc
        if case_id in seen_ids:
            raise InvalidTestSetError(f"{path}: duplicate '#' {case_id} (row {excel_row})")
        seen_ids.add(case_id)

        question = _clean_cell(row[question_col])
        if not question:
            raise InvalidTestSetError(
                f"{path}: row {excel_row} (case #{case_id}) has an empty 'Question'"
            )

        should_be_answered_raw = (
            _clean_cell(row[should_be_answered_col]) if should_be_answered_col else None
        )
        should_be_answered = (should_be_answered_raw or "yes").lower() != "no"

        expected_answer = _clean_cell(row[expected_answer_col])
        if should_be_answered and not expected_answer:
            raise InvalidTestSetError(
                f"{path}: row {excel_row} (case #{case_id}) has an empty 'Expected answer' "
                f"but 'Should be answered' is not No"
            )

        cases.append(
            TestCase(
                case_id=case_id,
                question=question,
                expected_answer=expected_answer,
                should_be_answered=should_be_answered,
                source_document=(
                    _clean_cell(row[source_document_col]) if source_document_col else None
                ),
                notes=_clean_cell(row[notes_col]) if notes_col else None,
            )
        )

    if not cases:
        raise InvalidTestSetError(f"{path}: no test cases found in sheet '{QUESTIONS_SHEET}'")

    return cases
