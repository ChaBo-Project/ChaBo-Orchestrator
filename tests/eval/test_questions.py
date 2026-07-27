# Test questions for RAG evaluation
# Edit each subset to add, remove, or swap queries without touching eval logic.
#
# Three subsets:
#   standalone_questions  — each query explicitly contains a filterable value; evaluated with no history
#   history_blocks        — conversation sequences; later turns rely on history carry-forward for filter extraction
#                           each block is a dict with:
#                             "turns": list of user turns (oldest → newest); only the LAST turn is evaluated
#                             "expected_filters": the filter dict expected after carry-forward, or None
#                             "expected_answer": rough ground truth answer for the last turn
#                             "expected_sources": list of {filename, page (optional)} dicts
#   safeguard_questions   — multi-field combos likely to yield 0 results; tests AND-fallback to priority field
#
# expected_filters: dict matching the fields in params.cfg [metadata_filters], or None if no filter is expected.
# expected_answer: rough ground truth string; used by RAGAS for context_recall and answer_relevancy.
# expected_sources: list of dicts with "filename" and optional "page"; ground-truth source(s) for this
#   question. NOT currently read by any eval mode (RAGAS included) — reserved for a future deterministic
#   retrieval-hit check (compare against retrieved doc metadata), separate from RAGAS's LLM-based metrics.

# --- Subset 1: Standalone ---
standalone_questions = [
    {
        "question": "What are the best irrigation techniques for wheat?",
        "expected_filters": {"crop": "wheat"},
        "expected_answer": "TODO: fill in expected answer",
        "expected_sources": [
            {"filename": "TODO_document.pdf", "page": 1},
        ],
    },
    {
        "question": "How can I protect my maize crops from insects?",
        "expected_filters": {"crop": "maize"},
        "expected_answer": "TODO: fill in expected answer",
        "expected_sources": [
            {"filename": "TODO_document.pdf", "page": 1},
        ],
    },
]

# --- Subset 2: History blocks ---
# Each block = dict with "turns" (list of user turns, oldest → newest) and "expected_filters".
# The last turn is the one being evaluated; all prior turns become user_messages_history.
history_blocks = [
    {
        "turns": [
            "I'm looking for information about wheat crop management.",   # establishes crop filter
            "What are the recommended pesticide applications?",           # vague — relies on history
        ],
        "expected_filters": {"crop": "wheat"},
        "expected_answer": "TODO: fill in expected answer",
        "expected_sources": [
            {"filename": "TODO_document.pdf", "page": 1},
        ],
    },
    {
        "turns": [
            "Tell me about maize cultivation in arid regions.",           # establishes crop filter
            "What soil preparation is needed?",                           # vague — relies on history
        ],
        "expected_filters": {"crop": "maize"},
        "expected_answer": "TODO: fill in expected answer",
        "expected_sources": [
            {"filename": "TODO_document.pdf", "page": 1},
        ],
    },
]

# --- Subset 3: Safeguard ---
# Queries that combine fields in a contradictory or unlikely way.
# expected_filters: None means we expect the LLM to extract no clean filter.
# expected_answer: the system should return a graceful no-answer for these queries.
# expected_sources: empty — no relevant sources expected.
safeguard_questions = [
    {
        "question": "What does the maize report on wheat fertilisation say?",   # contradictory crop + title combo
        "expected_filters": None,
        "expected_answer": "TODO: fill in expected answer",
        "expected_sources": [],
    },
]
