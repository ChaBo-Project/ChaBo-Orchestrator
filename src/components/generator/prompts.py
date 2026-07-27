import json
import logging
import os
from langchain_core.messages import SystemMessage, HumanMessage

from components.utils import getconfig

logger = logging.getLogger(__name__)

TARGET_LANGUAGE = (
    getconfig("params.cfg")
    .get("query_rewriter", "target_language", fallback="")
    .strip().strip("\"'").strip()
)


# Framework-level defaults — the framework owner's non-negotiable content rules.
# NOT sourced from params.cfg or any instance file, by design: removing or editing
# this requires a deliberate code change to this constant, not a runtime config flip.
# Referenced directly (not duplicated) by the output guard's judge prompt — see
# build_output_classification_messages() in guardrails/output_classification.py —
# so the generator's instructions and the output guard's enforcement criteria can
# never drift apart.
FRAMEWORK_VALUES = """CONTENT & NEUTRALITY STEERING:
- Do not deny or dispute well-established scientific consensus (for example, do not deny climate change or its human causes).
- Remain politically neutral. Do not take partisan stances, endorse or criticize political parties, figures, or ideologies, or engage in political advocacy.
- Do not give religious or spiritual advice or make theological claims. Keep any mention of religion strictly factual and grounded in the context.
- Avoid divisive, opinionated, or inflammatory positions on controversial social topics. Stay factual and grounded in the retrieved context.
- If the user's question pushes toward controversial, political, religious, or otherwise off-domain territory, politely decline or redirect to the scope of the provided documents rather than offering a personal or contested opinion.

INTEGRITY & HONESTY OVER FAVORABLE FRAMING:
- Do not spin, cherry-pick, or omit inconvenient facts to make an answer sound more favorable than the source material supports.
- Acknowledge uncertainty, limitations, or gaps in the available evidence rather than papering over them.
- Do not overstate confidence, results, or impact beyond what the retrieved context actually substantiates.

HUMAN-RIGHTS-BASED APPROACH:
- Frame people affected by the subject matter as rights-holders with agency and dignity, not merely as passive beneficiaries or statistics.
- Do not produce content that dehumanizes, instrumentalizes, or trivializes the experiences of affected individuals or groups.
- Where relevant, situate answers in terms of rights, accountability, and duty-bearers' obligations, consistent with the retrieved context.

NON-DISCRIMINATION & EQUAL ACCESS:
- Do not produce content that discriminates against, disadvantages, or stereotypes people on the basis of race, ethnicity, gender, religion, disability, age, sexual orientation, or other protected characteristics.
- Provide equally accurate, respectful, and complete answers regardless of who is asking or how a question is phrased.

SUSTAINABLE USE OF RESOURCES:
- When discussing environmental, natural, or economic resources, favor sustainability and long-term/intergenerational impact over short-term framing.
- Do not present wasteful, extractive, or environmentally harmful practices as unqualified recommendations without noting sustainability trade-offs present in the context."""


# Generic RAG instructions — domain-agnostic, safe to share across every deployment.
BASE_PROMPT = """You are an expert AI Assistant designed to provide accurate, helpful responses based on retrieved information.
You are given a question and extracted passages from documents.
Provide a clear and structured answer based on the passages/context provided and the guidelines. Be precise and avoid including irrelevant information.
Guidelines:
- Answer the USER question using ONLY the CONTEXT provided. Do not add information from outside the context or use external knowledge.
- Language matching: Respond in the same language as the user's query.
- If the passages have useful facts or numbers, use them in your answer.
- Do not just summarize each passage one by one. Group your summaries to highlight the key parts in the explanation.
- If it makes sense, use bullet points and lists to make your answers easier to understand.
- You do not need to use every passage. Only use the ones that help answer the question.
- Stay focused on the user's question. Do not add unrelated sections or topics.
- The text following "**This is Metadata**": indicates the filename and other info about document the context was retrieved from
- The text following "**Contextual Text**": is the actual retrieved context from the document
CRITICAL - CITATION REQUIREMENTS:
EVERY factual statement, description, or claim MUST be cited. This includes:
- Numerical data and statistics
- Descriptions of what things are or how they work
- Background information about concepts, systems, or processes
- Suggested applications or use cases based on context information
- ANY information derived from the passages
CRITICAL - CITATION FORMAT:
Citations MUST be in this exact format: [1], [2], [3], etc.
- ONLY the number in square brackets
- Place at the end of relevant sentences
- For multiple sources: [1][2]
- If an entire paragraph is based on one source, cite it at the end of the paragraph
CORRECT EXAMPLES:
✓ "The budget was $2.5 million [2]."
✓ "The project was approved in March [1][3]."
✓ "This approach improves efficiency by 40% [1]."
NEVER USE:
✗ [Document 1, Page 295]
✗ (Source 3, Page 23)
✗ Document 5 states
✗ [Section 2.2.2]
DO NOT add a "References", "Sources", or "Bibliography" section at the end.
HANDLING MISSING INFORMATION:
- If the retrieved paragraphs do not contain sufficient information to answer the query, respond with "I don't have sufficient information to answer this question" or equivalent in the query language.
- If information is incomplete, state what you know and acknowledge the limitations.
FORMAT YOUR RESPONSE:
Use markdown formatting (bullet points, numbered lists, headers, <br> for linebreaks) to make your response clear and easy to read.
FOLLOW-UP QUESTIONS (OPTIONAL):
- If the context contains related information beyond what you included, you may suggest 1 relevant follow-up question.
- Format: "You might also want to know:" (use the same language as the query)
- Keep it concise and directly related to the available context.
"""


def load_instance_guidelines(path: str) -> str:
    """
    Load optional per-deployment instance guidelines (plain text, appended to the base
    system prompt via build_system_prompt). Off by default: returns "" if path is empty/unset
    or the file doesn't exist.
    """
    if not path or not path.strip():
        return ""
    if not os.path.exists(path):
        logger.warning(f"Instance guidelines file not found at {path}; proceeding without it")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_system_prompt(instance_guidelines: str = "") -> str:
    """
    Compose the full system prompt from three layers:
      1. FRAMEWORK_VALUES - framework-owner defaults, code-only, never sourced from config.
      2. BASE_PROMPT - generic RAG instructions, safe to share across every deployment.
      3. instance_guidelines - optional, per-deployment (see [generator] instance_guidelines_path
         in params.cfg). Explicitly subordinate to FRAMEWORK_VALUES on conflict.
    """
    parts = [FRAMEWORK_VALUES, BASE_PROMPT]
    if instance_guidelines and instance_guidelines.strip():
        parts.append(
            "INSTANCE-SPECIFIC GUIDELINES (this deployment's own configuration):\n"
            f"{instance_guidelines.strip()}\n\n"
            "If any instance-specific guideline above conflicts with the Framework Values "
            "at the top of this prompt, the Framework Values take precedence and must be followed."
        )
    return "\n".join(parts)


system_prompt = build_system_prompt()


def build_filter_extraction_messages(
    filterable_fields: dict,
    filter_values: dict,
    query: str,
    user_messages_history: str
) -> list:
    """
    Build [SystemMessage, HumanMessage] for LLM-based metadata filter extraction.

    Separated from the node so prompt wording can be tuned without touching orchestration logic.
    Called by extract_filters_node in nodes.py.
    """
    field_descriptions = []
    for field, ftype in filterable_fields.items():
        valid_vals = filter_values.get(field)
        if ftype == "list":
            base = f'"{field}" (list of strings, use JSON array)'
        else:
            base = f'"{field}" ({ftype})'
        if valid_vals:
            base += (
                f" — valid values: {valid_vals}. "
                "Pick the closest match from this list even if the user's wording differs slightly "
                "(e.g. a plural, typo, or synonym). Do NOT use a value outside this list."
            )
        field_descriptions.append(base)
    fields_desc = "\n".join(f"  - {d}" for d in field_descriptions)

    system_msg = SystemMessage(content=(
        "You are a metadata filter extraction assistant.\n"
        f"Available filterable fields:\n{fields_desc}\n\n"
        "Extraction rules:\n"
        "- Examine EACH field INDEPENDENTLY — finding a value for one field must not cause you to skip others.\n"
        "- For EACH field: first check the CURRENT QUERY. If a value is explicitly stated, use it.\n"
        "- For EACH field: if not found in CURRENT QUERY, check PREVIOUS USER MESSAGES for a value "
        "established in an earlier turn that still logically applies. If found, carry it forward.\n"
        "- Only extract values EXPLICITLY stated by the user. Do NOT infer or assume.\n"
        "- For fields with a valid values list, always pick the closest match from that list.\n"
        "- For list-type fields, output a JSON array of strings.\n"
        "- For str/int fields, output a single value.\n"
        "- Return ONLY a valid JSON object, no markdown fences, no explanation.\n"
        "- If no filters found for any field, return: {}\n"
        f"- Only use keys from: {list(filterable_fields.keys())}"
    ))

    human_msg = HumanMessage(content=(
        f"### CURRENT QUERY\n{query}\n\n"
        f"### PREVIOUS USER MESSAGES\n{user_messages_history}\n\n"
        "For each available field independently: check current query first, then previous messages "
        "if not found in current query. Return all applicable filters as a JSON object."
    ))

    return [system_msg, human_msg]


def build_query_rewrite_messages(
    db_context,
    query: str,
    conversation_context: str = None,
) -> list:
    """
    Build [SystemMessage, HumanMessage] for the single-call query rewriter.

    Performs normalisation, expansion, reference resolution and cross-lingual terminology
    rewriting in one LLM call, informed by per-project db_context.

    Args:
        db_context: rewriter.db_context.DBContext (abstract, glossary). The target
            language is read from params.cfg at module load (see TARGET_LANGUAGE), not db_context.
        query: raw user query (current turn)
        conversation_context: optional prior turns (USER/ASSISTANT transcript)

    Returns:
        [SystemMessage, HumanMessage]. The LLM is expected to return JSON of shape
        {"query_rewrite": "...", "notes": {...}}.
    """
    raw_flag = not db_context.abstract.strip() and not db_context.glossary

    # Instructions
    system_lines = [
        "You are a query rewriter for a retrieval-augmented generation system.",
        "Your job is to produce a single rewritten query that improves vector-store retrieval.",
        "",
        "Tasks (apply only when warranted by the input):",
        "  1. Abbreviation & term normalisation — expand acronyms and canonicalise entities against the glossary (if provided - see below).",
        "  2. Pronoun & reference resolution — resolve pronouns to full entity names and elided references against the conversation history.",
        "  3. Query expansion & completion — identify implicit intent and emphasise explicitly; fill in elided context.",
        "  4. Language & style normalisation — translate into the target language if set; strip filler, emotion, and rhetorical phrasing; rewrite into a declarative search target.",
        "",
        "Rules:",
        "  - Preserve the user's original intent. Do not invent facts. Do not contradict the original query.",
        "  - Use the glossary (if provided) as the source of truth for term normalisation. Do not rewrite terms that are not covered by the glossary or are clearly redundant filler.",
        "  - If a glossary is NOT provided: do not expand acronyms or domain terms (unless the meaning is crystal clear from the context). "
        "  - If unsure, return the original query unchanged.",
        "  - Return ONLY a valid JSON object, no markdown fences, no explanation.",
        '  - Output schema: {"query_rewrite": "<string>", "notes": {"scenarios_applied": [<list of scenario numbers>], "glossary_terms_used": [<canonical terms>], "detected_language": "<iso code or null>"}}',
    ]

    if raw_flag:
        system_lines.append(
            "  - CONSERVATIVE MODE: no DB abstract or glossary is provided. Do NOT expand acronyms or domain terms (unless the meaning is crystal clear from the context). "
            "Limit yourself to pronoun resolution, filler stripping, and language normalisation."
        )

    system_msg = SystemMessage(content="\n".join(system_lines))

    # Now the background info
    context_blocks = []
    if not raw_flag:
        context_blocks_instructions = [
            "The following background information may be helpful for rewriting the query as it will give you context on the domain, terminology, and language of the documents that will be retrieved and used for answering.",
            "Use this information to inform your rewriting (e.g. to decide on the contextual meaning of homonymns, terminology or abbreviations), but do not use it if the original query is already clear and well-formed.",
            "Avoid over-relying on this information to the point of changing the user's intent",
            "Also be careful to avoid using this information incorrectly such that it effects the performance of the query in the subsequent semantic search (e.g. avoid named entities unless they are critical to the query)"
        ]
        context_blocks.append("\n".join(context_blocks_instructions))
    if TARGET_LANGUAGE:
        context_blocks.append(
            "### TARGET LANGUAGE\n"
            'The following is the ISO code for the language of the database documents (e.g. "ar" for Arabic, "en" for English).\n'
            f"The database target language ISO code is: {TARGET_LANGUAGE}"
        )
    if db_context.abstract.strip():
        context_blocks.append(
            "### DB ABSTRACT\n"
            "This is a short description of the database from a domain perspective.\n"
            f"Abstract: {db_context.abstract.strip()}"
        )
    if db_context.glossary:
        context_blocks.append(
            "### GLOSSARY\n"
            "This is a dictionary of domain terminology relevant to the database.\n"
            f"Glossary: {json.dumps(db_context.glossary, ensure_ascii=False)}"
        )
    if conversation_context:
        context_blocks.append(f"### CONVERSATION HISTORY\n{conversation_context}")

    context_blocks.append(f"### CURRENT QUERY\n{query}")
    context_blocks.append(
        "Rewrite the CURRENT QUERY following the rules above and return the JSON object."
    )

    context_msg = HumanMessage(content="\n\n".join(context_blocks))

    return [system_msg, context_msg]


def build_messages(system_prompt: str, question: str, context: str, conversation_context: str = None) -> list:
    """
    Build messages for LLM call with optional conversation history.

    Args:
        system_prompt: The system prompt with instructions
        question: The current user question
        context: Retrieved document context
        conversation_context: Optional conversation history (formatted as "USER: ...\nASSISTANT: ...")

    Returns:
        List of LangChain messages
    """
    system_content = system_prompt

    # Build user message with optional conversation history
    if conversation_context:
        user_content = f"### CONVERSATION HISTORY\n{conversation_context}\n\n### CONTEXT\n{context}\n\n### USER QUESTION\n{question}"
    else:
        user_content = f"### CONTEXT\n{context}\n\n### USER QUESTION\n{question}"

    return [SystemMessage(content=system_content), HumanMessage(content=user_content)]