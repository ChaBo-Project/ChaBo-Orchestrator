import re
import logging
logger = logging.getLogger(__name__)
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document


# ---------------------------------------------------------------------
# Core Processing Functions
# ---------------------------------------------------------------------
def parse_citations(response: str) -> List[int]:
    """
    Parse citation numbers from response text: single `[1]` (also `[1][3]`) and grouped
    `[1,2,3]` / `[1, 3]`. Returns sorted unique numbers.

    Keep in step with the citation format the prompt asks for (BASE_PROMPT, "CITATION
    FORMAT" in prompts.py). Ranges such as `[1-3]` are deliberately not read: the prompt
    never asks for them, and bracketed ranges in ordinary text ("a scale of [1-5]") would
    list sources that were never cited.
    """
    citation_pattern = r'\[([\d,\s]+)\]'
    matches = re.findall(citation_pattern, response)

    citation_numbers = set()
    for match in matches:
        citation_numbers.update(int(num) for num in re.findall(r'\d+', match))

    citation_numbers = sorted(citation_numbers)
    logger.debug(f"Probable Citations found: {citation_numbers}")
    return citation_numbers

# [generator] SOURCES_DISPLAY values: which sources are listed under the answer.
SOURCES_DISPLAY_MODES = ("cited", "all_retrieved")


def extract_sources(
    processed_results: List[Dict[str, Any]],
    cited_numbers: List[int],
    mode: str = "cited",
) -> List[Dict[str, Any]]:
    """
    Select the sources to list, in ascending citation order.

    `mode` (see SOURCES_DISPLAY_MODES): "cited" keeps only the sources the answer cites;
    "all_retrieved" keeps every retrieved document, cited or not.

    Each returned copy carries its citation number in `_citation_number` (its position in
    the context the LLM saw), so later steps can label it with the same `[n]` the answer
    uses. Cited numbers outside the retrieved results are skipped.
    """
    if mode not in SOURCES_DISPLAY_MODES:
        raise ValueError(f"Unknown sources display mode {mode!r}; expected one of {SOURCES_DISPLAY_MODES}")

    valid_numbers = sorted({
        number for number in cited_numbers
        if 1 <= number <= len(processed_results)
    })
    unmatched = sorted(set(cited_numbers) - set(valid_numbers))
    if unmatched:
        logger.warning(
            f"Answer cites {unmatched}, but only {len(processed_results)} documents were "
            f"retrieved; those citations have no source."
        )

    if mode == "all_retrieved":
        valid_numbers = list(range(1, len(processed_results) + 1))

    cited_sources = []
    for citation_num in valid_numbers:
        source = processed_results[citation_num - 1].copy()
        source['_citation_number'] = citation_num
        cited_sources.append(source)

    return cited_sources

def clean_citations(response: str) -> str:
    """Normalize all citation formats to [x] and remove unwanted sections"""
    
    # Remove References/Sources/Bibliography sections
    ref_patterns = [
        r'\n\s*#+\s*References?\s*:?.*$',
        r'\n\s*#+\s*Sources?\s*:?.*$',
        r'\n\s*#+\s*Bibliography\s*:?.*$',
        r'\n\s*References?\s*:.*$',
        r'\n\s*Sources?\s*:.*$',
        r'\n\s*Bibliography\s*:.*$',
    ]
    for pattern in ref_patterns:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.DOTALL)
    
    # Fix (Document X, Page Y, Year Z) -> [X]
    response = re.sub(
        r'\(Document\s+(\d+)(?:,\s*Page\s+\d+)?(?:,\s*(?:Year\s+)?\d+)?\)',
        r'[\1]',
        response,
        flags=re.IGNORECASE
    )
    
    # Fix [Document X, Page Y, Year Z] -> [X]
    response = re.sub(
        r'\[Document\s+(\d+)(?:[^\]]*)\]', 
        r'[\1]', 
        response, 
        flags=re.IGNORECASE
    )
    
    # Fix [Document X: filename, Page Y, Year Z] -> [X]
    response = re.sub(
        r'\[Document\s+(\d+):[^\]]+\]',
        r'[\1]',
        response,
        flags=re.IGNORECASE
    )
    
    # Fix [X.Y.Z] style (section numbers) -> [X]
    response = re.sub(
        r'\[(\d+)\.[\d\.]+\]', 
        r'[\1]', 
        response
    )
    
    # Fix (Document X) -> [X]
    response = re.sub(
        r'\(Document\s+(\d+)\)', 
        r'[\1]', 
        response, 
        flags=re.IGNORECASE
    )
    
    # Fix "Document X, Page Y, Year Z" (no brackets) -> [X]
    response = re.sub(
        r'Document\s+(\d+)(?:,\s*Page\s+\d+)?(?:,\s*(?:Year\s+)?\d+)?(?=\s|[,.])',
        r'[\1]',
        response,
        flags=re.IGNORECASE
    )
    
    # Fix "Document X states/says/mentions" -> [X]
    response = re.sub(
        r'Document\s+(\d+)\s+(?:states|says|mentions|reports|indicates|notes|shows)',
        r'[\1]',
        response,
        flags=re.IGNORECASE
    )
    
    # Clean up any double citations [[1]] -> [1]
    response = re.sub(r'\[\[(\d+)\]\]', r'[\1]', response)
    
    # Clean up multiple spaces
    response = re.sub(r'\s+', ' ', response)
    
    return response.strip()

def process_context(
    context: List[Document],
    metadata_fields_to_include: Optional[List[str]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Processes LangChain Documents, extracts content and selected metadata,
    and returns the formatted context string and the processed list of results.
    
    Args:
        context: A list of LangChain Document objects from the retriever.
        metadata_fields_to_include: Optional list of metadata keys (e.g., ['source', 'page']) 
                                    to include in the formatted context string sent to the LLM.
                                    
    Returns:
        A tuple: (formatted_context_string, processed_results_list)
    """
    logger.debug(f"Context Processing: \n Context: {context} \n Metadata_fileds: {metadata_fields_to_include}")
    logger.info(f"Context Processing with Metadata_fileds_to_include: {metadata_fields_to_include}")
    
    # 1. Input Validation
    if not isinstance(context, list) or not all(isinstance(doc, Document) for doc in context):
        # Raise a specific error if input is not what's expected
        raise ValueError("Context must be a list of LangChain Document objects.")
    if not context:
        return "", []

    processed_results = []
    metadata_fields_to_include = metadata_fields_to_include or []

    # 2. Standardize Structure and Build Context String
    context_parts = []
    
    for i, doc in enumerate(context, 1):
        # The primary dictionary that holds all info for this document
        doc_info = {
            'answer': doc.page_content,
            '__all_metadata__': doc.metadata, # Store all metadata for citation linking later
            '_citation_number_key': i        # Store the citation number
        }
        #logger.debug(f"DocInfo of {i} context: {doc_info}")
        
        # Extract selected metadata fields for the prompt string
        metadata_str_parts = []
        for field in metadata_fields_to_include:
            value = doc.metadata.get(field)
            if value is not None:
                # Store the value in the doc_info dict and format for the prompt string
                doc_info[field] = value
                
                # Format the field for readability in the prompt
                field_name = field.replace('_', ' ').title()
                metadata_str_parts.append(f"{field_name}: {value}")
        
        # Build the document string
        if metadata_str_parts:
            metadata_line = " | ".join(metadata_str_parts)
            # Example output: [1] (Type: decision, Meeting ID: 123)
            context_str = f"[{i}] **This is Metadata** \n ({metadata_line})\n **Contextual Text** \n {doc.page_content}"
        else:
            context_str = f"[{i}]\n{doc.page_content}"
        
        logger.debug(f" Updated Context {i}: {context_str}")
        context_parts.append(context_str)
        processed_results.append(doc_info) # Collect the standardized dict for later use
    formatted_context = "\n---\n".join(context_parts)
    
    return formatted_context, processed_results

def create_sources_list(
    cited_sources: List[Dict[str, Any]],
    title_metadata_fields: List[str],
    link_metadata_field: str
    ) -> List[Dict[str, Optional[str]]]:
    """
    Create the cited-sources list, using configuration for title and link fields.

    Frontend-agnostic in that it won't cause an error - but no guarantees for UIs other than Chabo-ChatUI

    Args:
        cited_sources: List of standardized dictionaries that were cited.
        title_metadata_fields: List of metadata keys (e.g., ['document_type', 'decision_number'])
                               to use to build the source title.
        link_metadata_field: The single metadata key (e.g., 'document_url') to use for the source link (URL).

    Each entry is `{"number", "uri", "title"}`; `number` is the source's citation number (the
    `[n]` the answer uses for it), or None if the input carries none.
    """
    sources = []
    logger.info("creating cited sources list")
    logger.debug(f"Raw Cited sources: {cited_sources}")

    for result in cited_sources:
        # We access the original, full metadata dictionary
        all_meta = result.get('__all_metadata__', {})
        citation_num = result.get('_citation_number')
        
        # 1. Build Title using configured fields
        title_parts = []
        for field in title_metadata_fields:
            value = all_meta.get(field)
            if value is not None:
                title_parts.append(str(value))
        
        # Create a descriptive title
        title = " - ".join(title_parts) if title_parts else (
            f"Source {citation_num}" if citation_num is not None else "Source"
        )

        # 2. Extract Link using configured field. No link (or a bare '#') stays None —
        # the renderer decides how to present a source that has no URL.
        link = all_meta.get(link_metadata_field)
        if not link or link == '#':
            link = None

        sources.append({
            "number": citation_num,
            "uri": link,
            "title": title
        })
    
    logger.debug(f"formatted cited sources :{sources}")

    return sources