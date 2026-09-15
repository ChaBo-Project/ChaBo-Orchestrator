import re
import logging
logger = logging.getLogger(__name__)
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document


# ---------------------------------------------------------------------
# Core Processing Functions
# ---------------------------------------------------------------------
def parse_citations(response: str) -> List[int]:
    """Parse citation numbers from response text, handling both [1] and [1,2,3] formats"""
    # Match both single citations [1] and comma-separated citations [1,2,3,4,5]
    citation_pattern = r'\[([\d,\s]+)\]'
    matches = re.findall(citation_pattern, response)

    citation_numbers = set()
    for match in matches:
        # Split by comma and extract all numbers
        numbers = re.findall(r'\d+', match)
        citation_numbers.update(int(num) for num in numbers)

    citation_numbers = sorted(list(citation_numbers))
    logger.debug(f"Probable Citations found: {citation_numbers}")
    return citation_numbers

def extract_sources(processed_results: List[Dict[str, Any]], cited_numbers: List[int]) -> List[Dict[str, Any]]:
    """Extract sources that were cited in the response"""
    if not cited_numbers:
        return []
    
    cited_sources = []
    for citation_num in cited_numbers:
        source_index = citation_num - 1
        
        if 0 <= source_index < len(processed_results):
            source = processed_results[source_index].copy()  # Make copy to avoid modifying original
            source['_citation_number'] = citation_num  # Preserve original citation number
            cited_sources.append(source)
    #logger.debug(f"Extracted citations : {cited_sources}")
    
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
    """
    sources = []
    logger.info("creating cited sources list")
    logger.debug(f"Raw Cited sources: {cited_sources}")

    for result in cited_sources:
        # We access the original, full metadata dictionary
        all_meta = result.get('__all_metadata__', {})
        citation_num = result.get('_citation_number', 'N/A')
        
        # 1. Build Title using configured fields
        title_parts = []
        for field in title_metadata_fields:
            value = all_meta.get(field)
            if value is not None:
                title_parts.append(str(value))
        
        # Create a descriptive title
        title = " - ".join(title_parts) if title_parts else f"Source {citation_num}"

        # 2. Extract Link using configured field. No link (or a bare '#') stays None —
        # the renderer decides how to present a source that has no URL.
        link = all_meta.get(link_metadata_field)
        if not link or link == '#':
            link = None

        sources.append({
            "uri": link,
            "title": title
        })
    
    logger.debug(f"formatted cited sources :{sources}")

    return sources