"""
Document Ingestor Module
Processes PDF, DOCX and plain-text (.txt/.md) files, extracting text and chunking for RAG
pipelines.

TO DO:
- Possible support for additional file types (e.g., HTML)
- Review rate limits / context window size. The current inference endpoint is limited.
- Different context strategies
> Currently when a file is added, the query just goes to the retriever.
This is fine, but the issue is a user is probably saying something like 'examine the file with regard to [something vector db related]
And the vector DB is getting the entire query string which probably isn't always good (i.e. adds noise).
So would be better to do analysis and possible query re-write with an LLM before sending to the retriever (where file == true). 

"""
import os
import logging
import re
from io import BytesIO
from typing import Tuple, Dict, Any

import PyPDF2
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def extract_text_from_pdf_bytes(file_content: bytes) -> Tuple[str, Dict[str, Any]]:
    """Extract text from PDF bytes (in memory)"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        metadata = {"total_pages": len(pdf_reader.pages)}

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += f"\n--- Page {page_num + 1} ---\n{page_text}"

        return text, metadata
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def extract_text_from_docx_bytes(file_content: bytes) -> Tuple[str, Dict[str, Any]]:
    """Extract text from DOCX bytes (in memory)"""
    try:
        doc = DocxDocument(BytesIO(file_content))
        text = ""
        metadata = {"total_paragraphs": 0}

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += f"{paragraph.text}\n"
                metadata["total_paragraphs"] += 1

        return text, metadata
    except Exception as e:
        logger.error(f"DOCX extraction error: {str(e)}")
        raise Exception(f"Failed to extract text from DOCX: {str(e)}")


def extract_text_from_plain_bytes(file_content: bytes) -> Tuple[str, Dict[str, Any]]:
    """
    Decode plain-text files (.txt / .md).

    Kept for the ChatUI route, which can upload a .txt/.md directly. Frontends that run
    their own extraction (OpenWebUI, LibreChat) send text on the chat request instead and
    reach `process_text()` without passing through here.
    
    UTF-8 first, latin-1 as a lossless byte-preserving fallback.
    """
    try:
        text = file_content.decode("utf-8")
    except UnicodeDecodeError:
        text = file_content.decode("latin-1")
        logger.warning("Plain-text file was not valid UTF-8; decoded as latin-1")
    return text, {"total_chars": len(text)}


def clean_and_chunk_text(text: str, config) -> str:
    """Clean text and split into chunks, returning formatted context"""
    # Basic text cleaning
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Get chunking parameters from config
    chunk_size = config.getint('ingestor', 'chunk_size', fallback=700)
    chunk_overlap = config.getint('ingestor', 'chunk_overlap', fallback=50)
    max_chunks = config.getint('ingestor', 'max_chunks', fallback=20)  # Limit chunks sent to LLM
    separators_str = config.get('ingestor', 'separators', fallback=r'\n\n,\n,. ,! ,? , ,')
    separators = [s.strip() for s in separators_str.split(',')]

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=False
    )

    chunks = text_splitter.split_text(text)

    # Limit the number of chunks to prevent context overflow
    chunks_to_use = chunks[:max_chunks]
    if len(chunks) > max_chunks:
        logger.warning(f"Document has {len(chunks)} chunks, limiting to first {max_chunks} chunks")

    # Create formatted context with chunk markers
    context_parts = []
    for i, chunk_text in enumerate(chunks_to_use):
        context_parts.append(f"[Chunk {i+1}]: {chunk_text}")

    return "\n\n".join(context_parts)


def process_document(file_content: bytes, filename: str, config) -> str:
    """
    Main document processing function - processes file and returns chunked context.

    Args:
        file_content: Raw bytes of the uploaded file
        filename: Name of the file (used to determine file type)
        config: Loaded in main.py

    Returns:
        Formatted chunked context string ready for RAG pipeline

    Raises:
        ValueError: If file type is unsupported
        Exception: If processing fails
    """
    try:
        # Extract text based on file type (in memory)
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension == '.pdf':
            text, extraction_metadata = extract_text_from_pdf_bytes(file_content)
        elif file_extension == '.docx':
            text, extraction_metadata = extract_text_from_docx_bytes(file_content)
        elif file_extension in ('.txt', '.md'):
            text, extraction_metadata = extract_text_from_plain_bytes(file_content)
        else:
            raise ValueError(
                f"Unsupported file type: {file_extension} (supported: .pdf, .docx, .txt, .md)"
            )

        # Clean and chunk text
        context = clean_and_chunk_text(text, config)

        logger.info(
            f"Successfully processed document {filename}: "
            f"{len(text)} characters, {extraction_metadata}"
        )

        return context

    except ValueError:
        logger.error(f"Document processing failed for {filename}: unsupported file type")
        raise
    except Exception as e:
        logger.error(f"Document processing failed for {filename}: {str(e)}")
        raise Exception(f"Processing failed: {str(e)}")


def process_text(text: str, filename: str, config) -> str:
    """
    Chunk already-extracted text into the same context format as process_document().

    Generic chat frontends extract text from files and pass raw text,
    so this function applies cleaning, chunking, and the max_chunks cap only.

    Args:
        text: Plain text of the attachment
        filename: Name of the file, used for logging and citation labelling only
        config: Loaded in main.py

    Returns:
        Formatted chunked context string ready for RAG pipeline
    """
    context = clean_and_chunk_text(text, config)
    logger.info(f"Successfully processed text attachment {filename}: {len(text)} characters")
    return context
