import logging
# Set up logger
logger = logging.getLogger(__name__)
from typing import List, Dict, Any, Union, AsyncGenerator, Optional
import asyncio

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage
from ..utils import getconfig, get_config_value
from ..llm import LLMClient, build_llm_client
from .prompts import system_prompt, build_messages
from .sources import (
    process_context, parse_citations, 
    extract_sources, create_sources_list, clean_citations
)



class Generator:
    """
    A generic RAG answer generation component that supports multiple LLM providers 
    and reads configuration from kwargs, environment variables, or params.cfg.
    """
    
    # 1. Define the configuration map for RAG/metadata parameters.
    CONFIG_MAP = {
        "context_metadata_fields":  ("generator", "CONTEXT_META_FIELDS", "GENERATOR_CONTEXT_META_FIELDS", "source,page"),
        "title_metadata_fields":    ("generator", "TITLE_META_FIELDS", "GENERATOR_TITLE_META_FIELDS", "source,document_id"),
        "link_metadata_field":      ("generator", "LINK_META_FIELD", "GENERATOR_LINK_META_FIELD", "url")
    }

    def __init__(self, config_path: str = "params.cfg", llm_client: Optional[LLMClient] = None, **kwargs):
        logger.info("Initializing Generator component with config precedence...")

        # 2. Load Configuration
        config_file = getconfig(config_path)

        # Resolve config (provider/model handled by the LLMClient).
        resolved_config = {}
        for key, params in self.CONFIG_MAP.items():
            section, option, env_var = params[:3]
            fallback = params[3] if len(params) > 3 else None

            # 1. Prioritize kwargs (explicitly passed to the constructor)
            if key in kwargs:
                value = kwargs[key]
                logger.debug(f"Config '{key}' loaded from kwargs.")
            else:
                # 2. Use the unified utility to check ENV then config file
                value = get_config_value(config_file, section, option, env_var, fallback)

            # Type conversion
            if key in ['context_metadata_fields', 'title_metadata_fields']:
                # For list fields, parse the comma-separated string unless it's already a list from kwargs
                if isinstance(value, str):
                    value = [item.strip() for item in value.split(',') if item.strip()]

            resolved_config[key] = value

        # 3. Assign resolved values to instance attributes (RAG/Metadata)
        self.context_metadata_fields = resolved_config['context_metadata_fields']
        self.title_metadata_fields = resolved_config['title_metadata_fields']
        self.link_metadata_field = resolved_config['link_metadata_field']

        # 4. Inference layer
        # Use LLMClient or build the default generation client.
        self.llm_client = llm_client if llm_client is not None else build_llm_client(config_file, "generation")

        logger.info(f"Generator initialized with provider: {self.llm_client.provider}, model: {self.llm_client.model}")
        logger.debug(f"Metadata Config: Context={self.context_metadata_fields}, Title={self.title_metadata_fields}")


    async def _call_llm(self, messages: list) -> str:
        """Provider-agnostic LLM call (non-streaming). Delegates to the LLMClient."""
        return await self.llm_client.ainvoke(messages)

    async def _call_llm_streaming(self, messages: list) -> AsyncGenerator[str, None]:
        """Provider-agnostic streaming LLM call. Delegates to the LLMClient."""
        async for chunk in self.llm_client.astream(messages):
            yield chunk

    # Convenience pass-throughs so a Generator can stand in wherever an LLMClient is expected.
    async def ainvoke(self, messages: list) -> str:
        return await self.llm_client.ainvoke(messages)

    async def astream(self, messages: list) -> AsyncGenerator[str, None]:
        async for chunk in self.llm_client.astream(messages):
            yield chunk
    
    # # --- Response Post-Processing Method (Centralized Logic) ---

    # def _process_final_response(self, answer: str, processed_results: List[Dict[str, Any]], chatui_format: bool) -> Union[str, Dict[str, Any]]:
    #     """Handles final citation cleaning and output formatting."""
    #     # Clean citations
    #     answer = clean_citations(answer)

    #     if chatui_format:
    #         result = {"answer": answer}
    #         if processed_results:
    #             # Only include sources if processing successful
    #             cited_numbers = parse_citations(answer)
    #             cited_sources = extract_sources(processed_results, cited_numbers)
    #             result["sources"] = create_sources_list(cited_sources)
    #         return result
    #     else:
    #         return answer

    # # --- Main Generation Methods (Public Interface) ---

    async def generate(self, query: str, context: Union[str, List[Dict[str, Any]], None], chatui_format: bool = False, conversation_context: str = None) -> Union[str, Dict[str, Any]]:
        """Generate an answer to a query using provided context (non-streaming)"""
        if not query.strip():
            error_msg = "Query cannot be empty"
            return {"error": error_msg} if chatui_format else f"Error: {error_msg}"
        logger.info(f"Generating answer for query: {query[:50]}")

        try:
            # 1. Process Context
            formatted_context, processed_results = process_context(context,
               metadata_fields_to_include=self.context_metadata_fields)

            # 2. Build Messages (with system prompt and optional conversation history)
            messages = build_messages(system_prompt, query, formatted_context, conversation_context)



            # 3. Call LLM
            answer = await self._call_llm(messages)
            
            if chatui_format:
                result = {"answer": answer}
                if processed_results:
                    cited_numbers = parse_citations(answer)
                    cited_sources = extract_sources(processed_results, cited_numbers)
                    result["sources"] = create_sources_list(cited_sources, 
                                        title_metadata_fields=self.title_metadata_fields,
                                        link_metadata_field=self.link_metadata_field)
                return result
            else:
                return answer       

        except Exception as e:
            logger.exception("Generation failed")
            error_msg = str(e)
            return {"error": error_msg} if chatui_format else f"Error: {error_msg}"

    async def generate_streaming(self, query: str, context: Union[str, List[Dict[str, Any]], None], chatui_format: bool = False, conversation_context: str = None) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """Generate a streaming answer to a query using provided context through RAG"""
        if not query.strip():
            error_msg = "Query cannot be empty"
            # Return error in the streaming format
            if chatui_format:
                yield {"event": "error", "data": {"error": error_msg}}
            else:
                yield f"Error: {error_msg}"
            return
        logger.info(f"Generating streaming answer for query: {query[:50]}")
        if conversation_context:
            logger.info(f"Using conversation context: {len(conversation_context)} chars")

        try:
            # 1. Process Context
            formatted_context, processed_results = process_context(context,
                            metadata_fields_to_include =self.context_metadata_fields)

            # 2. Build Messages (with system prompt and optional conversation history)
            messages = build_messages(system_prompt, query, formatted_context, conversation_context)

            # 3. Stream the response and accumulate for citation parsing
            accumulated_response = ""
            async for chunk in self._call_llm_streaming(messages):
                accumulated_response += chunk
                
                # Yield the raw text chunks immediately
                if chatui_format:
                    yield {"event": "data", "data": chunk}
                else:
                    yield chunk

            # 4. Final Post-Processing (after stream is complete)
            cleaned_response = clean_citations(accumulated_response)

            # Send final answer with sources at the end if in ChatUI format
            if chatui_format:
                final_answer_data = {"text": cleaned_response}

                # Add sources if available
                if processed_results:
                    cited_numbers = parse_citations(cleaned_response)
                    cited_sources = extract_sources(processed_results, cited_numbers)
                    sources = create_sources_list(cited_sources,
                                title_metadata_fields=self.title_metadata_fields,
                                link_metadata_field=self.link_metadata_field)
                    final_answer_data["webSources"] = sources
                    logger.info(f"Final answer webSources: {sources}")

                yield {"event": "final_answer", "data": final_answer_data}

                # Send END event for ChatUI format
                yield {"event": "end", "data": {}}

        except Exception as e:
            logger.exception("Streaming generation failed")
            error_msg = str(e)
            if chatui_format:
                yield {"event": "error", "data": {"error": error_msg}}
            else:
                yield f"Error: {error_msg}"

