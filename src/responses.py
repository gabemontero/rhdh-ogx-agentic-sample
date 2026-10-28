import json
import os
from typing import Any

from ogx_client import OgxClient
from ogx_client import OpenAIResponseObject as ResponseObject
from ogx_client import (
    OpenAIResponseOutputMessageFileSearchToolCallResults as FileSearchToolCallResult,  # noqa
)
from openai import OpenAI

from src.constants import (
    DEFAULT_INGESTION_CONFIG_PATHS,
    DEFAULT_LLAMA_STACK_URL,
    DEFAULT_RAG_METADATA_FILE_PATHS,
    PIPELINE_CATEGORIES,
)
from src.exceptions import NoVectorStoresFoundError
from src.types import Pipeline, SourceTypes
from src.utils import list_all_vector_store_files, list_all_vector_stores, logger


class RAGService:
    """
    service for managing RAG vector stores and providing retrieval capabilities.
    """

    def __init__(
        self,
        pipelines: "list[Pipeline]",
        llama_stack_url: "str" = DEFAULT_LLAMA_STACK_URL,
        ingestion_config_path: "str | None" = None,
        file_metadata_path: "str | None" = None,
    ) -> "None":
        self.llama_stack_url = llama_stack_url
        self.client: "OgxClient | None" = None
        self.openai_client: "OpenAI | None" = None
        self.vector_store_map: "dict[str, list[str]]" = {}
        self.all_vector_store_ids: "list[str]" = []

        # source URL mapping: category -> GitHub base URL for PDFs
        self.source_url_map: "dict[str, str]" = {}
        self.ingestion_config: "dict[str, Any]" = {}

        # file metadata: file_id -> {original_filename, github_url, category}
        self.file_metadata: "dict[str, dict[str, str]]" = {}

        # Store pipelines for metadata regeneration
        self.pipelines = pipelines

        # load ingestion config if available
        if ingestion_config_path is None:
            for path in DEFAULT_INGESTION_CONFIG_PATHS:
                if os.path.exists(path):
                    ingestion_config_path = path
                    break

        # Always load source URL map from pipelines if provided
        if pipelines:
            self._load_source_url_map(pipelines)

        # load file metadata if available
        if file_metadata_path is None:
            for path in DEFAULT_RAG_METADATA_FILE_PATHS:
                if os.path.exists(path):
                    file_metadata_path = path
                    break

        if file_metadata_path and os.path.exists(file_metadata_path):
            self._load_file_metadata(file_metadata_path)

    def _load_file_metadata(self, metadata_path: "str") -> "None":
        """
        loads file metadata mapping from JSON.
        """
        try:
            with open(metadata_path, "r") as f:
                self.file_metadata = json.load(f)
            logger.info(
                f"RAG Service: Loaded metadata for "
                f"{len(self.file_metadata)} files from {metadata_path}"
            )
        except Exception as e:
            logger.warning(f"RAG Service: Could not load file metadata: {e}")

    def _validate_and_regenerate_metadata(self) -> "None":
        """
        Validates that loaded file_metadata matches actual files in vector stores.
        If there's a mismatch (stale metadata), regenerates metadata from vector stores.
        This handles the case where:
        - Pod restarts with existing vector stores but stale/missing metadata
        - Container has metadata from a different OGX instance
        """
        if not self.client:
            logger.warning(
                "RAG Service: Cannot validate metadata - client not initialized"
            )
            return

        # Collect all file_ids from vector stores
        actual_file_ids: "set[str]" = set()
        file_id_to_category: "dict[str, str]" = {}

        for category, vs_ids in self.vector_store_map.items():
            for vs_id in vs_ids:
                try:
                    files = list_all_vector_store_files(self.client, vs_id)
                    if not files:
                        continue
                    for file_info in files:
                        file_id = getattr(file_info, "id", None) or getattr(
                            file_info, "file_id", None
                        )
                        if file_id:
                            actual_file_ids.add(file_id)
                            file_id_to_category[file_id] = category
                except Exception as e:
                    logger.debug(f"Could not list files for vector store {vs_id}: {e}")

        if not actual_file_ids:
            logger.warning("RAG Service: No files found in vector stores")
            return

        # Check if current metadata has matching file_ids
        metadata_file_ids = set(self.file_metadata.keys())
        matching_ids = actual_file_ids & metadata_file_ids

        if matching_ids == actual_file_ids:
            logger.info(
                f"RAG Service: File metadata is valid ({len(matching_ids)} files match)"
            )
            return

        # Metadata is stale or missing - regenerate from vector stores
        logger.warning(
            f"RAG Service: File metadata mismatch detected! "
            f"Actual files: {len(actual_file_ids)}, "
            f"Matching metadata: {len(matching_ids)}. "
            f"Regenerating metadata from vector stores..."
        )

        self._regenerate_file_metadata(file_id_to_category)

    def _regenerate_file_metadata(
        self, file_id_to_category: "dict[str, str]"
    ) -> "None":
        """
        Regenerates file metadata by querying file details from OGX.
        Uses category information to construct GitHub URLs.

        This handles pod restarts where the local metadata file is lost but
        vector stores still exist in OGX. The original filename is
        preserved in OGX's file storage (since we upload with the
        original filename, just with .txt extension instead of .pdf).
        """
        if not self.client:
            return

        regenerated_metadata: "dict[str, dict[str, str]]" = {}

        for file_id, category in file_id_to_category.items():
            try:
                # Try to get file details from OGX
                file_info = self.client.files.retrieve(file_id)

                # Extract filename - try multiple attributes
                filename: "str | None" = None
                if hasattr(file_info, "filename") and file_info.filename:
                    filename = str(file_info.filename)
                elif hasattr(file_info, "name") and file_info.name:
                    filename = str(file_info.name)

                # Get base URL for this category
                base_url = self.source_url_map.get(category, "")

                if isinstance(filename, str):
                    # Clean up filename if it's a full path
                    if "/" in filename:
                        filename = os.path.basename(filename)

                    # Convert .txt back to .pdf (ingestion converts pdf->txt)
                    original_filename: str = filename
                    if filename.lower().endswith(".txt"):
                        original_filename = filename[:-4] + ".pdf"

                    # Skip if filename looks like a random temp file
                    # (e.g., tmp12345.txt from old ingestion code)
                    is_valid_filename = not (
                        filename.startswith("tmp")
                        and len(filename) > 10
                        and filename[3:].split(".")[0].isdigit()
                    )

                    if is_valid_filename:
                        github_url: str = (
                            f"{base_url}/{original_filename}" if base_url else ""
                        )

                        regenerated_metadata[file_id] = {
                            "original_filename": original_filename,
                            "github_url": github_url,
                            "category": category,
                        }
                        logger.info(
                            f"RAG Service: Regenerated metadata for "
                            f"{file_id} -> {original_filename}"
                        )
                    else:
                        # Random temp filename - use fallback
                        regenerated_metadata[file_id] = {
                            "original_filename": f"Document from {category}",
                            "github_url": base_url,
                            "category": category,
                        }
                        logger.debug(
                            f"RAG Service: Temp filename detected for {file_id}, "
                            f"using category fallback"
                        )
                else:
                    # Fallback: no filename available
                    regenerated_metadata[file_id] = {
                        "original_filename": f"Document from {category}",
                        "github_url": base_url,
                        "category": category,
                    }
                    logger.debug(
                        f"RAG Service: No filename for {file_id}, using fallback"
                    )

            except Exception as e:
                logger.debug(f"Could not retrieve file {file_id}: {e}")
                # Still add basic metadata so we have category info
                base_url = self.source_url_map.get(category, "")
                regenerated_metadata[file_id] = {
                    "original_filename": f"Document from {category}",
                    "github_url": base_url,
                    "category": category,
                }

        if regenerated_metadata:
            self.file_metadata = regenerated_metadata
            logger.info(
                f"RAG Service: Regenerated metadata for "
                f"{len(regenerated_metadata)} files"
            )

    def _load_source_url_map(self, pipelines: "list[Pipeline]") -> "dict[str, str]":
        """
        loads pipelines to map vector stores to source URLs.
        """
        for pipeline in pipelines:
            if not pipeline.enabled:
                continue

            if pipeline.source == SourceTypes.GITHUB:
                # Strip .git suffix if present (matches ingest.py behavior)
                url = pipeline.source_config.url.rstrip("/").removesuffix(".git")
                branch = pipeline.source_config.branch
                path = pipeline.source_config.path or ""

                # GitHub blob URL format: https://github.com/{org}/{repo}/blob/{branch}/{path}
                base_url = f"{url}/blob/{branch}/{path}".rstrip("/")

                category = (
                    pipeline.vector_store_name.split("-")[0]
                    if pipeline.vector_store_name
                    else ""
                )

                if category:
                    self.source_url_map[category] = base_url
                    logger.info(
                        f"RAG Service: Mapped category '{category}' "
                        f"to source URL: {base_url}"
                    )

        logger.info(
            f"RAG Service: Loaded source mappings for "
            f"{len(self.source_url_map)} categories"
        )
        return self.source_url_map

    def initialize(self) -> "bool":
        """
        initializes the OGX client and load vector stores.
        """
        try:
            # Initialize OpenAI client for structured output
            # (points to LlamaStack's OpenAI-compatible endpoint)
            openai_base_url = f"{self.llama_stack_url}/v1"
            self.openai_client = OpenAI(
                base_url=openai_base_url,
                api_key="not-needed",  # LlamaStack doesn't require API key
            )
            logger.info(
                f"RAG Service: OpenAI client initialized (base_url={openai_base_url})"
            )

            # Construct OpenAI first: initializing it after an OGX HTTPS request
            # can cause an SSL error on the next request over that connection.
            self.client = OgxClient(base_url=self.llama_stack_url)
            self.client.models.list()
            logger.info("RAG Service: OGX client initialized successfully")

            return self.load_vector_stores()
        except Exception as e:
            logger.error(f"RAG Service: Failed to initialize: {e}")
        return False

    def map_to_category(self, category: "str", vs_name: "str", vs_id: "str") -> "None":
        """
        maps to categories based on naming conventions from ingestion-config.yaml
        """
        if category not in vs_name:
            return

        if category not in self.vector_store_map:
            self.vector_store_map[category] = []
        self.vector_store_map[category].append(vs_id)
        logger.debug(f"RAG Service: Mapped '{vs_name}' -> '{category}'")

    def check_vector_stores_exist(self, pipelines: "list[Pipeline]") -> "bool":
        """
        checks if all required vector stores exist based on enabled pipelines.
        """
        if not self.client:
            logger.error("RAG Service: Client not initialized")
            return False

        expected_stores = set()
        for pipeline in pipelines:
            if pipeline.enabled:
                expected_stores.add(pipeline.vector_store_name.lower())

        if not expected_stores:
            logger.warning("RAG Service: No enabled pipelines found")
            return False

        vector_stores = list_all_vector_stores(self.client)
        existing_stores = set()
        for vs in vector_stores:
            vs_name = vs.name.lower() if vs.name else vs.id.lower()
            existing_stores.add(vs_name)

        missing_stores = expected_stores - existing_stores

        if missing_stores:
            logger.info(f"RAG Service: Missing vector stores: {missing_stores}")
            return False

        logger.info(
            f"RAG Service: All {len(expected_stores)} required vector stores exist"
        )
        return True

    def load_vector_stores(self) -> "bool":
        """
        loads vector stores from OGX and map them by category.

        raises:: NoVectorStoresFoundError if no vector stores are found.
        """
        if not self.client:
            logger.error("RAG Service: Client not initialized")
            return False

        vector_stores = list_all_vector_stores(self.client)

        if not vector_stores:
            err = "RAG Service: No vector stores found"
            logger.warning(err)
            raise NoVectorStoresFoundError(err)

        for vs in vector_stores:
            vs_name_lower = vs.name.lower() if vs.name else vs.id.lower()
            vs_id = vs.id
            self.all_vector_store_ids.append(vs_id)

            for category in PIPELINE_CATEGORIES:
                self.map_to_category(category, vs_name_lower, vs_id)

        logger.info(
            f"RAG Service: Loaded {len(self.all_vector_store_ids)} vector stores"
        )
        logger.info(
            f"RAG Service: Categories available: {list(self.vector_store_map.keys())}"
        )

        # Validate and regenerate file metadata if stale/missing
        # This handles the case where pod restarts with existing vector stores
        # but the local file_metadata.json is outdated or missing
        self._validate_and_regenerate_metadata()

        return len(self.all_vector_store_ids) > 0

    def get_vector_store_ids(self, category: "str | None" = None) -> "list[str]":
        """
        gets vector store IDs for a specific category or all stores.
        """
        if category:
            return self.vector_store_map.get(category, [])
        return self.all_vector_store_ids

    def get_file_search_tool(
        self, category: "str | None" = None
    ) -> "dict[str, Any] | None":
        """
        gets file_search tool configuration for OpenAI responses API.
        This tool format works with both OpenAI SDK and OGX's
        OpenAI-compatible endpoint.
        """
        vector_store_ids = self.get_vector_store_ids(category)

        if not vector_store_ids:
            logger.warning(f"RAG Service: No vector stores for category '{category}'")
            return None

        return {"type": "file_search", "vector_store_ids": vector_store_ids}

    def get_source_url(self, category: "str", filename: "str") -> "str":
        """
        constructs the GitHub URL for a source file.
        """
        base_url = self.source_url_map.get(category, "")
        if base_url and filename:
            return f"{base_url}/{filename}"
        return ""

    def _get_file_id(self, result: "FileSearchToolCallResult") -> "str | None":
        """
        helper to extract file_id from a result object.
        """
        if hasattr(result, "file_id"):
            return result.file_id
        elif hasattr(result, "id"):
            return result.id
        elif hasattr(result, "filename"):
            return result.filename
        elif hasattr(result, "name"):
            return result.name
        elif isinstance(result, dict):
            return (
                result.get("file_id")
                or result.get("id")
                or result.get("filename")
                or result.get("name")
            )
        return None

    def extract_sources_from_response(
        self, rag_response: "ResponseObject", category: "str"
    ) -> "list[dict[str, str]]":
        """
        extracts source document references from a RAG response.
        Uses the file_metadata mapping to get original filenames and GitHub URLs.
        Returns a list of dicts with 'filename', 'url', and optionally 'snippet'.
        """
        sources = []
        seen_files = set()

        for output_item in rag_response.output:
            if not (
                hasattr(output_item, "type") and output_item.type == "file_search_call"
            ):
                continue

            results = getattr(output_item, "results", None)
            if not results:
                continue

            for result in results:
                snippet = None
                file_id = self._get_file_id(result)

                if hasattr(result, "text") and result.text:
                    text = str(result.text)
                    snippet = text[:200] + "..." if len(text) > 200 else text
                elif isinstance(result, dict) and result.get("text"):
                    text = str(result.get("text", ""))
                    snippet = text[:200] + "..." if len(text) > 200 else text
                else:
                    snippet = None

                if not (file_id and file_id not in seen_files):
                    continue

                seen_files.add(file_id)

                if file_id in self.file_metadata:
                    metadata = self.file_metadata[file_id]
                    sources.append(
                        {
                            "filename": metadata.get("original_filename", file_id),
                            "url": metadata.get("github_url", ""),
                            "snippet": snippet or "",
                        }
                    )
                    logger.debug(
                        f"Found source: {metadata.get('original_filename')} "
                        f"-> {metadata.get('github_url')}"
                    )
                else:
                    # Metadata not found - use category base URL instead of
                    # constructing invalid URL with file_id
                    base_url = self.source_url_map.get(category, "")
                    # Link to category folder, not invalid file_id URL
                    sources.append(
                        {
                            "filename": f"Document from {category} knowledge base",
                            "url": base_url,
                            "snippet": snippet or "",
                        }
                    )
                    logger.debug(
                        f"No metadata for file_id {file_id}, using category URL"
                    )

        if not sources:
            sources = self._get_files_from_vector_store(category)

        return sources

    def _get_files_from_vector_store(self, category: "str") -> "list[dict[str, str]]":
        """
        get all files from the vector store for a category.
        """
        if not self.client:
            return []

        sources = []
        seen_files = set()
        vector_store_ids = self.get_vector_store_ids(category)

        for vs_id in vector_store_ids:
            try:
                files = list_all_vector_store_files(self.client, vs_id)
            except Exception as e:
                logger.debug(f"Could not list files for vector store {vs_id}: {e}")
                continue

            if not files:
                continue

            for file_info in files:
                file_id = getattr(file_info, "id", None) or getattr(
                    file_info, "file_id", None
                )

                if file_id and file_id not in seen_files:
                    seen_files.add(file_id)

                    # look up in file_metadata
                    if file_id in self.file_metadata:
                        metadata = self.file_metadata[file_id]
                        sources.append(
                            {
                                "filename": metadata.get("original_filename", file_id),
                                "url": metadata.get("github_url", ""),
                                "snippet": "",
                            }
                        )
                        # next iteration if found in metadata
                        continue

                    # Metadata not found - use category base URL
                    base_url = self.source_url_map.get(category, "")
                    sources.append(
                        {
                            "filename": f"Document from {category} knowledge base",
                            "url": base_url,  # Link to category folder
                            "snippet": "",
                        }
                    )

        return sources
