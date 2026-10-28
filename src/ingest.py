import asyncio
import json
import os
import sys
import tempfile
import time
from typing import Any, cast

import requests
import yaml
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from github import Auth, Github
from github.ContentFile import ContentFile
from github.Repository import Repository
from ogx_client import OgxClient, OpenAIAttachFileRequest
from ogx_client import (
    OpenAICreateVectorStoreRequestWithExtraBody as CreateVectorStoreRequest,
)
from ogx_client import OpenAIFileObject as File

from src.constants import (
    DEFAULT_CHUNK_SIZE_IN_TOKENS,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HTTP_REQUEST_TIMEOUT,
    DEFAULT_INGESTION_MODE,
    DEFAULT_OGX_RETRY_DELAY,
    DEFAULT_OGX_WAITING_RETRIES,
)
from src.exceptions import IngestionPipelineError
from src.types import (
    RAW_PIPELINES_TYPE,
    Pipeline,
    SourceConfig,
    SourceTypes,
    VectorDBConfig,
)
from src.utils import (
    check_ogx_availability,
    clean_text,
    list_all_vector_store_files,
    list_all_vector_stores,
    logger,
)


class IngestionService:
    """
    Service for ingesting documents into vector databases.
    """

    def __init__(self, config_path: "str") -> "None":
        self.vector_db_ids = None

        with open(config_path, "r") as f:
            _config: "dict[str, Any]" = yaml.safe_load(f)

            # validate ingest configuration
            if not self._config_is_valid(_config):
                logger.error("Invalid ingestion config file")
                sys.exit(1)

        self.config_path = config_path

        # OGX setup - prefer env var over config file
        self.llama_stack_url: "str" = os.environ.get(
            "LLAMA_STACK_URL", _config["llamastack"]["base_url"]
        )
        self.client = self._initialize_ogx_client()
        self.vector_store_ids: "list[str]" = []

        # Environment overrides let the app use an existing server's models.
        _embedding_dimension = int(
            os.environ.get(
                "EMBEDDING_DIMENSION",
                _config["vector_db"].get("embedding_dimension")
                or DEFAULT_EMBEDDING_DIMENSION,
            )
        )
        _embedding_model = os.environ.get(
            "EMBEDDING_MODEL",
            _config["vector_db"].get("embedding_model") or DEFAULT_EMBEDDING_MODEL,
        )
        self.vector_db_provider_id = os.environ.get(
            "VECTOR_DB_PROVIDER_ID", _config["vector_db"].get("provider_id", "")
        )
        # Only explicit overrides belong in the server request. In particular,
        # a model override must not inherit another model's YAML dimension.
        self.vector_store_options: dict[str, Any] = {}
        if os.environ.get("EMBEDDING_MODEL"):
            self.vector_store_options["embedding_model"] = _embedding_model
        if os.environ.get("EMBEDDING_DIMENSION"):
            self.vector_store_options["embedding_dimension"] = str(_embedding_dimension)
        if self.vector_db_provider_id:
            self.vector_store_options["provider_id"] = self.vector_db_provider_id
        _chunk_size_in_tokens = (
            _config["vector_db"].get("chunk_size_in_tokens")
            if _config["vector_db"].get("chunk_size_in_tokens")
            else DEFAULT_CHUNK_SIZE_IN_TOKENS
        )
        # ingestion_mode: async vs sync
        self.ingestion_mode = (
            _config["vector_db"].get("ingestion_mode", DEFAULT_INGESTION_MODE).lower()
        )
        if self.ingestion_mode not in ("async", "sync"):
            logger.warning(
                f"Invalid ingestion_mode '{self.ingestion_mode}', "
                f"defaulting to '{DEFAULT_INGESTION_MODE}'"
            )
            self.ingestion_mode = DEFAULT_INGESTION_MODE
        logger.info(f"Ingestion mode: {self.ingestion_mode}")
        self.vector_db_config = VectorDBConfig(
            embedding_model=_embedding_model,
            embedding_dimension=_embedding_dimension,
            chunk_size_in_tokens=_chunk_size_in_tokens,
        )
        self.file_metadata = {}

        # File metadata output path - prefer env var for writable location in containers
        self.file_metadata_path = os.environ.get(
            "RAG_FILE_METADATA", "rag_file_metadata.json"
        )

        # Document converter setup
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self.chunker = HybridChunker()

        # GitHub client setup
        gh_token = os.getenv("GITHUB_TOKEN")
        self.gh_client = Github(auth=Auth.Token(gh_token)) if gh_token else Github()
        logger.debug("Github client initialized")

        # Pipelines setup
        self.pipelines: "list[Pipeline]" = self._parse_pipelines(_config["pipelines"])

    def _parse_pipelines(self, raw_pipelines: "RAW_PIPELINES_TYPE") -> "list[Pipeline]":
        """
        parses raw pipeline configurations into Pipeline dataclass instances.
        """
        logger.debug("Parsing pipeline configurations...")
        pipelines: "list[Pipeline]" = []

        for p_title in raw_pipelines:
            logger.debug(f"Parsing pipeline {p_title}...")
            _pipeline_config: "dict[str, Any]" = raw_pipelines[p_title]
            logger.debug(f"Pipeline {p_title} type: {_pipeline_config['source']}")

            if _pipeline_config["source"] == SourceTypes.GITHUB:
                config = _pipeline_config.get("config")
                if not isinstance(config, dict):
                    logger.error(
                        f"Pipeline '{p_title}' has invalid config: expected dict"
                    )
                    continue
                url_value = config.get("url", "")
                url_str = url_value if isinstance(url_value, str) else ""
                branch_value = config.get("branch", "main")
                branch_str = branch_value if isinstance(branch_value, str) else "main"
                path_value = config.get("path", "")
                path_str = path_value if isinstance(path_value, str) else ""
                source_config = SourceConfig(
                    url=url_str,
                    branch=branch_str,
                    path=path_str,
                    urls=None,
                )
            elif _pipeline_config["source"] == SourceTypes.URL:
                urls_value = _pipeline_config.get("urls", [])
                urls_list: "list[str]" = (
                    urls_value if isinstance(urls_value, list) else []
                )
                source_config = SourceConfig(
                    url="",
                    branch="",
                    path="",
                    urls=urls_list,
                )
            else:
                logger.error(
                    f"Unknown source type '{_pipeline_config['source']}' "
                    f"in pipeline '{_pipeline_config['name']}'"
                )
                continue

            enabled_value = _pipeline_config.get("enabled", False)
            enabled_bool = (
                bool(enabled_value)
                if not isinstance(enabled_value, bool)
                else enabled_value
            )

            source_value = _pipeline_config.get("source")
            if not isinstance(source_value, str):
                logger.error(f"Pipeline '{p_title}' has invalid source type")
                continue

            # build the Pipeline object for each different pipeline
            # found inside the ingestion_config.yaml
            pipeline = Pipeline(
                name=str(_pipeline_config["name"]),
                enabled=enabled_bool,
                version=str(_pipeline_config["version"]),
                vector_store_name=f"{_pipeline_config['vector_store_name']}",
                source=source_value,
                source_config=source_config,
            )
            pipelines.append(pipeline)
            logger.debug(f"Pipeline {p_title} sucessfully parsed!")

        return pipelines

    def _config_is_valid(self, raw_config: "dict[str, Any]") -> "bool":
        """
        loads and validates configuration.
        """
        try:
            logger.debug("Validating ingestion config...")
            assert "llamastack" in raw_config
            assert "base_url" in raw_config["llamastack"]
            assert "vector_db" in raw_config
            assert "embedding_model" in raw_config["vector_db"]
            assert "pipelines" in raw_config
        except AssertionError as e:
            logger.error(f"Configuration validation error: {e}")
            return False

        logger.debug("Ingestion config validated successfully!")
        return True

    def _initialize_ogx_client(
        self,
        max_retries: "int" = DEFAULT_OGX_WAITING_RETRIES,
        retry_delay: "int" = DEFAULT_OGX_RETRY_DELAY,
    ) -> "OgxClient":
        logger.debug(
            f"Connecting OGX Client to Server at {self.llama_stack_url}..."
        )

        for attempt in range(max_retries):
            result = check_ogx_availability(self.llama_stack_url)

            if result["connected"]:
                logger.debug("OGX Client connected successfully!")
                return OgxClient(base_url=self.llama_stack_url)

            if attempt < max_retries - 1:
                logger.info(
                    f"Attempt {attempt + 1}/{max_retries}: "
                    f"OGX not ready yet. "
                    f"Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to connect to OGX "
                    f"after {max_retries} attempts: {result['error_message']}"
                )

        logger.error("Failed to connect to OGX. Exiting.")
        sys.exit(1)

    def _create_local_file(
        self, content: "ContentFile", path: "str", download_dir: "str"
    ) -> "str":
        """
        creates a local file with the content of the
        given github file
        """
        logger.debug(f"Creating local file for {content.path}...")
        file_content = content.decoded_content

        relative_path = content.path
        if path and relative_path.startswith(path):
            relative_path = relative_path[len(path) :].lstrip("/")

        local_file_path = os.path.join(download_dir, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        with open(local_file_path, "wb") as f:
            f.write(file_content)

        logger.debug(f"Local file created at {content.path}")
        return local_file_path

    def _fetch_github_dir_contents(
        self,
        repo: "Repository",
        contents: "list[ContentFile]",
        branch: "str",
        path: "str",
        download_dir: "str",
    ) -> "list[str]":
        """
        fetches recursively contents from GitHub
        """
        pdf_files = []
        for content in contents:
            if content.type == "dir":
                # recursive fetch of directory contents
                try:
                    sub_contents = repo.get_contents(content.path, ref=branch)
                    if isinstance(sub_contents, list):
                        contents_list: "list[ContentFile]" = []
                        for c in sub_contents:
                            if isinstance(c, ContentFile):
                                contents_list.append(c)
                            elif hasattr(c, "type") and hasattr(c, "name"):
                                contents_list.append(cast(ContentFile, c))
                        pdf_files.extend(
                            self._fetch_github_dir_contents(
                                repo, contents_list, branch, path, download_dir
                            )
                        )
                    elif isinstance(sub_contents, ContentFile):
                        # Single file, not a directory
                        if sub_contents.name.lower().endswith(".pdf"):
                            local_file_path = self._create_local_file(
                                sub_contents, path, download_dir
                            )
                            pdf_files.append(local_file_path)
                except Exception as e:
                    logger.error(f"Error accessing directory {content.path}: {e}")
            # handle pdf files
            elif content.name.lower().endswith(".pdf"):
                try:
                    local_file_path = self._create_local_file(
                        content, path, download_dir
                    )
                    pdf_files.append(local_file_path)
                except Exception as e:
                    logger.error(f"Error downloading {content.path}: {e}")

        return pdf_files

    def _get_github_repo(self, url: "str") -> "Repository | None":
        """
        fetches PyGithub repo from GitHub URL.

        URL format: https://github.com/owner/repo or https://github.com/owner/repo.git
        """
        logger.debug(f"Getting GitHub repository from URL: {url}")
        url_parts = url.rstrip("/").rstrip(".git").split("github.com/")
        if len(url_parts) < 2:
            return None

        repo_path = url_parts[1]
        repo_parts = repo_path.split("/")
        if len(repo_parts) < 2:
            return None

        owner = repo_parts[0]
        repo_name = repo_parts[1]

        repo_title = f"{owner}/{repo_name}"

        if not repo_title:
            logger.error(f"Invalid GitHub URL: {url}")
            return None

        # Initialize GitHub client
        try:
            repo = self.gh_client.get_repo(f"{repo_title}")
            logger.debug(f"Accessed repository: {repo_title}")
            return repo
        except Exception as e:
            logger.error(f"Failed to access repository {repo_title}: {e}")
            return None

    def fetch_from_github(
        self, url: "str", path: "str", branch: "str", temp_dir: "str"
    ) -> "list[str]":
        """
        fetches documents from a GitHub repository using PyGithub API.
        """
        logger.debug(f"Fetching from GitHub: {url} (branch: {branch}, path: {path})")
        pdf_files: "list[str]" = []

        repo = self._get_github_repo(url)
        if repo is None:
            logger.error(f"Could not access repository at {url}")
            return []

        # create download directory
        download_dir = os.path.join(temp_dir, "repo")
        os.makedirs(download_dir, exist_ok=True)

        # NOTE: This is done synchronously to avoid rate
        # limits from Github
        try:
            logger.debug(f"Accessing contents at path: {path if path else '/'}")
            contents = repo.get_contents(path if path else "", ref=branch)

            if isinstance(contents, list):
                contents_list: "list[ContentFile]" = []
                for c in contents:
                    if isinstance(c, ContentFile):
                        contents_list.append(c)
                    elif hasattr(c, "type") and hasattr(c, "name"):
                        contents_list.append(cast(ContentFile, c))
                pdf_files = self._fetch_github_dir_contents(
                    repo, contents_list, branch, path, download_dir
                )
            elif contents.name.lower().endswith(".pdf"):
                local_file_path = self._create_local_file(contents, path, download_dir)
                pdf_files.append(local_file_path)
                logger.debug(f"Downloaded content: {contents.path}")

        except Exception as e:
            logger.error(f"Error fetching contents from path '{path}': {e}")
            return []

        logger.debug(f"Found {len(pdf_files)} PDF files")
        return pdf_files

    def _download_single_url(self, url: str, download_dir: str) -> str | None:
        """
        downloads a single URL synchronously
        """
        filename = os.path.basename(url.split("?")[0])
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        file_path = os.path.join(download_dir, filename)

        logger.debug(f"Downloading document from url: {url}")
        try:
            response = requests.get(url, timeout=DEFAULT_HTTP_REQUEST_TIMEOUT)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            return file_path

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None

    async def fetch_from_url(self, urls: "list[str]", temp_dir: "str") -> "list[str]":
        """
        fetches documents from direct URLs concurrently.
        """
        logger.debug(f"Fetching {len(urls)} documents from URLs concurrently")

        download_dir = os.path.join(temp_dir, "url_files")
        os.makedirs(download_dir, exist_ok=True)

        # download all URLs concurrently using thread pool
        results = await asyncio.gather(
            *[
                asyncio.to_thread(self._download_single_url, url, download_dir)
                for url in urls
            ]
        )

        # filter out None values (failed downloads)
        pdf_files = [path for path in results if path is not None]

        logger.debug(f"Downloaded {len(pdf_files)} PDF files from URLs concurrently")
        return pdf_files

    def _process_single_document(
        self, file_path: str, github_base_url: str, category: str
    ) -> File | None:
        """
        processes a single PDF document.
        Preserves original filename when uploading to OGX for
        better metadata recovery after pod restarts.
        """
        try:
            original_filename = os.path.basename(file_path)
            logger.info(f"Processing document: {original_filename}")
            result = self.converter.convert(file_path)

            # clean text for special characters
            markdown_text = result.document.export_to_markdown()
            cleaned_text = clean_text(markdown_text)

            # create temp file with original filename (not random temp name)
            # this preserves the filename in OGX for metadata recovery
            temp_dir = tempfile.mkdtemp()
            # replace .pdf extension with .txt for the processed content
            processed_filename = original_filename
            if processed_filename.lower().endswith(".pdf"):
                processed_filename = processed_filename[:-4] + ".txt"
            tmp_file_path = os.path.join(temp_dir, processed_filename)

            try:
                # create file in OGX
                with open(tmp_file_path, "w", encoding="utf-8") as tmp_file:
                    tmp_file.write(cleaned_text)

                file_create_response = self.client.files.create(
                    file=str(tmp_file_path), purpose="assistants"
                )

                file_id = file_create_response.id
                github_url = (
                    f"{github_base_url}/{original_filename}" if github_base_url else ""
                )

                self.file_metadata[file_id] = {
                    "original_filename": original_filename,
                    "github_url": github_url,
                    "category": category,
                }
                logger.info(f"Mapped file_id '{file_id}' -> '{original_filename}'")
                return file_create_response
            finally:
                # clean files
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    async def process_documents(
        self, pdf_files: "list[str]", github_base_url="", category=""
    ) -> "list[File]":
        """
        processes PDF files into chunks using docling concurrently.
        """
        logger.info(
            f"Processing {len(pdf_files)} documents with docling concurrently..."
        )

        # process all documents concurrently using thread pool
        results = await asyncio.gather(
            *[
                asyncio.to_thread(
                    self._process_single_document, file_path, github_base_url, category
                )
                for file_path in pdf_files
            ]
        )

        # filter out None values (failed processing)
        documents = [doc for doc in results if doc is not None]

        logger.info(f"Total documents processed: {len(documents)}")
        return documents

    def _insert_single_document(
        self, vector_store_id: str, vector_store_name: str, doc: File
    ) -> bool:
        """
        inserts a single document into vector store and checks status
        """
        try:
            file_ingest_response = self.client.vector_stores_files.create(
                vector_store_id=vector_store_id,
                open_ai_attach_file_request=OpenAIAttachFileRequest(file_id=doc.id),
            )

            # check the actual status of the insertion
            if hasattr(file_ingest_response, "status"):
                if file_ingest_response.status == "failed":
                    error_msg = "Unknown error"
                    if (
                        hasattr(file_ingest_response, "last_error")
                        and file_ingest_response.last_error
                    ):
                        error_msg = file_ingest_response.last_error.message
                    logger.error(
                        f"✗ Document insertion failed for "
                        f"'{vector_store_name}': {error_msg}"
                    )
                    return False
                elif file_ingest_response.status in ("completed", "in_progress"):
                    logger.info(
                        f"✓ Document inserted into '{vector_store_name}' "
                        f"(status: {file_ingest_response.status})"
                    )
                    return True
                else:
                    logger.warning(
                        f"⚠ Document has unexpected status "
                        f"'{file_ingest_response.status}' for '{vector_store_name}'"
                    )
                    return False
            else:
                logger.info(
                    f"✓ Document inserted into '{vector_store_name}' (no status field)"
                )
                return True
        except Exception as e:
            logger.error(
                f"Error inserting document {doc.id} into '{vector_store_name}': {e}"
            )
            return False

    async def create_vector_db(
        self, vector_store_name: "str", documents: "list[File]"
    ) -> "bool":
        """
        creates vector database and inserts documents.
        uses async (concurrent) or sync (sequential) mode based on config.

        all (LlamaStack) client calls are wrapped in thread (asyncio.to_thread)
        to avoid blocking the concurrent execution.
        """
        if not documents:
            logger.warning(f"No documents to insert for {vector_store_name}")
            return False

        logger.info(f"Creating vector database: {vector_store_name}")

        vector_store = None
        try:
            # asyncio.to_thread moves the synchronous OGX-client
            # call to a separate thread so the asyncio (event) loop keeps
            # running.
            vector_store = await asyncio.to_thread(
                self.client.vector_stores.create,
                CreateVectorStoreRequest(
                    name=vector_store_name,
                    additional_properties=self.vector_store_options,
                ),
            )
            self.vector_store_ids.append(vector_store.id)
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                logger.info(
                    f"Vector DB '{vector_store_name}' already exists, continuing..."
                )

                # vector store exists, find it by name to get its ID
                # to continue ingestion
                for vs in await asyncio.to_thread(
                    list_all_vector_stores, self.client
                ):
                    if vs.name == vector_store_name:
                        vector_store = vs
                        break

                if vector_store is None:
                    logger.error(
                        f"Could not find existing vector store '{vector_store_name}'"
                    )
                    return False

                # if the existing store already contains files, we skip
                # ingestion entirely. Simple rule to prevent duplicate documents
                # when multiple parallel sessions race to populate the same store.
                # TODO: investigate more robust approaches
                try:
                    existing_files = await asyncio.to_thread(
                        list_all_vector_store_files,
                        self.client,
                        vector_store.id,
                    )
                    if existing_files and len(existing_files) > 0:
                        logger.info(
                            f"Vector store '{vector_store_name}' already has files, "
                            f"skipping document insertion"
                        )
                        return True
                except Exception as list_err:
                    logger.debug(
                        f"Could not list files for '{vector_store_name}': {list_err}"
                    )
            else:
                logger.error(f"Failed to register vector DB '{vector_store_name}': {e}")
                return False

        # ingest documents into the vector store.
        # - async mode: all documents are inserted concurrently using
        # asyncio.gather (each ingestion action runs in its own thread).
        # - sync mode: documents are inserted one at a time sequentially.
        try:
            if self.ingestion_mode == "async":
                logger.info(
                    f"Inserting {len(documents)} documents into vector store "
                    f"concurrently (async mode)..."
                )
                results = await asyncio.gather(
                    *[
                        asyncio.to_thread(
                            self._insert_single_document,
                            vector_store.id,
                            vector_store_name,
                            doc,
                        )
                        for doc in documents
                    ]
                )
            else:  # sync mode
                logger.info(
                    f"Inserting {len(documents)} documents into vector store "
                    f"sequentially (sync mode)..."
                )
                results = []
                for doc in documents:
                    result = await asyncio.to_thread(
                        self._insert_single_document,
                        vector_store.id,
                        vector_store_name,
                        doc,
                    )
                    results.append(result)

            # check if all insertions were successful
            success = all(results)
            if success:
                logger.info(
                    f"✓ Successfully inserted all {len(documents)} documents "
                    f"into '{vector_store_name}'"
                )
            else:
                failed_count = sum(1 for r in results if not r)
                logger.warning(
                    f"⚠ {failed_count}/{len(documents)} documents failed to insert "
                    f"into '{vector_store_name}'"
                )
            return success

        except Exception as e:
            logger.error(f"Error inserting documents into '{vector_store_name}': {e}")
            return False

    async def process_pipeline(self, pipeline: "Pipeline") -> "bool":
        """
        processes a single pipeline with concurrent operations.
        """
        logger.info(f"Processing pipeline: {pipeline.name}")

        if pipeline.enabled is False:
            logger.info(f"Pipeline '{pipeline.name}' is disabled, skipping")
            return True

        # fetching important values from pipeline
        vector_store_name = pipeline.vector_store_name
        source = pipeline.source
        category = (
            vector_store_name.split("-")[0] if vector_store_name else pipeline.name
        )

        github_base_url = ""
        if source == SourceTypes.GITHUB:
            github_url = pipeline.source_config.url.rstrip(".git").rstrip("/")
            github_base_url = (
                f"{github_url}/blob/"
                f"{pipeline.source_config.branch}/"
                f"{pipeline.source_config.path}".rstrip("/")
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            if source == SourceTypes.GITHUB:
                path_value = pipeline.source_config.path
                if path_value is None:
                    logger.error(
                        f"Pipeline '{pipeline.name}' has None path for GitHub source"
                    )
                    return False
                # github fetching is synchronous to avoid rate-limits,
                # runs in thread
                pdf_files = await asyncio.to_thread(
                    self.fetch_from_github,
                    pipeline.source_config.url,
                    path_value,
                    pipeline.source_config.branch,
                    temp_dir,
                )
            elif source == SourceTypes.URL:
                urls_value = pipeline.source_config.urls
                if urls_value is None:
                    logger.error(
                        f"Pipeline '{pipeline.name}' has None urls for URL source"
                    )
                    return False
                pdf_files = await self.fetch_from_url(urls_value, temp_dir)
            else:
                logger.error(f"Unknown source type: {source}")
                return False

            # if no pdf files found, log and skip
            if not pdf_files:
                logger.warning(f"No PDF files found for pipeline '{pipeline.name}'")
                return False

            documents = await self.process_documents(
                pdf_files, github_base_url, category
            )

            if not documents:
                logger.warning(f"No documents processed for pipeline '{pipeline.name}'")
                return False

            return await self.create_vector_db(vector_store_name, documents)

    def save_file_metadata(self) -> "None":
        """
        saves file metadata mapping to JSON for use by RAG service.
        Uses self.file_metadata_path (from RAG_FILE_METADATA env var).
        """
        if not self.file_metadata:
            logger.warning("No file metadata to save")
            return

        try:
            with open(self.file_metadata_path, "w") as f:
                json.dump(self.file_metadata, f, indent=2)
            logger.info(
                f"Saved file metadata for {len(self.file_metadata)} "
                f"files to {self.file_metadata_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save file metadata: {e}")

    async def run(self) -> "None":
        """
        runs the ingestion service.
        Mode is determined by ingestion_mode setting:
        - async: pipelines and documents processed concurrently
        - sync: pipelines and documents processed sequentially
        """
        mode_label = "concurrent" if self.ingestion_mode == "async" else "sequential"
        logger.info(f"Starting RAG Ingestion Service ({mode_label} mode)")
        logger.info(f"Configuration: {os.path.abspath(self.config_path)}")
        total = len(self.pipelines)

        # filter out disabled pipelines
        enabled_pipelines = [p for p in self.pipelines if p.enabled]
        skipped = total - len(enabled_pipelines)

        logger.info(
            f"Processing {len(enabled_pipelines)} enabled pipelines {mode_label}ly..."
        )

        # process all enabled pipelines (concurrent or sequential based on mode)
        async def process_with_error_handling(pipeline: Pipeline) -> bool:
            try:
                return await self.process_pipeline(pipeline)
            except Exception as e:
                logger.error(
                    f"Unexpected error processing pipeline '{pipeline.name}': {e}"
                )
                return False

        if self.ingestion_mode == "async":
            results = await asyncio.gather(
                *[process_with_error_handling(p) for p in enabled_pipelines]
            )
        else:
            results = []
            for pipeline in enabled_pipelines:
                result = await process_with_error_handling(pipeline)
                results.append(result)

        # count results
        successful = sum(1 for r in results if r)
        failed = len(results) - successful

        logger.debug(f"\n{'=' * 60}")
        logger.debug("Ingestion Summary")
        logger.debug(f"{'=' * 60}")
        logger.debug(f"Total pipelines: {total}")
        logger.debug(f"Successful: {successful}")
        logger.debug(f"Failed: {failed}")
        logger.debug(f"Skipped: {skipped}")
        logger.debug(f"{'=' * 60}\n")

        if successful == 0:
            logger.warning("all pipeline(s) failed. Check logs for details.")
            raise IngestionPipelineError(
                "All ingestion pipelines failed. Check logs for details."
            )
        elif failed > 0:
            logger.warning(f"{failed} pipeline(s) failed. Check logs for details.")
        else:
            logger.info("All pipelines completed successfully!")

        # Save file metadata for RAG source references
        self.save_file_metadata()
