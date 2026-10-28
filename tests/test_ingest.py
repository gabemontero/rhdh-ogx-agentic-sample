import json
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml

from src.exceptions import IngestionPipelineError
from src.ingest import IngestionService
from src.types import Pipeline, SourceConfig, SourceTypes


def _page(items):
    """Pagination-shaped mock page (has_more=False) for OGX list endpoints."""
    return SimpleNamespace(
        data=items, has_more=False, last_id=items[-1] if items else None
    )


class TestIngestionServiceInit:
    """
    tests for IngestionService initialization.
    """

    def test_init_with_valid_config(self, sample_config_file, mock_ogx_client):
        with patch(
            "src.ingest.check_ogx_availability",
            return_value={"connected": True, "error_message": ""},
        ):
            with patch(
                "src.ingest.OgxClient", return_value=mock_ogx_client
            ):
                service = IngestionService(sample_config_file)

                assert service.llama_stack_url == "http://localhost:8321"
                assert (
                    service.vector_db_config.embedding_model == "text-embedding-ada-002"
                )
                assert service.file_metadata == {}
                assert isinstance(service.pipelines, list)

    def test_init_with_invalid_config_missing_llamastack(
        self, temp_dir, mock_ogx_client
    ):
        invalid_config = {"vector_db": {}, "pipelines": {}}
        config_path = os.path.join(temp_dir, "invalid.yaml")
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)

        with patch(
            "src.ingest.check_ogx_availability",
            return_value={"connected": True, "error_message": ""},
        ):
            with patch(
                "src.ingest.OgxClient", return_value=mock_ogx_client
            ):
                with pytest.raises(SystemExit):
                    IngestionService(config_path)

    def test_init_with_github_token(
        self, sample_config_file, mock_ogx_client, monkeypatch
    ):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token")

        with patch(
            "src.ingest.check_ogx_availability",
            return_value={"connected": True, "error_message": ""},
        ):
            with patch(
                "src.ingest.OgxClient", return_value=mock_ogx_client
            ):
                service = IngestionService(sample_config_file)
                assert service.gh_client is not None

    def test_init_without_github_token(
        self, sample_config_file, mock_ogx_client, monkeypatch
    ):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        with patch(
            "src.ingest.check_ogx_availability",
            return_value={"connected": True, "error_message": ""},
        ):
            with patch(
                "src.ingest.OgxClient", return_value=mock_ogx_client
            ):
                service = IngestionService(sample_config_file)
                assert service.gh_client is not None


class TestInitializeLlamaStackClient:
    """
    tests for _initialize_ogx_client method.
    """

    def test_successful_connection_on_first_attempt(
        self, sample_config_file, mock_ogx_client
    ):
        with patch(
            "src.ingest.check_ogx_availability",
            return_value={"connected": True, "error_message": ""},
        ):
            with patch(
                "src.ingest.OgxClient", return_value=mock_ogx_client
            ):
                service = IngestionService(sample_config_file)
                assert service.client is not None

    def test_successful_connection_after_retry(
        self, sample_config_file, mock_ogx_client
    ):
        # First attempt fails, second succeeds
        with patch(
            "src.ingest.check_ogx_availability",
            side_effect=[
                {"connected": False, "error_message": "Connection refused"},
                {"connected": True, "error_message": ""},
            ],
        ):
            with patch(
                "src.ingest.OgxClient", return_value=mock_ogx_client
            ):
                with patch("src.ingest.time.sleep"):
                    service = IngestionService(sample_config_file)
                    assert service.client is not None

    def test_connection_failure_after_all_retries(self, sample_config_file):
        with patch(
            "src.ingest.check_ogx_availability",
            return_value={"connected": False, "error_message": "Connection refused"},
        ):
            with patch("src.ingest.time.sleep"):
                with pytest.raises(SystemExit):
                    IngestionService(sample_config_file)

    def test_connection_uses_correct_url(
        self, sample_config_file, mock_ogx_client
    ):
        with patch(
            "src.ingest.check_ogx_availability",
            return_value={"connected": True, "error_message": ""},
        ) as mock_check:
            with patch(
                "src.ingest.OgxClient", return_value=mock_ogx_client
            ):
                service = IngestionService(sample_config_file)
                mock_check.assert_called_with("http://localhost:8321")


class TestConfigValidation:
    """
    tests for _config_is_valid method.
    """

    def test_valid_config(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)
            assert service is not None

    def test_invalid_config_missing_vector_db(
        self, temp_dir, mock_ogx_client, mock_ogx_availability
    ):
        invalid_config = {
            "llamastack": {"base_url": "http://localhost:8321"},
            "pipelines": {},
        }
        config_path = os.path.join(temp_dir, "invalid.yaml")
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)

        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            with pytest.raises(SystemExit):
                IngestionService(config_path)


class TestParsePipelines:
    """
    tests for _parse_pipelines method.
    """

    def test_parse_github_pipeline(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

            github_pipelines = [
                p for p in service.pipelines if p.source == SourceTypes.GITHUB
            ]
            assert len(github_pipelines) > 0
            assert github_pipelines[0].source_config.url != ""
            assert github_pipelines[0].source_config.branch == "main"

    def test_parse_url_pipeline(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

            url_pipelines = [
                p for p in service.pipelines if p.source == SourceTypes.URL
            ]
            assert len(url_pipelines) > 0

    def test_parse_disabled_pipeline(
        self, temp_dir, mock_ogx_client, mock_ogx_availability
    ):
        config = {
            "llamastack": {"base_url": "http://localhost:8321"},
            "vector_db": {"embedding_model": "test", "chunk_size_in_tokens": 512},
            "pipelines": {
                "disabled_pipeline": {
                    "name": "disabled",
                    "enabled": False,
                    "version": "v1.0",
                    "vector_store_name": "test-db",
                    "source": "GITHUB",
                    "config": {"url": "https://github.com/test/repo"},
                }
            },
        }
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(config_path)
            assert len(service.pipelines) > 0
            assert service.pipelines[0].enabled is False


class TestFetchFromGithub:
    """
    tests for fetch_from_github method.
    """

    def test_fetch_from_github_with_valid_repo(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability, temp_dir, mock_github_repo
    ):
        mock_pdf_content = Mock()
        mock_pdf_content.name = "test.pdf"
        mock_pdf_content.path = "docs/test.pdf"
        mock_pdf_content.type = "file"
        mock_pdf_content.decoded_content = b"PDF content"

        mock_github_repo.get_contents.return_value = [mock_pdf_content]

        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        with patch.object(service, "_get_github_repo", return_value=mock_github_repo):
            pdf_files = service.fetch_from_github(
                "https://github.com/test/repo", "docs", "main", temp_dir
            )

            assert len(pdf_files) == 1
            assert pdf_files[0].endswith("test.pdf")

    def test_fetch_from_github_with_invalid_repo(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability, temp_dir
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        with patch.object(service, "_get_github_repo", return_value=None):
            pdf_files = service.fetch_from_github(
                "https://github.com/invalid/repo", "docs", "main", temp_dir
            )

            assert pdf_files == []

    def test_fetch_from_github_with_directory(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability, temp_dir, mock_github_repo
    ):
        mock_dir = Mock()
        mock_dir.type = "dir"
        mock_dir.path = "docs"

        mock_github_repo.get_contents.return_value = [mock_dir]

        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        with patch.object(service, "_get_github_repo", return_value=mock_github_repo):
            with patch.object(
                service, "_fetch_github_dir_contents", return_value=[]
            ) as mock_fetch:
                service.fetch_from_github(
                    "https://github.com/test/repo", "docs", "main", temp_dir
                )

                mock_fetch.assert_called_once()


class TestFetchFromUrl:
    """
    tests for fetch_from_url method.
    """

    @pytest.mark.anyio
    async def test_fetch_from_url_success(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability, temp_dir
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        mock_response = Mock()
        mock_response.content = b"PDF content"
        mock_response.raise_for_status = Mock()

        with patch("src.ingest.requests.get", return_value=mock_response):
            urls = ["https://example.com/test.pdf"]
            pdf_files = await service.fetch_from_url(urls, temp_dir)

            assert len(pdf_files) == 1
            assert pdf_files[0].endswith("test.pdf")

    @pytest.mark.anyio
    async def test_fetch_from_url_with_error(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability, temp_dir
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        with patch("src.ingest.requests.get", side_effect=Exception("Network error")):
            urls = ["https://example.com/test.pdf"]
            pdf_files = await service.fetch_from_url(urls, temp_dir)

            assert pdf_files == []

    @pytest.mark.anyio
    async def test_fetch_from_url_without_pdf_extension(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability, temp_dir
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        mock_response = Mock()
        mock_response.content = b"PDF content"
        mock_response.raise_for_status = Mock()

        with patch("src.ingest.requests.get", return_value=mock_response):
            urls = ["https://example.com/test"]
            pdf_files = await service.fetch_from_url(urls, temp_dir)

            assert len(pdf_files) == 1
            assert pdf_files[0].endswith(".pdf")


class TestProcessDocuments:
    """
    tests for process_documents method.
    """

    @pytest.mark.anyio
    async def test_process_documents_success(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability, sample_pdf_file
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        mock_file_response = Mock()
        mock_file_response.id = "test-file-id"
        service.client.files.create = Mock(return_value=mock_file_response)

        with patch.object(service, "converter") as mock_converter:
            mock_result = Mock()
            mock_document = Mock()
            mock_document.export_to_markdown.return_value = "Test content"
            mock_result.document = mock_document
            mock_converter.convert.return_value = mock_result

            documents = await service.process_documents(
                [sample_pdf_file],
                github_base_url="https://github.com/test/repo/blob/main/docs",
                category="legal",
            )

            assert len(documents) == 1
            assert "test-file-id" in service.file_metadata
            assert (
                service.file_metadata["test-file-id"]["original_filename"] == "test.pdf"
            )
            assert service.file_metadata["test-file-id"]["category"] == "legal"

    @pytest.mark.anyio
    async def test_process_documents_with_error(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability, sample_pdf_file
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        with patch.object(
            service, "converter", side_effect=Exception("Conversion error")
        ):
            documents = await service.process_documents(
                [sample_pdf_file], github_base_url="", category="legal"
            )

            assert documents == []


class TestCreateVectorDb:
    """
    tests for create_vector_db method.
    """

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "overrides, expected",
        [
            (
                {},
                {},
            ),
            (
                {"EMBEDDING_MODEL": "openai/text-embedding-3-small"},
                {"embedding_model": "openai/text-embedding-3-small"},
            ),
            (
                {"EMBEDDING_DIMENSION": "768"},
                {"embedding_dimension": "768"},
            ),
            (
                {"VECTOR_DB_PROVIDER_ID": "faiss"},
                {"provider_id": "faiss"},
            ),
            (
                {
                    "EMBEDDING_MODEL": "remote/embedding",
                    "EMBEDDING_DIMENSION": "768",
                    "VECTOR_DB_PROVIDER_ID": "faiss",
                },
                {
                    "embedding_model": "remote/embedding",
                    "embedding_dimension": "768",
                    "provider_id": "faiss",
                },
            ),
        ],
    )
    async def test_create_vector_db_sends_configured_embedding_settings(
        self,
        sample_config_file,
        mock_ogx_client,
        mock_ogx_availability,
        monkeypatch,
        overrides,
        expected,
    ):
        for name in ["EMBEDDING_MODEL", "EMBEDDING_DIMENSION", "VECTOR_DB_PROVIDER_ID"]:
            monkeypatch.delenv(name, raising=False)
        for name, value in overrides.items():
            monkeypatch.setenv(name, value)
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        result = await service.create_vector_db("test-vector-db", [Mock(id="doc-id")])

        assert result is True
        request = mock_ogx_client.vector_stores.create.call_args.args[0]
        assert request.to_dict() == {"name": "test-vector-db", **expected}

    @pytest.mark.anyio
    async def test_create_vector_db_success(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        mock_doc = Mock()
        mock_doc.id = "doc-id-1"

        result = await service.create_vector_db("test-vector-db", [mock_doc])

        assert result is True
        assert "test-vector-store-id" in service.vector_store_ids

    @pytest.mark.anyio
    async def test_create_vector_db_with_no_documents(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        result = await service.create_vector_db("test-vector-db", [])

        assert result is False

    @pytest.mark.anyio
    async def test_create_vector_db_already_exists(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        mock_ogx_client.vector_stores.create.side_effect = Exception(
            "already exists"
        )

        # Mock existing vector store
        mock_existing_vs = Mock()
        mock_existing_vs.name = "test-vector-db"
        mock_existing_vs.id = "existing-vector-store-id"
        mock_ogx_client.vector_stores.list.return_value = _page(
            [mock_existing_vs]
        )

        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        mock_doc = Mock()
        mock_doc.id = "doc-id-1"

        result = await service.create_vector_db("test-vector-db", [mock_doc])

        assert result is True

    @pytest.mark.anyio
    async def test_create_vector_db_already_exists_beyond_first_page(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        mock_ogx_client.vector_stores.create.side_effect = Exception(
            "already exists"
        )

        # named store sits on the SECOND page of the list response
        mock_other_vs = Mock()
        mock_other_vs.name = "another-vector-db"
        mock_other_vs.id = "other-vs-id"
        mock_existing_vs = Mock()
        mock_existing_vs.name = "test-vector-db"
        mock_existing_vs.id = "existing-vector-store-id"
        mock_ogx_client.vector_stores.list.side_effect = [
            SimpleNamespace(
                data=[mock_other_vs], has_more=True, last_id="other-vs-id"
            ),
            SimpleNamespace(
                data=[mock_existing_vs], has_more=False, last_id="existing-vector-store-id"
            ),
        ]

        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        mock_doc = Mock()
        mock_doc.id = "doc-id-1"

        result = await service.create_vector_db("test-vector-db", [mock_doc])

        # recovery path must find the store even past the first page
        assert result is True


class TestSaveFileMetadata:
    """
    tests for save_file_metadata method.
    """

    def test_save_file_metadata_success(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability, temp_dir
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        service.file_metadata = {
            "file-1": {"original_filename": "test.pdf", "category": "legal"}
        }

        output_path = os.path.join(temp_dir, "metadata.json")
        service.file_metadata_path = output_path
        service.save_file_metadata()

        assert os.path.exists(output_path)

        with open(output_path, "r") as f:
            saved_metadata = json.load(f)

        assert saved_metadata == service.file_metadata

    def test_save_file_metadata_with_empty_metadata(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability, temp_dir
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        service.file_metadata = {}

        output_path = os.path.join(temp_dir, "metadata.json")
        service.file_metadata_path = output_path
        service.save_file_metadata()

        assert not os.path.exists(output_path)


class TestProcessPipeline:
    """
    tests for process_pipeline method.
    """

    @pytest.mark.anyio
    async def test_process_pipeline_disabled(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        disabled_pipeline = Pipeline(
            name="test",
            enabled=False,
            version="v1.0",
            vector_store_name="test-db",
            source=SourceTypes.GITHUB,
            source_config=SourceConfig(url="", branch="main", path="", urls=None),
        )

        result = await service.process_pipeline(disabled_pipeline)

        assert result is True

    @pytest.mark.anyio
    async def test_process_pipeline_with_github_source(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        github_pipeline = Pipeline(
            name="test",
            enabled=True,
            version="v1.0",
            vector_store_name="test-db",
            source=SourceTypes.GITHUB,
            source_config=SourceConfig(
                url="https://github.com/test/repo",
                branch="main",
                path="docs",
                urls=None,
            ),
        )

        with patch.object(service, "fetch_from_github", return_value=[]):
            result = await service.process_pipeline(github_pipeline)

            assert result is False

    @pytest.mark.anyio
    async def test_process_pipeline_with_url_source(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        url_pipeline = Pipeline(
            name="test",
            enabled=True,
            version="v1.0",
            vector_store_name="test-db",
            source=SourceTypes.URL,
            source_config=SourceConfig(
                url="", branch="", path="", urls=["https://example.com/test.pdf"]
            ),
        )

        with patch.object(service, "fetch_from_url", return_value=[]):
            result = await service.process_pipeline(url_pipeline)

            assert result is False


class TestRun:
    """
    tests for run method.
    """

    @pytest.mark.anyio
    async def test_run_with_successful_pipelines(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        with patch.object(service, "process_pipeline", return_value=True):
            with patch.object(service, "save_file_metadata"):
                await service.run()

    @pytest.mark.anyio
    async def test_run_with_failed_pipelines(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        with patch.object(service, "process_pipeline", return_value=False):
            with patch.object(service, "save_file_metadata"):
                with pytest.raises(IngestionPipelineError):
                    await service.run()

    @pytest.mark.anyio
    async def test_run_with_mixed_results(
        self, sample_config_file, mock_ogx_client, mock_ogx_availability
    ):
        with patch("src.ingest.OgxClient", return_value=mock_ogx_client):
            service = IngestionService(sample_config_file)

        with patch.object(service, "process_pipeline", side_effect=[True, False]):
            with patch.object(service, "save_file_metadata"):
                await service.run()
