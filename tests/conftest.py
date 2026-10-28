import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml


@pytest.fixture(scope="session")
def anyio_backend():
    """Configure anyio to only use asyncio backend."""
    return "asyncio"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_ingestion_config():
    return {
        "llamastack": {"base_url": "http://localhost:8321"},
        "vector_db": {
            "embedding_model": "text-embedding-ada-002",
            "embedding_dimension": 1536,
            "chunk_size_in_tokens": 512,
        },
        "pipelines": {
            "legal_pipeline": {
                "name": "legal_pipeline",
                "enabled": True,
                "version": "v1.0",
                "vector_store_name": "legal-vector-db-v1-0",
                "source": "GITHUB",
                "config": {
                    "url": "https://github.com/test/repo.git",
                    "branch": "main",
                    "path": "docs",
                },
            },
            "url_pipeline": {
                "name": "url_pipeline",
                "enabled": True,
                "version": "v1.0",
                "vector_store_name": "url-vector-db-v1-0",
                "source": "URL",
                "urls": ["https://example.com/doc.pdf"],
            },
        },
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_ingestion_config):
    config_path = os.path.join(temp_dir, "ingestion-config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(sample_ingestion_config, f)
    return config_path


@pytest.fixture
def sample_file_metadata():
    return {
        "file-id-1": {
            "original_filename": "test.pdf",
            "github_url": "https://github.com/test/repo/blob/main/docs/test.pdf",
            "category": "legal",
        },
        "file-id-2": {
            "original_filename": "sample.pdf",
            "github_url": "https://github.com/test/repo/blob/main/docs/sample.pdf",
            "category": "hr",
        },
    }


@pytest.fixture
def sample_metadata_file(temp_dir, sample_file_metadata):
    metadata_path = os.path.join(temp_dir, "rag_file_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(sample_file_metadata, f)
    return metadata_path


@pytest.fixture
def mock_ogx_client():
    client = Mock()
    client.models.list.return_value = []
    # pagination-shaped empty page (bare Mock would loop/hang list helpers)
    client.vector_stores.list.return_value = SimpleNamespace(
        data=[], has_more=False, last_id=None
    )
    client.vector_stores.create.return_value = Mock(id="test-vector-store-id")
    client.vector_stores_files.create.return_value = Mock(
        id="test-file-id", status="completed"
    )
    client.vector_stores_files.list.return_value = SimpleNamespace(
        data=[], has_more=False, last_id=None
    )
    client.files.create.return_value = Mock(id="test-file-id")
    return client


@pytest.fixture
def mock_openai_client():
    client = Mock()

    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(content="safe"))]
    client.chat.completions.create.return_value = mock_completion

    # Mock for classification - create a simple object with instance attributes
    class MockParsed:
        def __init__(self):
            self.classification = "legal"
            self.namespace = "default"
            self.performance = "true"

    mock_parsed = MockParsed()

    mock_message = Mock()
    mock_message.parsed = mock_parsed

    mock_parsed_message = Mock()
    mock_parsed_message.message = mock_message

    mock_parsed_completion = Mock()
    mock_parsed_completion.choices = [mock_parsed_message]
    client.beta.chat.completions.parse.return_value = mock_parsed_completion

    mock_response = Mock()
    mock_response.output = []
    client.responses.create.return_value = mock_response

    return client


@pytest.fixture
def mock_github_repo():
    repo = Mock()
    repo.get_contents.return_value = []
    return repo


@pytest.fixture
def sample_workflow_state():
    return {
        "input": "Test question",
        "submission_id": "test-submission-id",
        "messages": [],
        "decision": "",
        "namespace": "",
        "data": "",
        "mcp_output": "",
        "github_issue": "",
        "rag_sources": [],
        "workflow_complete": False,
        "classification_message": "",
        "agent_timings": {},
        "rag_query_time": 0.0,
        "active_agent": "",
    }


@pytest.fixture
def sample_pdf_file(temp_dir):
    pdf_path = os.path.join(temp_dir, "test.pdf")
    pdf_content = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj\n<</Type/Pages/Count 0/Kids[]>>endobj\nxref\n0 3\n0000000000 65535 f\n0000000015 00000 n\n0000000060 00000 n\ntrailer\n<</Size 3/Root 1 0 R>>startxref\n110\n%%EOF"
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)
    return pdf_path


@pytest.fixture
def mock_document_converter():
    converter = Mock()
    mock_result = Mock()
    mock_document = Mock()
    mock_document.export_to_markdown.return_value = "Test markdown content"
    mock_result.document = mock_document
    converter.convert.return_value = mock_result
    return converter


@pytest.fixture
def mock_ogx_availability():
    """
    fixture that patches check_ogx_availability to return connected status.
    Use this in tests that need IngestionService initialization.
    """
    with patch(
        "src.ingest.check_ogx_availability",
        return_value={"connected": True, "error_message": ""},
    ) as mock:
        yield mock
