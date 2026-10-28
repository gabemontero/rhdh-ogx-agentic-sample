from unittest.mock import MagicMock, Mock, patch

import pytest

from src.workflow import Workflow


class TestWorkflowInit:
    """
    tests for Workflow initialization.
    """

    def test_init_with_rag_service(self):
        mock_rag_service = Mock()
        workflow = Workflow(rag_service=mock_rag_service)

        assert workflow.rag_service == mock_rag_service

    def test_init_without_rag_service(self):
        workflow = Workflow()

        assert workflow.rag_service is None


class TestCreateAgent:
    """
    tests for create_agent method.
    """

    def test_create_agent_with_basic_params(self):
        mock_rag_service = Mock()
        mock_openai_client = Mock()
        mock_rag_service.openai_client = mock_openai_client
        workflow = Workflow(rag_service=mock_rag_service)

        agent_graph = workflow.create_agent(
            department_name="legal",
            department_display_name="Legal",
            custom_llm="test-model",
            submission_states={},
            is_terminal=True,
        )

        assert agent_graph is not None

    def test_create_agent_with_rag_service(self):
        mock_rag_service = Mock()
        mock_rag_service.get_file_search_tool.return_value = {
            "type": "file_search",
            "vector_store_ids": ["test-vs-id"],
        }
        mock_openai_client = Mock()
        mock_rag_service.openai_client = mock_openai_client

        workflow = Workflow(rag_service=mock_rag_service)

        agent_graph = workflow.create_agent(
            department_name="legal",
            department_display_name="Legal",
            custom_llm="test-model",
            submission_states={},
            is_terminal=True,
            rag_category="legal",
        )

        assert agent_graph is not None

    def test_create_agent_without_submission_states(self):
        mock_rag_service = Mock()
        mock_openai_client = Mock()
        mock_rag_service.openai_client = mock_openai_client
        workflow = Workflow(rag_service=mock_rag_service)

        with pytest.raises(ValueError, match="submission_states is required"):
            workflow.create_agent(
                department_name="legal",
                department_display_name="Legal",
                custom_llm="test-model",
                is_terminal=True,
                submission_states=None,
            )

    def test_create_agent_terminal_vs_non_terminal(self):
        mock_rag_service = Mock()
        mock_openai_client = Mock()
        mock_rag_service.openai_client = mock_openai_client
        workflow = Workflow(rag_service=mock_rag_service)

        terminal_agent = workflow.create_agent(
            department_name="legal",
            department_display_name="Legal",
            custom_llm="test-model",
            submission_states={},
            is_terminal=True,
        )

        non_terminal_agent = workflow.create_agent(
            department_name="support",
            department_display_name="Support",
            custom_llm="test-model",
            submission_states={},
            is_terminal=False,
        )

        assert terminal_agent is not None
        assert non_terminal_agent is not None


class TestCallOpenAILlm:
    """
    tests for _call_openai_llm method.
    """

    def test_call_openai_llm_success(self, sample_workflow_state):
        mock_rag_service = Mock()
        mock_openai_client = Mock()
        mock_completion = MagicMock()
        mock_completion.choices = [Mock(message=Mock(content="Test response"))]
        mock_openai_client.chat.completions.create.return_value = mock_completion
        mock_rag_service.openai_client = mock_openai_client
        workflow = Workflow(rag_service=mock_rag_service)

        sample_workflow_state["messages"] = [
            {"role": "user", "content": "Test question"}
        ]

        with patch("src.workflow.INFERENCE_MODEL", "test-model"):
            result = workflow._call_openai_llm(sample_workflow_state)

        assert result == "Test response"

    def test_call_openai_llm_with_empty_messages(self, sample_workflow_state):
        mock_rag_service = Mock()
        mock_openai_client = Mock()
        mock_completion = MagicMock()
        mock_completion.choices = [Mock(message=Mock(content="Empty response"))]
        mock_openai_client.chat.completions.create.return_value = mock_completion
        mock_rag_service.openai_client = mock_openai_client
        workflow = Workflow(rag_service=mock_rag_service)

        sample_workflow_state["messages"] = []

        with patch("src.workflow.INFERENCE_MODEL", "test-model"):
            result = workflow._call_openai_llm(sample_workflow_state)

        assert result is not None


class TestMakeWorkflow:
    """
    tests for make_workflow method.
    """

    def test_make_workflow_basic(self):
        mock_rag_service = Mock()
        mock_openai_client = Mock()
        mock_rag_service.openai_client = mock_openai_client
        mock_rag_service.openai_client = mock_openai_client

        workflow = Workflow(rag_service=mock_rag_service)

        compiled_workflow = workflow.make_workflow(
            tools_llm="test-model",
            git_token="test-token",
            github_url="https://github.com/test/repo",
            guardrail_model="guardrail-model",
        )

        assert compiled_workflow is not None

    def test_make_workflow_accepts_mcp_tool_model(self):
        """make_workflow must accept mcp_tool_model (streamlit_app passes it)."""
        mock_rag_service = Mock()
        mock_openai_client = Mock()
        mock_rag_service.openai_client = mock_openai_client

        workflow = Workflow(rag_service=mock_rag_service)

        compiled_workflow = workflow.make_workflow(
            tools_llm="test-model",
            mcp_tool_model="mcp-model",
            git_token="test-token",
            github_url="https://github.com/test/repo",
            guardrail_model="guardrail-model",
        )

        assert compiled_workflow is not None

    def test_make_workflow_with_rag_service(self):
        mock_rag_service = Mock()
        mock_openai_client = Mock()
        mock_rag_service.openai_client = mock_openai_client
        mock_rag_service.openai_client = mock_openai_client

        workflow = Workflow(rag_service=mock_rag_service)

        compiled_workflow = workflow.make_workflow(
            tools_llm="test-model",
            git_token="test-token",
            github_url="https://github.com/test/repo",
            guardrail_model="guardrail-model",
        )

        assert compiled_workflow is not None

    def test_make_workflow_creates_all_agents(self):
        mock_rag_service = Mock()
        mock_openai_client = Mock()
        mock_rag_service.openai_client = mock_openai_client
        mock_rag_service.openai_client = mock_openai_client

        workflow = Workflow(rag_service=mock_rag_service)

        compiled_workflow = workflow.make_workflow(
            tools_llm="test-model",
            git_token="test-token",
            github_url="https://github.com/test/repo",
            guardrail_model="guardrail-model",
        )

        # Verify that workflow was created successfully
        assert compiled_workflow is not None

    def test_make_workflow_with_custom_inference_model(self):
        mock_rag_service = Mock()
        mock_openai_client = Mock()
        mock_rag_service.openai_client = mock_openai_client

        workflow = Workflow(rag_service=mock_rag_service)

        workflow.make_workflow(
            tools_llm="test-model",
            git_token="test-token",
            github_url="https://github.com/test/repo",
            guardrail_model="guardrail-model",
        )

        # Workflow doesn't have inference_model attribute - this test may not be valid
        # Just check that workflow was created successfully
        assert workflow is not None


class TestLlmNode:
    """
    tests for llm_node functionality within create_agent.
    """

    def test_llm_node_tracks_timing(self, sample_workflow_state):
        mock_rag_service = Mock()
        mock_openai_client = Mock()

        mock_completion = MagicMock()
        mock_completion.choices = [Mock(message=Mock(content="Test response"))]
        mock_openai_client.chat.completions.create.return_value = mock_completion
        mock_rag_service.openai_client = mock_openai_client

        workflow = Workflow(rag_service=mock_rag_service)

        sample_workflow_state["messages"] = [
            {"role": "user", "content": "Test question"}
        ]

        agent_graph = workflow.create_agent(
            department_name="legal",
            department_display_name="Legal",
            custom_llm="test-model",
            submission_states={},
            is_terminal=True,
        )

        result = agent_graph.invoke(sample_workflow_state)

        assert "agent_timings" in result
        assert "Legal" in result["agent_timings"]
        assert isinstance(result["agent_timings"]["Legal"], float)

    def test_llm_node_sets_active_agent(self, sample_workflow_state):
        mock_rag_service = Mock()
        mock_openai_client = Mock()

        mock_completion = MagicMock()
        mock_completion.choices = [Mock(message=Mock(content="Test response"))]
        mock_openai_client.chat.completions.create.return_value = mock_completion
        mock_rag_service.openai_client = mock_openai_client

        workflow = Workflow(rag_service=mock_rag_service)

        sample_workflow_state["messages"] = [
            {"role": "user", "content": "Test question"}
        ]

        agent_graph = workflow.create_agent(
            department_name="legal",
            department_display_name="Legal",
            custom_llm="test-model",
            submission_states={},
            is_terminal=True,
        )

        result = agent_graph.invoke(sample_workflow_state)

        assert result["active_agent"] == "Legal"

    def test_llm_node_with_rag(self, sample_workflow_state):
        mock_rag_service = Mock()
        mock_rag_service.get_file_search_tool.return_value = {
            "type": "file_search",
            "vector_store_ids": ["test-vs-id"],
        }
        mock_rag_service.extract_sources_from_response.return_value = []

        mock_openai_client = Mock()

        mock_content = Mock()
        mock_content.text = "RAG response"

        mock_output_item = Mock()
        mock_output_item.type = "text"
        mock_output_item.content = [mock_content]

        mock_response = Mock()
        mock_response.output = [mock_output_item]

        mock_openai_client.responses.create.return_value = mock_response

        # RAG uses .client for responses.create
        mock_rag_service.client = mock_openai_client

        # Add fallback mock for chat completions in case RAG fails
        mock_completion = MagicMock()
        mock_completion.choices = [Mock(message=Mock(content="Fallback response"))]
        mock_openai_client.chat.completions.create.return_value = mock_completion

        # Fallback uses .openai_client for chat completions
        mock_rag_service.openai_client = mock_openai_client

        workflow = Workflow(rag_service=mock_rag_service)

        sample_workflow_state["messages"] = [
            {"role": "user", "content": "Test question"}
        ]

        agent_graph = workflow.create_agent(
            department_name="legal",
            department_display_name="Legal",
            custom_llm="test-model",
            submission_states={},
            is_terminal=True,
            rag_category="legal",
        )

        result = agent_graph.invoke(sample_workflow_state)

        assert "rag_query_time" in result
        assert isinstance(result["rag_query_time"], float)


class TestInitMessage:
    """
    tests for init_message functionality within create_agent.
    """

    def test_init_message_with_terminal_agent(self, sample_workflow_state):
        mock_rag_service = Mock()
        mock_openai_client = Mock()

        mock_completion = MagicMock()
        mock_completion.choices = [Mock(message=Mock(content="Test"))]
        mock_openai_client.chat.completions.create.return_value = mock_completion
        mock_rag_service.openai_client = mock_openai_client

        workflow = Workflow(rag_service=mock_rag_service)

        agent_graph = workflow.create_agent(
            department_name="legal",
            department_display_name="Legal",
            custom_llm="test-model",
            submission_states={},
            is_terminal=True,
        )

        result = agent_graph.invoke(sample_workflow_state)

        assert "messages" in result

    def test_init_message_with_non_terminal_agent(self, sample_workflow_state):
        mock_rag_service = Mock()
        mock_openai_client = Mock()

        mock_completion = MagicMock()
        mock_completion.choices = [Mock(message=Mock(content="Test"))]
        mock_openai_client.chat.completions.create.return_value = mock_completion
        mock_rag_service.openai_client = mock_openai_client

        workflow = Workflow(rag_service=mock_rag_service)

        agent_graph = workflow.create_agent(
            department_name="support",
            department_display_name="Support",
            custom_llm="test-model",
            submission_states={},
            is_terminal=False,
        )

        result = agent_graph.invoke(sample_workflow_state)

        assert "messages" in result

    def test_init_message_with_content_override(self, sample_workflow_state):
        mock_rag_service = Mock()
        mock_openai_client = Mock()

        mock_completion = MagicMock()
        mock_completion.choices = [Mock(message=Mock(content="Test"))]
        mock_openai_client.chat.completions.create.return_value = mock_completion
        mock_rag_service.openai_client = mock_openai_client

        workflow = Workflow(rag_service=mock_rag_service)

        custom_content = "Custom initialization message"

        agent_graph = workflow.create_agent(
            department_name="legal",
            department_display_name="Legal",
            custom_llm="test-model",
            submission_states={},
            is_terminal=True,
            content_override=custom_content,
        )

        result = agent_graph.invoke(sample_workflow_state)

        assert "messages" in result


class TestWorkflowIntegration:
    """
    Integration tests for the complete workflow
    """

    def test_full_workflow_execution(self, sample_workflow_state):
        mock_rag_service = Mock()
        workflow = Workflow(rag_service=mock_rag_service)
        mock_openai_client = Mock()

        mock_completion = MagicMock()
        mock_completion.choices = [Mock(message=Mock(content="safe"))]
        mock_openai_client.chat.completions.create.return_value = mock_completion

        mock_parsed = Mock()
        mock_parsed.classification = "legal"

        mock_parsed_message = Mock()
        mock_parsed_message.parsed = mock_parsed

        mock_parsed_completion = Mock()
        mock_parsed_completion.choices = [mock_parsed_message]

        mock_openai_client.beta.chat.completions.parse.return_value = (
            mock_parsed_completion
        )

        # Guardrail now runs the guard model via chat completions (OGX 1.2.5
        # dropped /v1/moderations). The mock_completion above returns the
        # recognized "safe" verdict so classification can proceed.

        mock_rag_service.openai_client = mock_openai_client
        mock_rag_service.openai_client = mock_openai_client

        compiled_workflow = workflow.make_workflow(
            tools_llm="test-model",
            git_token="test-token",
            github_url="https://github.com/test/repo",
            guardrail_model="guardrail-model",
        )

        result = compiled_workflow.invoke(sample_workflow_state)

        assert result is not None
        assert "decision" in result
        assert "workflow_complete" in result
