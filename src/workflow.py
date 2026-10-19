import os
import time
from typing import Any, cast

from langgraph.graph import START, StateGraph
from ogx_client import OpenAIResponseObject as ResponseObject

from src.constants import (
    DEFAULT_INFERENCE_MODEL,
    DEFAULT_LLAMA_STACK_URL,
    NO_DOCS_INDICATORS,
)
from src.methods import (
    classification_agent,
    git_agent,
    perf_agent,
    pod_agent,
    support_classification_agent,
)
from src.responses import RAGService
from src.types import WorkflowAgentPrompts, WorkflowState
from src.utils import (
    extract_rag_response_text,
    logger,
    route_to_next_node,
    submission_states,
    support_route_to_next_node,
)

INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", DEFAULT_INFERENCE_MODEL)
LLAMA_STACK_URL = os.getenv("LLAMA_STACK_URL", DEFAULT_LLAMA_STACK_URL)


class Workflow:
    def __init__(
        self,
        rag_service: "RAGService | None" = None,
        rag_prompt: "str" = WorkflowAgentPrompts.RAG_PROMPT,
    ):
        self.rag_prompt = rag_prompt
        self.rag_service = rag_service

    def _convert_messages_to_openai_format(
        self, messages_state: "WorkflowState | dict[str, Any]"
    ) -> "list[dict[str, str]]":
        """Convert state messages to OpenAI format"""
        messages = []
        for msg in messages_state.get("messages", []):
            if isinstance(msg, dict):
                messages.append(msg)
            elif isinstance(msg, str):
                messages.append({"role": "user", "content": msg})
            elif hasattr(msg, "content"):
                # fallback to langchain message objects
                role = (
                    "assistant"
                    if hasattr(msg, "__class__") and "AI" in msg.__class__.__name__
                    else "user"
                )
                messages.append({"role": role, "content": msg.content})
            else:
                messages.append({"role": "user", "content": str(msg)})
        return messages

    def _call_openai_llm(self, state: "WorkflowState") -> "str":
        """
        calls OpenAI chat completions API as a fallback
        """
        if not self.rag_service or not self.rag_service.openai_client:
            raise ValueError("OpenAI client not initialized")

        messages = self._convert_messages_to_openai_format(state)
        completion = self.rag_service.openai_client.chat.completions.create(
            model=INFERENCE_MODEL,
            messages=cast("Any", messages),
        )
        return completion.choices[0].message.content or ""

    def create_agent(
        self,
        department_name: "str",
        department_display_name: "str",
        content_override: "str | None" = None,
        custom_llm: "str | None" = None,
        submission_states: "dict[str, 'WorkflowState'] | None" = None,
        rag_category: "str | None" = None,
        additional_prompt: "str" = "",
        is_terminal: "bool" = False,
        client_override: "Any | None" = None,
    ) -> "Any":
        """
        factory function to create department-specific agents with consistent structure.

        Args:
            department_name: Internal name for the department
            (e.g., 'legal', 'support')
            department_display_name: Display name (e.g., 'Legal', 'Software Support')
            content_override: Optional content to override default prompt
            custom_llm: LLM model name for inference (e.g., 'ollama/llama3.2:3b')
            submission_states: Dictionary to store submission states
            rag_category: Category name to select appropriate vector stores (e.g.,
            'legal', 'techsupport', 'hr', 'sales', 'procurement')
            additional_prompt: Additional prompt instructions to append to the
            RAG prompt
            is_terminal: Whether this agent marks the end of the workflow
            client_override: Optional client to use for LLM calls instead of
            self.rag_service.client (default: None)
        """
        if custom_llm is None:
            raise ValueError("custom_llm is required")

        if submission_states is None:
            raise ValueError("submission_states is required")

        def init_message(state: "WorkflowState") -> "WorkflowState":
            logger.info(f"init {department_name} message '{state}'")
            if content_override:
                content = content_override
            else:
                user_query = state.get("input", "")
                content = (
                    WorkflowAgentPrompts.DEFAULT_TERMINAL_AGENT_INITIAL_CONTENT
                    if is_terminal
                    else WorkflowAgentPrompts.DEFAULT_NON_TERMINAL_AGENT_INITIAL_CONTENT
                ).format(
                    department_display_name=department_display_name,
                    submission_id=state["submission_id"],
                    user_query=user_query,
                )

            new_state = dict(state)
            new_state["messages"] = [{"role": "user", "content": content}]
            return cast("WorkflowState", new_state)

        def llm_node(state: "WorkflowState") -> "WorkflowState":
            logger.debug(f"{department_display_name} LLM node processing")

            agent_start_time = time.time()
            if "agent_timings" not in state or state["agent_timings"] is None:
                state["agent_timings"] = {}
            if "rag_query_time" not in state:
                state["rag_query_time"] = 0.0
            if "status_history" not in state or state["status_history"] is None:
                state["status_history"] = []

            state["active_agent"] = department_display_name
            status_msg = f"👤 {department_display_name} Agent processes the request..."
            state["status_message"] = status_msg
            state["status_history"].append(status_msg)

            submission_id = state.get("submission_id")
            if submission_id:
                submission_states[submission_id] = cast("WorkflowState", dict(state))
                logger.info(
                    f"Updated status: {status_msg} for submission {submission_id}"
                )

            # determine which client to use
            client_to_use = (
                client_override
                if client_override is not None
                else (self.rag_service.client if self.rag_service else None)
            )

            # check if RAG is available for this department
            use_rag = (
                self.rag_service is not None
                and rag_category is not None
                and hasattr(self.rag_service, "get_file_search_tool")
            )

            file_search_tool = None
            if use_rag:
                file_search_tool = self.rag_service.get_file_search_tool(rag_category)  # type: ignore[possibly-missing-attribute]
                if file_search_tool:
                    logger.info(
                        f"{department_display_name}: "
                        f"RAG enabled with category '{rag_category}'"
                    )
                else:
                    logger.info(
                        f"{department_display_name}: No vector stores for "
                        f"'{rag_category}', using standard LLM"
                    )
                    use_rag = False

            if use_rag and file_search_tool and self.rag_service and client_to_use:
                # Use RAG with file_search tool
                try:
                    rag_prompt = self.rag_prompt.format(
                        department_display_name=department_display_name,
                        user_input=state["input"],
                    )

                    if additional_prompt is not None:
                        rag_prompt += f"""\n{additional_prompt}"""

                    logger.info(
                        f"{department_display_name}: Making RAG-enabled response call"
                    )
                    rag_start_time = time.time()
                    # TODO: Here we could consider making the client choice
                    # more flexible, meaning we could use either LlamaStackClient
                    # or OpenAI client.
                    # OpenAI client is accessible in self.rag_service.openai_client
                    rag_response = client_to_use.responses.create(
                        model=INFERENCE_MODEL,
                        input=rag_prompt,
                        tools=cast("Any", [file_search_tool]),
                    )
                    rag_end_time = time.time()
                    state["rag_query_time"] = rag_end_time - rag_start_time

                    rag_response_obj: "ResponseObject" = cast(
                        ResponseObject, rag_response
                    )
                    response_text = extract_rag_response_text(rag_response_obj)

                    if rag_category:
                        sources = self.rag_service.extract_sources_from_response(
                            rag_response_obj, rag_category
                        )
                    else:
                        sources = []
                    state["rag_sources"] = sources
                    if sources:
                        logger.info(
                            f"{department_display_name}: Found {len(sources)} "
                            "source documents"
                        )

                    rag_sources_useful = True
                    if response_text and sources:
                        response_lower = response_text.lower()
                        for indicator in NO_DOCS_INDICATORS:
                            if indicator in response_lower:
                                rag_sources_useful = False
                                logger.info(
                                    f"{department_display_name}: "
                                    f"Response indicates irrelevant docs"
                                    " ('{indicator}'):, hiding sources"
                                )
                                break

                    state["rag_sources"] = (
                        sources if rag_sources_useful and sources else []
                    )

                    if response_text:
                        cm = response_text
                        logger.info(
                            f"{department_display_name}: RAG response successful"
                        )
                    else:
                        logger.warning(
                            f"{department_display_name}: RAG response empty, "
                            "falling back to standard LLM"
                        )
                        cm = self._call_openai_llm(state)
                        state["rag_sources"] = []
                except Exception as e:
                    logger.error(
                        f"{department_display_name}: RAG call failed: {e}, "
                        "falling back to standard LLM"
                    )
                    cm = self._call_openai_llm(state)
            else:
                cm = self._call_openai_llm(state)

            state["messages"] = [{"role": "assistant", "content": cm}]
            state["classification_message"] = cm
            if is_terminal:
                state["workflow_complete"] = True

            agent_end_time = time.time()
            state["agent_timings"][department_display_name] = (
                agent_end_time - agent_start_time
            )

            completion_msg = f"✅ {department_display_name} Agent finished"
            state["status_history"].append(completion_msg)
            state["status_message"] = completion_msg

            submission_id = state.get("submission_id")
            if submission_id:
                submission_states[submission_id] = cast("WorkflowState", dict(state))
                logger.info(
                    f"Updated status: {completion_msg} for submission {submission_id}"
                )

            return state

        agent_builder = StateGraph(cast("Any", WorkflowState))
        agent_builder.add_node(f"{department_name}_set_message", init_message)
        agent_builder.add_node("llm_node", llm_node)
        agent_builder.add_edge(START, f"{department_name}_set_message")
        agent_builder.add_edge(f"{department_name}_set_message", "llm_node")
        agent_workflow = agent_builder.compile()

        return agent_workflow

    def make_workflow(
        self,
        tools_llm: "str",
        mcp_tool_model: "str | None" = None,
        git_token: "str | None" = None,
        guardrail_model: "str | None" = None,
        github_url: "str | None" = None,
        guardrail_mode: str = "llama_guard",
    ):
        """Create and configure the overall workflow with all agents and routing.

        Args:
            tools_llm: LLM model name for classification, inference, and RAG
            (e.g., 'openai/gemini-3.1-pro-preview')
            mcp_tool_model: LLM model name used by the agents that make MCP tool
            calls (git, pod, perf). Falls back to tools_llm when not provided.
            Some models (e.g. Gemini 3) cannot complete MCP tool loops, so this
            is kept separate from tools_llm (e.g. 'ollama/llama3.1:latest').
            git_token: GitHub personal access token for git agent MCP calls (optional)
            guardrail_model: Model name for content moderation/safety checks (optional)
            github_url: GitHub repository URL for creating issues (optional)
            guardrail_mode: Safety classifier protocol: llama_guard or gpt.
        """
        # agents that make MCP tool calls need a model capable of tool loops;
        # fall back to tools_llm for backward compatibility
        if mcp_tool_model is None:
            mcp_tool_model = tools_llm
        # Create all department agents using the factory function
        # RAG is enabled for legal and support agents using their
        # respective vector stores
        legal_agent = self.create_agent(
            "legal",
            "Legal",
            custom_llm=tools_llm,
            submission_states=submission_states,
            rag_category="legal",
            is_terminal=True,
            client_override=(
                self.rag_service.openai_client if self.rag_service else None
            ),
        )
        support_agent = self.create_agent(
            "support",
            "Software Support",
            custom_llm=tools_llm,
            submission_states=submission_states,
            rag_category="techsupport",
            is_terminal=False,
        )

        hr_agent = self.create_agent(
            department_name="hr",
            department_display_name="Human Resources",
            custom_llm=tools_llm,
            submission_states=submission_states,
            rag_category="hr",
            additional_prompt="""
            FantaCo's benefits description is organized into such categories as:
            - bare necessities: workspace, as well as benefits such as health care,
            vacation or PTO, retirement plans
            - beyond the basics: music, parties and activities, food and driving
            services, bonuses
            - and then some minor caveats: random set of participation requirements
            if possible, try to narrow the scope of the response to the details that
            fall under on of those sub-sections of the benefits document.
            """,
            is_terminal=True,
        )

        sales_agent = self.create_agent(
            department_name="sales",
            department_display_name="Sales",
            custom_llm=tools_llm,
            submission_states=submission_states,
            rag_category="sales",
            additional_prompt="""
            FantaCo's sales operation manual outlines policies over ten
            broad categories :
            - geographic territories
            - lead assignments
            - discounting
            - deal approval
            - quotas
            - compensations
            - CRMs
            - brands and communications
            - expenses
            - escalations
            - performance
            - compliance
            if possible, try to narrow the scope of the response to the
            details that fall under on of those sub-sections of the sales
            document.
            """,
            is_terminal=True,
        )
        procurement_agent = self.create_agent(
            department_name="procurement",
            department_display_name="Procurement",
            custom_llm=tools_llm,
            submission_states=submission_states,
            rag_category="procurement",
            additional_prompt="""
            FantaCo's procurement policies cover :
            - competitive bidding
            - vendor evaluation and categorization
            - ethics and transparency
            - review
            - spending limits
            if possible, try to narrow the scope of the response to
            the details that fall under on of those sub-sections of
            the procurement document.
            """,
            is_terminal=True,
        )

        def classification_node(state: "WorkflowState") -> "WorkflowState":
            if not self.rag_service or not self.rag_service.openai_client:
                raise ValueError("RAG service or OpenAI client not initialized")
            return classification_agent(
                state,
                openai_client=self.rag_service.openai_client,
                topic_llm=tools_llm,
                guardrail_model=guardrail_model,
                guardrail_mode=guardrail_mode,
            )

        def support_classification_node(state: "WorkflowState") -> "WorkflowState":
            if not self.rag_service or not self.rag_service.openai_client:
                raise ValueError("RAG service or OpenAI client not initialized")
            return support_classification_agent(
                state, openai_client=self.rag_service.openai_client, topic_llm=tools_llm
            )

        def git_agent_node(state: "WorkflowState") -> "WorkflowState":
            return git_agent(
                state,
                openai_client=(
                    self.rag_service.openai_client if self.rag_service else None
                ),
                tools_llm=mcp_tool_model,
                git_token=git_token,
                github_url=github_url,
            )

        def pod_agent_node(state: "WorkflowState") -> "WorkflowState":
            return pod_agent(
                state,
                openai_client=(
                    self.rag_service.openai_client if self.rag_service else None
                ),
                tools_llm=mcp_tool_model,
            )

        def perf_agent_node(state: "WorkflowState") -> "WorkflowState":
            return perf_agent(
                state,
                openai_client=(
                    self.rag_service.openai_client if self.rag_service else None
                ),
                tools_llm=mcp_tool_model,
            )

        overall_workflow = StateGraph(cast("Any", WorkflowState))
        overall_workflow.add_node(
            "classification_agent",
            classification_node,
        )
        overall_workflow.add_node("legal_agent", legal_agent)
        overall_workflow.add_node("hr_agent", hr_agent)
        overall_workflow.add_node("sales_agent", sales_agent)
        overall_workflow.add_node("procurement_agent", procurement_agent)
        overall_workflow.add_node("support_agent", support_agent)
        overall_workflow.add_node("pod_agent", pod_agent_node)
        overall_workflow.add_node("perf_agent", perf_agent_node)
        overall_workflow.add_node("git_agent", git_agent_node)
        overall_workflow.add_node(
            "support_classification_agent", support_classification_node
        )
        overall_workflow.add_edge(START, "classification_agent")
        overall_workflow.add_conditional_edges(
            "classification_agent", route_to_next_node
        )
        overall_workflow.add_edge("support_agent", "support_classification_agent")
        overall_workflow.add_conditional_edges(
            "support_classification_agent", support_route_to_next_node
        )
        overall_workflow.add_edge("pod_agent", "git_agent")
        overall_workflow.add_edge("perf_agent", "git_agent")
        workflow = overall_workflow.compile()

        return workflow
