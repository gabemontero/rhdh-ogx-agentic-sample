import asyncio
import json
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import httpx
import streamlit as st
from ogx_client import OgxClient

from src.constants import (
    DEFAULT_GUARDRAIL_MODEL,
    DEFAULT_INFERENCE_MODEL,
    DEFAULT_INGESTION_CONFIG,
    DEFAULT_LLAMA_STACK_URL,
    DEFAULT_MCP_TOOL_MODEL,
)
from src.exceptions import NoVectorStoresFoundError
from src.ingest import IngestionService
from src.responses import RAGService
from src.types import Pipeline, WorkflowState
from src.utils import (
    check_ogx_availability,
    ingestion_lock,
    list_all_vector_stores,
    logger,
    submission_states,
)
from src.workflow import Workflow

# API_KEY: OpenAI API key (not used directly but may be needed
API_KEY = os.getenv("OPENAI_API_KEY", "not applicable")

# INFERENCE_SERVER_OPENAI: URL of the inference server
INFERENCE_SERVER_OPENAI = os.getenv(
    "LLAMA_STACK_SERVER_OPENAI", "http://localhost:8321/v1"
)

# INFERENCE_MODEL: Model to use for inference
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", DEFAULT_INFERENCE_MODEL)

# GUARDRAIL_MODEL: Model to use for guardrails
GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL", DEFAULT_GUARDRAIL_MODEL)
GUARDRAIL_MODE = os.getenv("GUARDRAIL_MODE", "llama_guard")

# MCP_TOOL_MODEL: Model to use for MCP tool calls
MCP_TOOL_MODEL = os.getenv("MCP_TOOL_MODEL", DEFAULT_MCP_TOOL_MODEL)

# GITHUB_TOKEN: GitHub Personal Access Token for issue creation and comments
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "not applicable")

# GITHUB_URL: URL of the GitHub repository
GITHUB_URL = os.getenv("GITHUB_URL", "not applicable")

# GITHUB_ID: GitHub user ID
GITHUB_ID = os.getenv("GITHUB_ID", "not applicable")

# LLAMA_STACK_URL: URL of the OGX server
LLAMA_STACK_URL = os.getenv("LLAMA_STACK_URL", DEFAULT_LLAMA_STACK_URL)

# INGESTION_CONFIG: Path to the ingestion configuration file
INGESTION_CONFIG = os.getenv("INGESTION_CONFIG", DEFAULT_INGESTION_CONFIG)

# RAG_FILE_METADATA: Path to the RAG file metadata
RAG_FILE_METADATA = os.getenv("RAG_FILE_METADATA", "rag_file_metadata.json")

# EXECUTOR_WORKERS: Maximum number of concurrent workflow threads
EXECUTOR_WORKERS = int(os.getenv("EXECUTOR_WORKERS", "10"))

# AGENT_ICONS: Icons for different agent types
AGENT_ICONS = {
    "Classification": "🔍",
    "Legal": "⚖️",
    "Human Resources": "👥",
    "Sales": "💼",
    "Procurement": "🛒",
    "Software Support": "💻",
    "Pod": "☸️",
    "Performance": "⚡",
    "Git": "🔗",
}


@st.cache_resource
def get_config() -> "dict[str, str]":
    """
    gets application configuration (cached)
    """
    return {
        "inference_server": INFERENCE_SERVER_OPENAI,
        "llama_stack_url": LLAMA_STACK_URL,
        "ingestion_config": INGESTION_CONFIG,
        "rag_metadata": RAG_FILE_METADATA,
    }


@st.cache_resource
def get_ingestion_service() -> "IngestionService":
    """
    gets or creates a cached IngestionService instance
    """
    logger.info(f"Creating IngestionService with config: {INGESTION_CONFIG}")
    return IngestionService(INGESTION_CONFIG)


@st.cache_resource
def initialize_workflow(_pipelines: "list[Pipeline]") -> "tuple[Any, RAGService]":
    """
    initializes workflow with pipelines from ingestion (cached)
    Returns tuple of (compiled_workflow, rag_service)
    Note: _pipelines is prefixed with _ to avoid hashing by Streamlit
    """
    rag_service = RAGService(
        llama_stack_url=LLAMA_STACK_URL,
        ingestion_config_path=INGESTION_CONFIG,
        file_metadata_path=RAG_FILE_METADATA,
        pipelines=_pipelines,
    )
    try:
        rag_service_initialized = rag_service.initialize()
    except NoVectorStoresFoundError as e:
        # vector stores missing - trigger re-ingestion by resetting state
        logger.warning(f"Error during RAG initialization: {e}")
        ingestion_state = get_ingestion_state()
        ingestion_state["status"] = "pending"
        ingestion_state["message"] = "No vector stores found - re-ingestion required"
        ingestion_state["pipelines"] = None
        # clear cache to force re-initialization
        st.cache_resource.clear()
        st.rerun()

    if not rag_service_initialized:
        logger.warning("RAG Service initialization failed.")

    # build workflow graph with all agents and routing logic
    workflow_builder = Workflow(rag_service=rag_service)
    compiled_workflow = workflow_builder.make_workflow(
        tools_llm=INFERENCE_MODEL,
        mcp_tool_model=MCP_TOOL_MODEL,
        git_token=GITHUB_TOKEN,
        github_url=GITHUB_URL,
        guardrail_model=GUARDRAIL_MODEL,
        guardrail_mode=GUARDRAIL_MODE,
    )

    logger.info("✓ Workflow and RAG Service initialized and cached")
    return compiled_workflow, rag_service


def get_or_create_event_loop() -> "Any":
    """
    gets an existing event loop from session state or create a new one
    Used for ingestion tasks on the main thread
    """
    if "event_loop" not in st.session_state:
        # create persistent event loop for async task management
        st.session_state.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.event_loop)
    return st.session_state.event_loop


def get_or_create_executor() -> "ThreadPoolExecutor":
    """
    gets or creates a cached ThreadPoolExecutor for running workflows
    Each workflow will run in its own thread with its own event loop
    """
    if "thread_executor" not in st.session_state:
        # create thread pool executor for parallel workflow execution
        st.session_state.thread_executor = ThreadPoolExecutor(
            max_workers=EXECUTOR_WORKERS
        )
    return st.session_state.thread_executor


def get_futures_dict() -> "dict[str, Future[None]]":
    """
    gets futures dictionary from session state
    Tracks all running workflow futures
    """
    if "workflow_futures" not in st.session_state:
        # structure to store all running workflow futures
        st.session_state.workflow_futures = {}
    return st.session_state.workflow_futures


def has_active_workflows() -> "bool":
    """
    checks if any workflow futures are still running.
    Cleans up completed futures to prevent memory growth.
    """
    futures = get_futures_dict()
    if not futures:
        return False

    completed_ids = [sid for sid, future in futures.items() if future.done()]
    for sid in completed_ids:
        del futures[sid]

    return len(futures) > 0


def get_ingestion_state() -> "dict[str, Any]":
    """
    gets or initializes ingestion state
    """
    if "ingestion_state" not in st.session_state:
        st.session_state.ingestion_state = {
            "status": "pending",  # pending, running, completed, error, skipped
            "message": "",
            "ingested_count": 0,
            "pipelines": None,
        }
    return st.session_state.ingestion_state


def count_vector_stores() -> "int":
    """
    count the number of vector stores in the database
    """
    try:
        client = OgxClient(base_url=LLAMA_STACK_URL)
        count = len(list_all_vector_stores(client))
        logger.debug(f"Found {count} vector stores in database")
        return count
    except Exception as e:
        logger.warning(f"Failed to count vector stores: {e}")
        return 0


def get_ogx_status() -> "dict[str, Any]":
    """
    gets or initializes OGX status state.
    Tracks connectivity and model availability.
    """
    if "ogx_status" not in st.session_state:
        st.session_state.ogx_status = {
            "connected": False,
            "checked": False,
            "error_message": "",
            "available_models": [],
            "missing_models": [],
        }
    return st.session_state.ogx_status


def check_ogx_health() -> "dict[str, Any]":
    """
    performs health check of OGX: connectivity and model availability.
    """
    status = get_ogx_status()

    required_models = [INFERENCE_MODEL, GUARDRAIL_MODEL, MCP_TOOL_MODEL]
    result = check_ogx_availability(LLAMA_STACK_URL, required_models)

    status["connected"] = result["connected"]
    status["error_message"] = result["error_message"]
    status["available_models"] = result["available_models"]
    status["missing_models"] = result["missing_models"]
    status["checked"] = True

    return status


def render_ogx_errors() -> "bool":
    """
    renders error banners for OGX issues in sidebar.
    Returns True if there are blocking errors (no connection or missing models).
    """
    status = get_ogx_status()

    if not status["checked"]:
        return False

    # connection error case
    if not status["connected"]:
        st.error(
            "**OGX Unavailable**\n\n"
            f"{status['error_message']}\n\n"
            "**To fix:**\n"
            f"- Start OGX at `{LLAMA_STACK_URL}`\n"
            "- Or set `LLAMA_STACK_URL` to the correct address"
        )
        if st.button("Retry Connection"):
            check_ogx_health()
            st.rerun()
        return True

    # model unavailable case
    if status["missing_models"]:
        missing_list = "\n".join([f"- `{m}`" for m in status["missing_models"]])
        st.error(
            "**Models Not Found**\n\n"
            f"These models are not available in OGX:\n{missing_list}\n\n"
            "**To fix:**\n"
            "- Check your OGX `run.yaml` configuration\n"
            "- Or update the model environment variables"
        )
        return True

    return False


async def check_and_run_ingestion_if_needed() -> "None":
    """
    checks if all required vector stores exist and runs ingestion if needed.
    called automatically on startup (no user action required).
    """
    ingestion_state = get_ingestion_state()

    try:
        logger.info("Checking if ingestion is needed...")

        # check if vector stores for all pipelines exist
        ingestion_service = get_ingestion_service()
        pipelines = ingestion_service.pipelines
        temp_client = OgxClient(base_url=LLAMA_STACK_URL)
        temp_rag_service = RAGService(
            pipelines=pipelines, llama_stack_url=LLAMA_STACK_URL
        )
        temp_rag_service.client = temp_client
        all_stores_exist = temp_rag_service.check_vector_stores_exist(pipelines)

        if all_stores_exist:
            # all vector stores present - skip ingestion
            logger.info("All vector stores exist, skipping ingestion")
            vector_store_count = count_vector_stores()

            ingestion_state["status"] = "skipped"
            ingestion_state["message"] = (
                f"All vector stores exist - loaded {len(pipelines)}"
                " pipelines from config"
            )
            ingestion_state["pipelines"] = pipelines
            ingestion_state["vector_store_count"] = vector_store_count
        else:
            # vector stores missing - run ingestion to completion
            logger.info("Some vector stores missing, starting ingestion...")
            ingestion_state["status"] = "running"
            await run_ingestion_pipeline()

    except Exception as e:
        logger.error(f"Failed to check vector stores: {e}")
        # failed to check stores, mark as error to unblock UI
        ingestion_state["status"] = "error"
        ingestion_state["message"] = f"Failed to check vector stores: {str(e)}"
        ingestion_state["pipelines"] = []


async def run_ingestion_pipeline() -> "None":
    """
    runs the ingestion pipeline asynchronously.
    """
    ingestion_state = get_ingestion_state()

    try:
        ingestion_state["status"] = "running"
        ingestion_state["message"] = "Checking OGX server availability..."
        logger.info("Starting Ingestion Service...")

        # verify OGX server is reachable before ingestion
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{LLAMA_STACK_URL}/v1/health")
                if response.status_code != 200:
                    raise Exception(
                        f"OGX server returned status {response.status_code}"
                    )
                logger.info("✓ OGX server is available")
        except Exception as e:
            raise Exception(
                f"OGX server not available at {LLAMA_STACK_URL}. "
                f"Please start the server first: {e}"
            )

        ingestion_state["message"] = "Initializing ingestion service..."

        ingestion_service = get_ingestion_service()

        # run ingestion in background thread to avoid blocking
        ingestion_state["message"] = (
            "Running ingestion pipeline with concurrent processing..."
        )
        logger.info("Running ingestion pipeline with concurrent processing...")
        # run ingestion with concurrent processing
        await ingestion_service.run()

        # save pipeline config for workflow initialization
        ingestion_state["pipelines"] = ingestion_service.pipelines

        vector_store_count = await asyncio.to_thread(count_vector_stores)

        # mark complete - UI can now initialize workflow
        ingestion_state["status"] = "completed"
        ingestion_state["ingested_count"] = len(ingestion_service.pipelines)
        ingestion_state["vector_store_count"] = vector_store_count
        ingestion_state["message"] = (
            "Ingestion completed successfully with concurrent processing!"
        )
        logger.info(
            f"Ingestion completed: {ingestion_state['ingested_count']} "
            f"pipelines processed, {vector_store_count} vector stores in database"
        )

    except Exception as e:
        # pipeline failed, mark as error to unblock UI
        ingestion_state["status"] = "error"
        error_msg = str(e)
        ingestion_state["message"] = f"Ingestion failed: {error_msg[:200]}"
        logger.error(f"Ingestion pipeline failed: {e}", exc_info=True)


def _render_exchange_response(state: "WorkflowState") -> "None":
    """
    renders the agent response for a single exchange
    """
    # extract workflow state data
    decision = state.get("decision", "").lower()
    classification_msg = state.get("classification_message", "")
    is_complete = state.get("workflow_complete", False)
    active_agent = state.get("active_agent", "")
    status_message = state.get("status_message", "")
    status_history = state.get("status_history", [])
    is_error = decision in ("error", "unsafe", "unknown")

    # display status history with appropriate agent icons
    # TODO: find a way to handle this more efficiently
    for status_msg in status_history:
        if "Classification" in status_msg:
            status_icon = AGENT_ICONS.get("Classification", "🔍")
        elif "Support Classification" in status_msg:
            status_icon = AGENT_ICONS.get("Software Support", "💻")
        elif "Git" in status_msg:
            status_icon = AGENT_ICONS.get("Git", "🔗")
        elif "Pod" in status_msg:
            status_icon = AGENT_ICONS.get("Pod", "☸️")
        elif "Performance" in status_msg:
            status_icon = AGENT_ICONS.get("Performance", "⚡")
        elif "Legal" in status_msg:
            status_icon = AGENT_ICONS.get("Legal", "⚖️")
        elif "Human Resources" in status_msg or "HR" in status_msg:
            status_icon = AGENT_ICONS.get("Human Resources", "👥")
        elif "Sales" in status_msg:
            status_icon = AGENT_ICONS.get("Sales", "💼")
        elif "Procurement" in status_msg:
            status_icon = AGENT_ICONS.get("Procurement", "🛒")
        elif "Software Support" in status_msg:
            status_icon = AGENT_ICONS.get("Software Support", "💻")
        else:
            status_icon = "⏳"

        with st.chat_message("assistant", avatar=status_icon):
            if "✅" in status_msg:
                st.success(f"**{status_msg}**")
            else:
                st.info(f"**{status_msg}**")

    # show classification result and routing decision
    if decision and decision not in ("", "processing") and not is_error:
        agent_icon = AGENT_ICONS.get("Classification", "🔍")
        with st.chat_message("assistant", avatar=agent_icon):
            # map internal decision names to display names
            dept_map = {
                "legal": "Legal",
                "hr": "Human Resources",
                "sales": "Sales",
                "procurement": "Procurement",
                "techsupport": "Software Support",
                "support": "Software Support",
                "pod": "Pod",
                "perf": "Performance",
                "git": "Git",
            }
            dept_name = dept_map.get(decision, decision.title())
            dept_icon = AGENT_ICONS.get(dept_name, "🤖")
            st.success(
                f"**{agent_icon} Classification Agent**\n\n"
                f"Routed to {dept_icon} **{dept_name}**"
            )

    # handle active agent response, completion, or error
    if active_agent or is_complete or is_error:
        if is_error and not active_agent:
            agent_icon = AGENT_ICONS.get("Classification", "🔍")
            display_agent_name = "Classification Agent"
        else:
            agent_icon = AGENT_ICONS.get(active_agent, "🤖")
            if active_agent:
                display_agent_name = f"{active_agent} Agent"
            else:
                display_agent_name = "Agent"

        with st.chat_message("assistant", avatar=agent_icon):
            if not is_complete and not is_error:
                if not status_message:
                    st.info(
                        f"**{agent_icon} {display_agent_name}**\n\n"
                        f"⏳ Processing your request..."
                    )
            elif classification_msg and classification_msg != "Processing...":
                if is_error:
                    if decision == "unsafe":
                        st.error(
                            f"**{agent_icon} {display_agent_name}**\n\n❌ "
                            f"**Content Safety Issue**\n\n{classification_msg}"
                        )
                    elif decision == "unknown":
                        st.warning(
                            f"**{agent_icon} {display_agent_name}**\n\n⚠️"
                            f" **Unable to Process**\n\n{classification_msg}"
                        )
                    else:
                        st.error(
                            f"**{agent_icon} {display_agent_name}**\n\n❌"
                            f" **Error**\n\n{classification_msg}"
                        )
                else:
                    st.markdown(f"**{agent_icon} {display_agent_name}**")
                    st.write(classification_msg)
            elif is_error:
                st.error(
                    f"**{agent_icon} {display_agent_name}**\n\n❌ An error occurred."
                    f" Decision: {decision}"
                )

            # display workflow completion artifacts
            if is_complete and not is_error:
                # show RAG document sources used for response
                rag_sources = state.get("rag_sources", [])
                if rag_sources:
                    with st.expander("📚 Sources", expanded=False):
                        for idx, source in enumerate(rag_sources, 1):
                            filename = source.get("filename", "Unknown")
                            url = source.get("url", "")
                            if url:
                                st.markdown(f"{idx}. [{filename}]({url})")
                            else:
                                st.markdown(f"{idx}. {filename}")

                # display Kubernetes diagnostics from pod/perf agents
                mcp_output = state.get("mcp_output", "")
                if mcp_output:
                    with st.expander("📋 Diagnostics Output", expanded=True):
                        try:
                            parsed = json.loads(mcp_output)
                            st.json(parsed)  # Pretty print JSON
                        except (json.JSONDecodeError, ValueError):
                            st.code(mcp_output, language="text")

                # show GitHub issue created by git agent
                github_issue = state.get("github_issue", "")
                if github_issue:
                    st.markdown(f"[🎫 GitHub Issue Created]({github_issue})")
                elif "GitHub MCP Server Unavailable" in state.get("status_message", ""):
                    st.markdown(
                        "⚠️ GitHub Issue Not Created: GitHub MCP Server Unavailable"
                    )

            # handle completion of workflow
            if is_complete:
                agent_timings = state.get("agent_timings", {})
                if agent_timings:
                    st.markdown("**⏱️ Performance Metrics:**")
                    cols = st.columns(min(len(agent_timings), 3))
                    for idx, (agent_name, timing) in enumerate(agent_timings.items()):
                        with cols[idx % len(cols)]:
                            st.metric(agent_name, f"{timing:.2f}s")

                rag_time = state.get("rag_query_time", 0.0)
                if rag_time > 0:
                    st.metric("RAG Query", f"{rag_time:.2f}s")


def _run_async_in_thread(coro):
    """
    helper to run an async coroutine in a new thread with its own event loop
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def run_workflow_task_async(
    workflow: "Any", question: "str", submission_id: "str"
) -> "None":
    """
    async task to run a single workflow with concurrent operations
    This runs in its own thread with its own event loop
    """
    try:
        logger.info(f"Starting workflow task for submission {submission_id}")

        # run workflow - it will classify and route
        # the question through appropriate agents (classification -> dept agents)
        # Note: workflow.invoke is synchronous but we're in an async context
        # to allow future async operations within the workflow
        result = workflow.invoke(
            {
                "input": question,
                "submission_id": submission_id,
                "conversation_id": submission_id,
                "exchange_index": 0,
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
                "status_message": "",
                "status_history": [],
            },
        )

        # store result globally for UI rendering across reruns
        submission_states[submission_id] = result

        conversation_id = result.get("conversation_id")
        if conversation_id:
            # UI will read from submission_states
            pass

        logger.info(
            f"Workflow task completed for submission {submission_id}: "
            f"decision={result.get('decision')}, "
            f"complete={result.get('workflow_complete')}"
        )
    except Exception as e:
        logger.error(f"Workflow task failed for submission {submission_id}: {e}")
        error_state: "WorkflowState" = {
            "input": question,
            "submission_id": submission_id,
            "conversation_id": submission_id,
            "exchange_index": 0,
            "decision": "error",
            "classification_message": f"Error: {str(e)[:200]}",
            "workflow_complete": True,
            "mcp_output": "",
            "github_issue": "",
            "rag_sources": [],
            "messages": [],
            "namespace": "",
            "data": "",
            "agent_timings": {},
            "rag_query_time": 0.0,
            "active_agent": "",
            "status_message": "",
            "status_history": [],
        }
        submission_states[submission_id] = error_state


def run_workflow_task(
    workflow: "Workflow", question: "str", submission_id: "str"
) -> "None":
    """
    wrapper to run workflow task in thread with asyncio
    """
    _run_async_in_thread(run_workflow_task_async(workflow, question, submission_id))


def submit_workflow_task(
    workflow: "Workflow", question: "str", submission_id: "str"
) -> "None":
    """
    Submit a new workflow task to the ThreadPoolExecutor
    Each workflow runs in its own thread with its own event loop
    """
    executor = get_or_create_executor()
    futures = get_futures_dict()

    # submit workflow to run in background thread with its own event loop
    future = executor.submit(run_workflow_task, workflow, question, submission_id)
    futures[submission_id] = future

    logger.info(f"Submitted workflow task for {submission_id} to thread pool")


def _on_new_conversation() -> "None":
    """
    callback for the New Conversation button
    """
    st.session_state.selected_submission = None


def _on_select_conversation(conversation_id: "str") -> "None":
    """
    callback for selecting an existing conversation
    """
    st.session_state.selected_submission = conversation_id


def _on_clear_all() -> "None":
    """
    callback for the Clear All Conversations button
    """
    st.session_state.active_submissions = []
    st.session_state.selected_submission = None
    st.session_state.conversations = {}


def display_sidebar_conversations() -> "None":
    """
    displays conversation list in the sidebar.
    Status updates are driven by conditional polling in main().
    """
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("💬 Conversations")
    with col2:
        st.button(
            "➕",
            help="New conversation",
            use_container_width=True,
            on_click=_on_new_conversation,
        )

    # initialize conversation tracking state
    if "active_submissions" not in st.session_state:
        st.session_state.active_submissions = []

    if "selected_submission" not in st.session_state:
        st.session_state.selected_submission = None

    if st.session_state.active_submissions:
        for conversation_id in st.session_state.active_submissions:
            conversation_exchanges = st.session_state.conversations.get(
                conversation_id, []
            )

            if not conversation_exchanges:
                continue

            # get latest workflow state for this conversation
            latest_exchange = conversation_exchanges[-1]
            latest_submission_id = latest_exchange.get("submission_id", "")
            latest_state = submission_states.get(latest_submission_id, latest_exchange)

            is_complete = latest_state.get("workflow_complete", False)
            decision = latest_state.get("decision", "").lower()

            # determine status icon based on workflow state
            if decision in ("error", "unsafe", "unknown"):
                status_icon = "❌"
            elif is_complete:
                status_icon = "✅"
            else:
                status_icon = "⏳"

            first_question = conversation_exchanges[0].get("input", "")
            question_preview = (
                first_question[:25] + "..."
                if len(first_question) > 25
                else first_question
            )

            # show message count if multi-turn conversation
            exchange_count_str = ""
            if len(conversation_exchanges) > 1:
                exchange_count_str = f" ({len(conversation_exchanges)} msgs)"

            # highlight selected conversation
            button_type = (
                "primary"
                if st.session_state.selected_submission == conversation_id
                else "secondary"
            )
            st.button(
                f"{status_icon} {question_preview}{exchange_count_str}",
                key=f"select_{conversation_id}",
                type=button_type,
                use_container_width=True,
                on_click=_on_select_conversation,
                args=(conversation_id,),
            )
    else:
        st.info("No conversations yet")

    # clear all conversations button
    st.button("Clear All Conversations", on_click=_on_clear_all)

    st.divider()

    with st.expander("🤖 Agent Reference", expanded=False):
        st.markdown("**Available Agents:**")
        for agent, icon in AGENT_ICONS.items():
            st.markdown(f"{icon} **{agent}**")
        st.markdown(
            """
            <div style='font-size: 0.85em; color: #666; margin-top: 10px;'>
            Each agent specializes in different areas:
            • Classification routes your question
            • Department agents handle specific topics
            • Technical agents interact with systems
            </div>
            """,
            unsafe_allow_html=True,
        )


def display_chat_fragment() -> "None":
    """
    displays chat messages for the selected conversation.
    Updates during active workflows are driven by conditional polling in main().
    """
    if not st.session_state.get("selected_submission"):
        return

    conversation_id = st.session_state.selected_submission
    conversation_exchanges = st.session_state.conversations.get(conversation_id, [])

    for exchange in conversation_exchanges:
        with st.chat_message("user"):
            st.write(exchange.get("input", ""))

        # get current workflow state and render agent response
        submission_id = exchange.get("submission_id", "")
        current_state = submission_states.get(submission_id, exchange)

        _render_exchange_response(current_state)


def main() -> "None":
    st.set_page_config(
        page_title="Agentic AI Workflow",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # check OGX health on first load
    ogx_status = get_ogx_status()
    if not ogx_status["checked"]:
        ogx_status = check_ogx_health()

    # ingestion state runs automatically on startup
    ingestion_state = get_ingestion_state()

    with st.sidebar:
        st.title("⚙️ Configuration")

        # show OGX connection status
        if ogx_status["checked"]:
            if ogx_status["connected"]:
                st.success("**OGX:** Connected")
            else:
                st.error("**OGX:** Disconnected")

        st.markdown(f"**Inference Model:** `{INFERENCE_MODEL}`")
        st.markdown(f"**Guardrail Model:** `{GUARDRAIL_MODEL}`")
        st.markdown(f"**Guardrail Mode:** `{GUARDRAIL_MODE}`")
        st.markdown(f"**MCP Tool Model:** `{MCP_TOOL_MODEL}`")
        st.divider()

        # render OGX errors
        has_blocking_error = render_ogx_errors()
        if has_blocking_error:
            st.divider()

        st.subheader("📦 Ingestion Status")

        if "vector_store_count" not in ingestion_state:
            ingestion_state["vector_store_count"] = count_vector_stores()

        vector_store_count = ingestion_state.get("vector_store_count", 0)

        if ingestion_state["status"] == "pending":
            st.info("🔍 Checking...")
        elif ingestion_state["status"] == "running":
            st.info("⏳ Running...")
        elif ingestion_state["status"] == "completed":
            st.success(f"✅ Completed ({ingestion_state['ingested_count']} items)")
        elif ingestion_state["status"] == "skipped":
            st.info("⏭️ Skipped")
        elif ingestion_state["status"] == "error":
            st.error("❌ Failed")

        if vector_store_count > 0:
            st.metric("Vector Stores in Database", vector_store_count)
        st.divider()

    # block workflow if OGX is not connected
    if not ogx_status["connected"]:
        st.title("🤖 Agentic AI Workflow - Connection Error")
        st.error(
            "**Cannot start workflow: OGX is not available**\n\n"
            f"{ogx_status['error_message']}\n\n"
            "Please check the sidebar for details on how to resolve this issue."
        )
        return

    # block workflow if required models are not available
    if ogx_status.get("missing_models"):
        missing_list = ", ".join(ogx_status["missing_models"])
        st.title("🤖 Agentic AI Workflow - Model Error")
        st.error(
            "**Cannot start workflow: Required models are not available**\n\n"
            f"Missing models: {missing_list}\n\n"
            "Please check the sidebar for details on how to resolve this issue."
        )
        return

    # block workflow until ingestion complete
    if ingestion_state["status"] in ("pending", "running"):
        st.title("🤖 Agentic AI Workflow - Initializing")

        if ingestion_state["status"] == "running":
            st.info("⏳ Running data ingestion pipeline... Please wait.")
        else:
            # status is "pending", trigger automatic check (will transition
            # to "skipped" or "running")
            st.info("🔍 Checking vector stores...")
            # using _ingestion_lock here to block multiple streamlit sessions
            # ingesting in parallel. First session gets the lock and creates
            # vector stores. All other sessions will get the lock only after
            # that's completed, check again (stores now exist), and finally
            # will skip ingestion.
            with ingestion_lock:
                loop = get_or_create_event_loop()
                loop.run_until_complete(check_and_run_ingestion_if_needed())
            st.rerun()
            return

        time.sleep(1.0)
        st.rerun()
        return

    pipelines = ingestion_state.get("pipelines")
    if pipelines is None:
        logger.info("Pipelines not available, parsing from ingestion config")
        ingestion_service = get_ingestion_service()
        pipelines = ingestion_service.pipelines
        logger.info(f"Loaded {len(pipelines)} pipelines from config")

    # initialize workflow once (cached) - builds full agent graph
    if "workflow" not in st.session_state:
        workflow, rag_service = initialize_workflow(pipelines)
        st.session_state.workflow = workflow
        vector_store_count = len(rag_service.all_vector_store_ids)
        ingestion_state["vector_store_count"] = vector_store_count
        logger.info(f"Vector stores in database: {vector_store_count}")
    else:
        workflow = st.session_state.workflow

    with st.sidebar:
        display_sidebar_conversations()

    st.title("🤖 Agentic AI Workflow")
    st.markdown(
        """
    Submit your questions and track their processing in real-time.
    Multiple submissions can run concurrently.
    """
    )

    if "workflow" in st.session_state and st.session_state.workflow is not None:
        with st.expander("🔀 View Workflow Graph"):
            try:
                graph_ascii = st.session_state.workflow.get_graph().draw_ascii()
                st.code(graph_ascii, language="text")
            except Exception as e:
                st.error(f"Could not display graph: {e}")

    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

    # render chat messages for selected conversation
    st.button("🔄 Refresh Chat", help="Manual Chat Refresh to see latest updates")
    display_chat_fragment()

    # chat input at bottom (standard chat interface pattern)
    question = st.chat_input("Ask a question...", key="chat_input")

    if question:
        # determine conversation ID (existing or new)
        if st.session_state.selected_submission:
            # add to existing conversation
            conversation_id = st.session_state.selected_submission
            exchange_index = len(
                st.session_state.conversations.get(conversation_id, [])
            )
        else:
            # create new conversation
            conversation_id = str(uuid.uuid4())
            exchange_index = 0
            if "active_submissions" not in st.session_state:
                st.session_state.active_submissions = []
            # add to top of conversation list
            st.session_state.active_submissions.insert(0, conversation_id)
            st.session_state.selected_submission = conversation_id
            st.session_state.conversations[conversation_id] = []

        submission_id = str(uuid.uuid4())

        # initialize workflow state for this exchange
        exchange_state: "WorkflowState" = {
            "input": question,
            "submission_id": submission_id,
            "conversation_id": conversation_id,
            "exchange_index": exchange_index,
            "decision": "",
            "classification_message": "Processing...",
            "workflow_complete": False,
            "mcp_output": "",
            "github_issue": "",
            "rag_sources": [],
            "messages": [],
            "namespace": "",
            "data": "",
            "agent_timings": {},
            "rag_query_time": 0.0,
            "active_agent": "",
            "status_message": "",
            "status_history": [],
        }

        # store exchange in conversation and global state
        st.session_state.conversations[conversation_id].append(exchange_state)
        submission_states[submission_id] = exchange_state

        # submit async workflow task
        submit_workflow_task(workflow, question, submission_id)
        # rerun to update sidebar with new conversation
        st.rerun()

    # fragment polls for background state changes
    # note:: only active while workflows are running;
    # renders nothing visible.
    if has_active_workflows():
        _poll_for_updates()


@st.fragment(run_every="0.3s")
def _poll_for_updates() -> "None":
    """
    fragment that bridges background agent threads and the UI
    checks if any agent has written a state update and triggers a full page
    rerun only when one is detected

    does not block the main thread, so widget interactions remain responsive
    """
    if submission_states.update_event.is_set():
        submission_states.update_event.clear()
        st.rerun()


if __name__ == "__main__":
    main()
