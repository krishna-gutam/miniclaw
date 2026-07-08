import os
import json
import traceback
from typing import Annotated
from typing_extensions import TypedDict, NotRequired

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
)

from tools.registry import get_tools

load_dotenv()

DEFAULT_SYSTEM_PROMPT = """You are an expert coding assistant. You help users with coding tasks by reading files, executing commands, editing code, and writing new files.

Permission rules:
- NEVER make changes automatically.
- Before any action that modifies files, creates files, overwrites files, installs packages, executes scripts, or runs shell/PowerShell commands, first explain:
  1. What you plan to do
  2. Why you want to do it
  3. Which files or commands will be affected

- Ask explicitly for user approval before proceeding.
- Wait for a clear confirmation such as "yes", "approve", or equivalent before executing any modifying action.
- If the user has already explicitly authorized a specific action in the current request, you may proceed only with that specific action.
- If multiple changes are required, summarize all planned changes and request approval once before making them.
- Be concise in your responses.
"""


def sanitize_content(content) -> str:
    """
    Safely extracts a string from LangChain's potential list-based content blocks.
    Some model providers may return content as lists of blocks instead of plain text.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        extracted = []
        for block in content:
            if isinstance(block, str):
                extracted.append(block)
            elif isinstance(block, dict):
                if "text" in block:
                    extracted.append(str(block["text"]))
                elif "content" in block:
                    extracted.append(str(block["content"]))
        return "\n".join(extracted)

    return str(content)


class State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    approved: NotRequired[bool]
    chat_id: NotRequired[str]


def clone_message_with_sanitized_content(message: BaseMessage) -> BaseMessage:
    """
    Avoid mutating messages stored in LangGraph checkpoint memory.
    Return a sanitized copy only when needed.
    """
    if isinstance(message.content, str):
        return message

    content = sanitize_content(message.content)

    if isinstance(message, AIMessage):
        return AIMessage(
            content=content,
            tool_calls=message.tool_calls,
            additional_kwargs=message.additional_kwargs,
            response_metadata=message.response_metadata,
            id=message.id,
            name=message.name,
        )

    if isinstance(message, HumanMessage):
        return HumanMessage(
            content=content,
            additional_kwargs=message.additional_kwargs,
            response_metadata=message.response_metadata,
            id=message.id,
            name=message.name,
        )

    if isinstance(message, SystemMessage):
        return SystemMessage(
            content=content,
            additional_kwargs=message.additional_kwargs,
            response_metadata=message.response_metadata,
            id=message.id,
            name=message.name,
        )

    if isinstance(message, ToolMessage):
        return ToolMessage(
            content=content,
            tool_call_id=message.tool_call_id,
            additional_kwargs=message.additional_kwargs,
            response_metadata=message.response_metadata,
            id=message.id,
            name=message.name,
        )

    return message


def get_executed_tool_call_ids(messages: list[BaseMessage]) -> set:
    """Returns IDs of tool calls that already have a corresponding ToolMessage."""
    executed_ids = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            executed_ids.add(message.tool_call_id)
    return executed_ids


def get_pending_tool_calls(messages: list[BaseMessage]) -> list:
    """
    Find tool calls produced by the assistant that have not yet received ToolMessage results.
    """
    executed_ids = get_executed_tool_call_ids(messages)
    pending = []

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_call_id = tool_call.get("id")
                if tool_call_id and tool_call_id not in executed_ids:
                    pending.append(tool_call)

    return pending


def format_tool_calls_for_approval(tool_calls: list[dict]) -> str:
    """
    Create user-readable approval text for Telegram.
    """
    lines = [
        "I need your approval before running the following tool action(s):",
        "",
    ]

    for index, tool_call in enumerate(tool_calls, start=1):
        name = tool_call.get("name", "unknown_tool")
        args = tool_call.get("args", {})

        try:
            formatted_args = json.dumps(args, indent=2, ensure_ascii=False)
        except Exception:
            formatted_args = str(args)

        lines.append(f"{index}. Tool: `{name}`")
        lines.append("Arguments:")
        lines.append(formatted_args)
        lines.append("")

    lines.append('Reply with "yes" or "approve" to continue.')

    return "\n".join(lines)


def create_agent():
    model_name = os.getenv("MODEL")

    if not model_name:
        raise ValueError("MODEL environment variable is not set.")

    model = ChatGoogleGenerativeAI(model=model_name)

    tools = get_tools()
    print(tools)
    tools_by_name = {tool.name: tool for tool in tools}

    model_with_tools = model.bind_tools(tools)

    def call_model(state: State):
        messages = state.get("messages", [])

        sanitized_messages: list[BaseMessage] = [
            SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
        ]

        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            sanitized_messages.append(clone_message_with_sanitized_content(message))

        response = model_with_tools.invoke(sanitized_messages)

        return {
            "messages": [response],
            "approved": False,
        }

    def execute_pending_tools(state: State):
        messages = state.get("messages", [])
        pending_tool_calls = get_pending_tool_calls(messages)
        chat_id = state.get("chat_id", "")

        tool_messages = []

        for tool_call in pending_tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            if not tool_call_id:
                continue

            # Inject chat_id for our new tools
            if tool_name in ("set_reminder", "create_recurring_reminder") and "chat_id" not in tool_args:
                tool_args["chat_id"] = chat_id

            if tool_name not in tools_by_name:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: unknown tool '{tool_name}'.",
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            tool = tools_by_name[tool_name]

            try:
                result = tool.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(
                        content=sanitize_content(result),
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )
            except Exception as exc:
                error_text = (
                    f"Tool '{tool_name}' failed.\n\n"
                    f"Error: {exc}\n\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )

                tool_messages.append(
                    ToolMessage(
                        content=error_text,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )

        return {
            "messages": tool_messages,
            "approved": False,
        }

    def route_from_start(state: State):
        """
        If the Telegram layer says the user approved and there are pending tools,
        execute pending tools first.

        Otherwise, call the model normally.
        """
        approved = state.get("approved", False)
        messages = state.get("messages", [])

        if approved and get_pending_tool_calls(messages):
            return "tools"

        return "agent"

    def route_after_agent(state: State):
        """
        Do not automatically execute tools.
        If the model asks for tools, stop and let Telegram ask for approval.
        """
        return END

    workflow = StateGraph(State)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_pending_tools)

    workflow.add_conditional_edges(
        START,
        route_from_start,
        {
            "agent": "agent",
            "tools": "tools",
        },
    )

    workflow.add_edge("tools", "agent")

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            END: END,
        },
    )

    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


graph = create_agent()