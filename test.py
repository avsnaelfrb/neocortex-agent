import inspect
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from ollama import ChatResponse, chat
from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue

OBSIDIAN_PATH_DIR = Path().home() / "Documents" / "Exocortex"
LOGGER = logging.getLogger("neocortex")


def configure_logging() -> None:
    log_level_name = os.getenv("NEOCORTEX_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def preview_text(value: Any, limit: int = 160) -> str:
    text = str(value).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def summarize_tools(tools: list[Any] | None) -> list[str]:
    if not tools:
        return []

    summary: list[str] = []
    for tool in tools:
        tool_name = getattr(tool, "__name__", repr(tool))
        tool_kind = "async" if inspect.iscoroutinefunction(tool) else "sync"
        summary.append(f"{tool_name}<{tool_kind}>")
    return summary


def extract_tool_calls(message: Any) -> list[str]:
    tool_calls = getattr(message, "tool_calls", None) or []
    summary: list[str] = []

    for tool_call in tool_calls:
        function = getattr(tool_call, "function", None)
        function_name = getattr(function, "name", "unknown")
        arguments = getattr(function, "arguments", None)
        summary.append(f"{function_name} args={arguments}")

    return summary


def log_response_metadata(
    logger: logging.Logger, response: ChatResponse, label: str
) -> None:
    message = getattr(response, "message", None)
    content = getattr(message, "content", "") or ""
    tool_calls = extract_tool_calls(message)

    logger.info(
        "%s completed | done=%s | done_reason=%s | content_len=%s | tool_calls=%s | "
        "prompt_eval_count=%s | eval_count=%s",
        label,
        getattr(response, "done", None),
        getattr(response, "done_reason", None),
        len(content),
        tool_calls or "-",
        getattr(response, "prompt_eval_count", None),
        getattr(response, "eval_count", None),
    )
    logger.info("%s content preview: %s", label, preview_text(content))


class Agent:
    def __init__(
        self, temp: float, system_prompt: str, llm: str = "lfm2.5:latest"
    ) -> None:
        self.temp = temp
        self.system_prompt = system_prompt
        self.llm = llm
        logger_name = self.llm.replace(":", "_").replace("/", "_")
        self.logger = logging.getLogger(f"neocortex.agent.{logger_name}")

    def _stream_with_logging(
        self, response: Iterator[ChatResponse]
    ) -> Iterator[ChatResponse]:
        def iterator() -> Iterator[ChatResponse]:
            chunk_count = 0
            saw_content = False
            saw_tool_calls = False

            for chunk_count, part in enumerate(response, start=1):
                message = getattr(part, "message", None)
                content = getattr(message, "content", "") or ""
                tool_calls = extract_tool_calls(message)

                if content:
                    saw_content = True
                if tool_calls:
                    saw_tool_calls = True

                self.logger.info(
                    "stream chunk=%s | content_len=%s | tool_calls=%s | done=%s | done_reason=%s",
                    chunk_count,
                    len(content),
                    tool_calls or "-",
                    getattr(part, "done", None),
                    getattr(part, "done_reason", None),
                )
                yield part

            self.logger.info(
                "stream finished | chunks=%s | saw_content=%s | saw_tool_calls=%s",
                chunk_count,
                saw_content,
                saw_tool_calls,
            )

        return iterator()

    def ollama_chat(
        self,
        user_input: str | Any | None,
        format: JsonSchemaValue | Literal["", "json"] | None = None,
        tools=None,
        thinking_mode: bool = True,
        stream_mode: bool = False,
    ) -> ChatResponse | Iterator[ChatResponse]:

        messages = [
            {"role": "system", "content": f"{self.system_prompt}"},
            {"role": "user", "content": f"{user_input}"},
        ]
        tool_summary = summarize_tools(tools)
        self.logger.info(
            "dispatch chat | model=%s | stream=%s | think=%s | temp=%s | format=%s | tools=%s",
            self.llm,
            stream_mode,
            thinking_mode,
            self.temp,
            format,
            tool_summary or "-",
        )
        self.logger.info("system prompt preview: %s", preview_text(self.system_prompt))
        self.logger.info("user input preview: %s", preview_text(user_input))

        if tool_summary:
            async_tools = [tool for tool in tool_summary if tool.endswith("<async>")]
            if async_tools:
                self.logger.warning(
                    "async tools registered: %s | current code does not execute tool calls automatically",
                    async_tools,
                )

        if stream_mode:
            response = chat(
                model=self.llm,
                messages=messages,
                options={"temperature": self.temp},
                stream=True,
                tools=tools,
                think=thinking_mode,
                format=format,
            )

            return self._stream_with_logging(response)
        else:
            response = chat(
                model=self.llm,
                messages=messages,
                options={"temperature": self.temp},
                stream=False,
                tools=tools,
                think=thinking_mode,
                format=format,
            )
            log_response_metadata(self.logger, response, "chat")

            return response


# class ToolRegister():
#     def __init__(self, tool_name: str, description: str, ) -> None:
#         pass


def list_vault() -> list[str]:
    """
    list all file in obsidian vault
    """
    all_files: list[str] = []
    for item in OBSIDIAN_PATH_DIR.iterdir():
        if not item.name.startswith(".") and not item.name.startswith("_"):
            all_files.append(item.name)
    return all_files


# available_functions = {"list_vault": list_vault}


def main():
    configure_logging()
    LOGGER.info("starting neocortex debug session")

    agent = Agent(temp=0.1, system_prompt="you are a helpfull assistant")
    context_agent = Agent(
        temp=0.0,
        system_prompt="""
        you are agent for summarize context for another agent, intent the user question and use the available tools

        only response laike this:

        CONTEXT:
            (summarize the context)
        USER_QUESTION:
            (pass the user question)
        """,
    )

    user_chat = str(input("test: "))
    LOGGER.info(
        "received user input | chars=%s | preview=%s",
        len(user_chat),
        preview_text(user_chat),
    )

    response_intent_agent = context_agent.ollama_chat(
        user_input=user_chat, stream_mode=False, thinking_mode=False, tools=[list_vault]
    )
    intent_tool_calls = extract_tool_calls(response_intent_agent.message)
    if intent_tool_calls:
        LOGGER.warning(
            "context agent returned tool calls, but current flow does not execute them: %s",
            intent_tool_calls,
        )

    intent_content = getattr(response_intent_agent.message, "content", "") or ""
    LOGGER.info(
        "forwarding context output to main agent | preview=%s",
        preview_text(intent_content),
    )
    LOGGER.warning(
        "main agent is invoked without tools; only the context agent can request tools in this flow"
    )
    response_main_agent = agent.ollama_chat(
        user_input=response_intent_agent.message.content, stream_mode=True
    )

    streamed_chunks = 0
    for part in response_main_agent:
        streamed_chunks += 1
        # if part.message.tool_calls:
        #     for tool in part.message.tool_calls:
        #         function_to_call = available_functions.get(tool.function.name)
        #         if function_to_call:
        #             print(f"\nCalling Tools {tool.function.name}")
        #             return
        #         else:
        #             print(f"Function {tool.function.name} not found")
        #             return

        print(part.message.content, end="", flush=True)

    LOGGER.info("main agent stream loop finished | printed_chunks=%s", streamed_chunks)


if __name__ == "__main__":
    main()
