from collections.abc import Iterator
from typing import Any, Literal

from ollama import ChatResponse, chat
from pydantic import BaseModel

from agent_tools import AgentTools
from agents_core import InferenceProfile
from logging_config import get_logger


logger = get_logger(__name__)


class Conversation:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str, tool_calls: list[dict[str, Any]] | None):
        self.messages.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls}
        )

    def add_tool(self, content: Any, func_name, func_arg):
        self.messages.append(
            {
                "role": "tool",
                "content": content,
                "tool_calls": [
                    {"function": {"name": func_name, "arguments": func_arg}}
                ],
            }
        )


class StreamAccumulator:
    def __init__(self) -> None:
        self.content = ""
        self.thinking = ""
        self.tool_calls: list[dict[str, Any]] = []

    def append_iter(self, chunk: Iterator[ChatResponse]):
        if chunk.message.thinking:
            self.thinking += chunk.message.thinking
            print(chunk.message.thinking, end="", flush=True)
        if chunk.message.content:
            self.content += chunk.message.content
            print(chunk.message.content, end="", flush=True)
        if chunk.message.tool_calls:
            self.tool_calls.extend(chunk.message.tool_calls)


class OllamaClient:
    def generate(self, profile: InferenceProfile, conversation: Conversation):
        messages = [{"role": "system", "content": profile.system_prompt}]

        messages.extend(conversation.messages)

        result = chat(
            model=profile.llm,
            messages=messages,
            options={"temperature": profile.temp},
            format=profile.format,
            think=profile.thinking_mode,
            stream=profile.stream_mode,
            tools=profile.tools,
        )

        if profile.stream_mode:
            streamAcc = StreamAccumulator()
            for part in result:
                streamAcc.append_iter(part)
            return (streamAcc.content, streamAcc.thinking, streamAcc.tool_calls)
        else:
            return result.message


def main():
    logger.info("Application started")

    while True:
        try:
            tools = AgentTools()
            available_functions = {
                "list_vault": tools.list_vault,
                "search_file": tools.search_file,
            }
            agent_tool = InferenceProfile(
                temp=0.1,
                system_prompt="you are a assistant agent with tool calling capabilities, answer with simple sentences.",
                thingking_mode=False,
                stream_mode=True,
                tools=[tools.list_vault, tools.search_file],
                llm="qwen3.5:2b",
            )
            messages = Conversation()

            print("\n")
            print("-"*64)
            user_chat = str(input("Send question :> "))
            print("-"*64,"\n")

            messages.add_user(user_chat)
            logger.info("User input received: %s", user_chat)

            response = OllamaClient().generate(
                profile=agent_tool, conversation=messages
            )

            content, thinking, tool_calls = response
            messages.add_assistant(content=str(content), tool_calls=tool_calls)
            logger.info(
                "Agent response received (content_length=%d, tool_calls=%d)",
                len(str(content)),
                len(tool_calls),
            )

            for tool in tool_calls:
                if function_to_calls := available_functions.get(tool.function.name):
                    print(f"calling funtion tool {tool.function.name}\n")
                    logger.info(
                        "Calling tool %s with arguments %s",
                        tool.function.name,
                        tool.function.arguments,
                    )
                    output_tool = function_to_calls(**tool.function.arguments)
                    logger.debug(
                        "Tool %s returned: %s", tool.function.name, output_tool
                    )
                    messages.add_tool(
                        content=str(output_tool),
                        func_name=tool.function.name,
                        func_arg=tool.function.arguments,
                    )

            if any(msg.get("role") == "tool" for msg in messages.messages):
                print("==> Sending back to agent <==\n")

                res = OllamaClient().generate(profile=agent_tool, conversation=messages)
                content_res, thinking_res, tool_calls_res = res
                messages.add_assistant(
                    content=str(content_res), tool_calls=tool_calls_res
                )
                logger.info(
                    "Final agent response received (content_length=%d)",
                    len(str(content_res)),
                )

            else:
                logger.info("Agent response completed without tool calls")
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
            print("\n")
            print("==> byee <==")
            break
        except Exception:
            logger.exception("Application stopped because of an unexpected error")
            raise


if __name__ == "__main__":
    main()
