from collections.abc import Iterator
from typing import Any, Literal

from ollama import ChatResponse, chat
from pydantic import BaseModel

from agent_tools import AgentTools
from agents_core import InferenceProfile


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
    while True:
        try:
            tools = AgentTools()
            available_functions = {
                "list_vault": tools.list_vault,
                "search_file": tools.search_file,
            }
            agent_tool = InferenceProfile(
                temp=0.1,
                system_prompt="you are a tool calling agent, answer with simple sentences.",
                thingking_mode=False,
                stream_mode=True,
                tools=[tools.list_vault, tools.search_file],
                llm="qwen3.5:2b",
            )
            messages = Conversation()

            print("-" * 64)
            user_chat = str(input("Send question :> "))
            print("-" * 64, "\n")

            response = OllamaClient().generate(
                profile=agent_tool, conversation=messages
            )

            messages.add_user(user_chat)

            content, thinking, tool_calls = response
            messages.add_assistant(content=str(content), tool_calls=tool_calls)

            for tool in tool_calls:
                if function_to_calls := available_functions.get(tool.function.name):
                    print(f"calling funtion tool {tool.function.name}")
                    output_tool = function_to_calls(**tool.function.arguments)
                    messages.add_tool(
                        content=str(output_tool),
                        func_name=tool.function.name,
                        func_arg=tool.function.arguments,
                    )

            if any(msg.get("role") == "tool" for msg in messages.messages):
                print("\n")
                print("-" * 20, "Sending back to agent", "-" * 20, "\n")

                res = OllamaClient().generate(profile=agent_tool, conversation=messages)
                content_res, thinking_res, tool_calls_res = res
                messages.add_assistant(
                    content=str(content_res), tool_calls=tool_calls_res
                )

                print(
                    "\n\n------------------- Final Conversation ------------------- \n"
                )
                print(messages.messages)
                print("\nPanjang conversation: ", len(messages.messages))
            else:
                print("\n\nNo tool calls returned")
        except KeyboardInterrupt:
            print("\n\nbyee -----")
            break


if __name__ == "__main__":
    main()
