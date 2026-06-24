from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union

from ollama import ChatResponse, Tool, chat
from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue

OBSIDIAN_PATH_DIR = Path().home() / "Documents" / "Exocortex"

class Conversation:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})


class OllamaClient:
    def generate(self, agent: Agent, conversation: Conversation):
        messages = [{"role": "system", "content": agent.system_prompt}]

        messages.extend(conversation.messages)

        return chat(
            model=agent.llm,
            messages=messages,
            options={"temperature": agent.temp},
            format=agent.format,
            think=agent.thinking_mode,
            stream=agent.stream_mode,
            tools=agent.tools
        )


class Agent:
    def __init__(
        self,
        temp: float,
        system_prompt: str,
        thingking_mode: bool,
        stream_mode: bool,
        tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
        format: Optional[Union[Literal["", "json"], JsonSchemaValue]] = None,
        llm: str = "lfm2.5:latest",
    ) -> None:
        self.temp = temp
        self.system_prompt = system_prompt
        self.thinking_mode = thingking_mode
        self.stream_mode = stream_mode
        self.tools = tools
        self.format: JsonSchemaValue | Literal["", "json"] | None = format
        self.llm = llm


def list_vault() -> list[str]:
    """
    list all file in obsidian vault
    """
    all_files: list[str] = []
    for item in OBSIDIAN_PATH_DIR.iterdir():
        if not item.name.startswith(".") and not item.name.startswith("_"):
            all_files.append(item.name)
    return all_files


def main():
    available_functions = {"list_vault": list_vault}
    agent_tool = Agent(
        temp=0.1,
        system_prompt="you are a tool calling agent",
        thingking_mode=False,
        stream_mode=True,
        tools=[list_vault],
        format='json',
        llm="qwen3.5:2b",
    )
    
    user_chat = str(input("test: "))

    messages = Conversation()
    messages.add_user(user_chat)
    print(messages.messages)
    # response_main_agent = agent.ollama_chat(
    #     messages=messages, stream_mode=True, tools=[list_vault]
    # )

    # tool_output = {
    #     "role": "tool",
    #     "content": "",
    #     "tool_name": "",
    # }

    # for part in response_main_agent:
    #     if part.message.thinking:
    #         print(part.message.thinking, end="", flush=True)
    #     if part.message.content:
    #         print(part.message.content, end="", flush=True)
    #     if part.message.tool_calls:
    #         print("")
    #         for tool in part.message.tool_calls:
    #             if function_to_call := available_functions.get(tool.function.name):
    #                 print(
    #                     f"Calling function {tool.function.name} with argument {tool.function.arguments}"
    #                 )
    #                 output = function_to_call(**tool.function.arguments)
    #                 print(f"Function output > {output} \n")
    #                 messages.append(part.message)
    #                 tool_output.update(
    #                     {"content": f"{output}", "tool_name": f"{tool.function.name}"}
    #                 )
    #             else:
    #                 print(f"Function {tool.function.name} not found")

    # messages.append(tool_output)
    # print(messages)
    # print("-" * 20, "Sending back to agent", "-" * 20)
    # if any(msg.get("role") == "tool" for msg in messages):
    #     res_agent = Agent(
    #         temp=0.2,
    #         system_prompt="you are a helpfull assistant, answer the question with simple sentence",
    #         llm="qwen3.5:2b",
    #     )
    #     res = response_main_agent = res_agent.ollama_chat(
    #         messages=messages, stream_mode=True, tools=[list_vault], thinking_mode=False
    #     )
    #     done_thinking = False
    #     for part in res:
    #         if part.message.thinking:
    #             print(part.message.thinking, end="", flush=True)
    #         if part.message.content:
    #             if not done_thinking:
    #                 print("\n----- Final result:")
    #                 done_thinking = True
    #             print(part.message.content, end="", flush=True)
    #         if part.message.tool_calls:
    #             print("")
    #             print("Model returned tool calls")
    #             print(part.message.tool_calls)

    #     print("=" * 20, "final messages", "=" * 20)
    #     print(messages)
    # else:
    #     print("No tool calls returned")


if __name__ == "__main__":
    main()
