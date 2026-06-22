from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from ollama import ChatResponse, chat
from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue

OBSIDIAN_PATH_DIR = Path().home() / "Documents" / "Exocortex"


class Agent:
    def __init__(
        self, temp: float, system_prompt: str, llm: str = "lfm2.5:latest"
    ) -> None:
        self.temp = temp
        self.system_prompt = system_prompt
        self.llm = llm

    def build_message(self, user_content: str) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": f"{self.system_prompt}"},
            {"role": "user", "content": f"{user_content}"},
        ]
        return messages

    def ollama_chat(
        self,
        messages: list[dict[str, str]],
        format: JsonSchemaValue | Literal["", "json"] | None = None,
        tools=None,
        thinking_mode: bool = True,
        stream_mode: bool = False,
    ) -> ChatResponse | Iterator[ChatResponse]:

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

            return response
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

            return response


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
    agent = Agent(
        temp=0.1,
        system_prompt="you are a tool calling agent",
        llm="qwen3.5:2b",
    )
    user_chat = str(input("test: "))
    messages = agent.build_message(user_content=user_chat)
    print(messages)
    response_main_agent = agent.ollama_chat(
        messages=messages, stream_mode=True, tools=[list_vault]
    )

    tool_output = {
        "role": "tool",
        "content": "",
        "tool_name": "",
    }

    for part in response_main_agent:
        if part.message.thinking:
            print(part.message.thinking, end="", flush=True)
        if part.message.content:
            print(part.message.content, end="", flush=True)
        if part.message.tool_calls:
            print("")
            for tool in part.message.tool_calls:
                if function_to_call := available_functions.get(tool.function.name):
                    print(
                        f"Calling function {tool.function.name} with argument {tool.function.arguments}"
                    )
                    output = function_to_call(**tool.function.arguments)
                    print(f"Function output > {output} \n")
                    messages.append(part.message)
                    tool_output.update(
                        {"content": f"{output}", "tool_name": f"{tool.function.name}"}
                    )
                else:
                    print(f"Function {tool.function.name} not found")

    messages.append(tool_output)
    print(messages)
    print("-" * 20, "Sending back to agent", "-" * 20)
    if any(msg.get("role") == "tool" for msg in messages):
        res_agent = Agent(temp=0.2, system_prompt="you are a helpfull assistant, answer the question with simple sentence", llm="qwen3.5:2b ")
        res = response_main_agent = res_agent.ollama_chat(
            messages=messages, stream_mode=True, tools=[list_vault], thinking_mode=False
        )
        done_thinking = False
        for part in res:
            if part.message.thinking:
                print(part.message.thinking, end="", flush=True)
            if part.message.content:
                if not done_thinking:
                    print("\n----- Final result:")
                    done_thinking = True
                print(part.message.content, end="", flush=True)
            if part.message.tool_calls:
                print("")
                print("Model returned tool calls")
                print(part.message.tool_calls)

        print("="*20, "final messages", "="*20)
        print(messages)
    else:
        print("No tool calls returned")


if __name__ == "__main__":
    main()
