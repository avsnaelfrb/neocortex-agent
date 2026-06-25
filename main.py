from collections.abc import Iterator
from pathlib import Path
import subprocess
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union

from ollama import ChatResponse, Tool, chat
from pydantic.json_schema import JsonSchemaValue

HOME_DIR = Path().home() 
OBSIDIAN_PATH_DIR = HOME_DIR / "Documents" / "Exocortex"


class Conversation:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str, tool_calls: list[dict[str, Any]]):
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

    def add_tool(self, content: Any, func_name, func_arg):
        self.messages.append(
            {"role": "tool", "content": content, "tool_calls": [{"function": {"name": func_name, "arguments": func_arg}}]}
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
            tools=agent.tools,
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


def list_vault() -> dict[str, list[str] | int]:
    """
    desc: list all files in obisidian vault user
    output: list of string, string of files name
    args: no arguments
    """
    FILES_VAULT: list[str] = []
    for file in OBSIDIAN_PATH_DIR.iterdir():
        if not file.name.startswith(".") and not file.name.startswith("_"):
            FILES_VAULT.append(file.name)

    return {"all_files": FILES_VAULT, "total_files": len(FILES_VAULT)}

def search_file(keyword: str) -> list[str]:
    """
    desc: search files in obsidian vault user with one word argument
    output: return list of result files if similar with keyword argument
    args: one keyword for search file
    """
    process = subprocess.run(['fd', keyword, OBSIDIAN_PATH_DIR], capture_output=True, text=True).stdout
    list = process.strip().split("\n")
    clean_list = []
    for item in list:
        clean_list.append(Path(item).name)
            
    return clean_list

def main():
    available_functions = {"list_vault": list_vault, "search_file": search_file}
    agent_tool = Agent(
        temp=0.1,
        system_prompt="you are a tool calling agent",
        thingking_mode=True,
        stream_mode=False,
        tools=[list_vault, search_file],
        llm="qwen3.5:2b",
    )
    print("-"*64)
    user_chat = str(input("Send question :> "))
    print("-"*64, "\n")

    messages = Conversation()
    messages.add_user(user_chat)
    # print(messages.messages)

    response = OllamaClient().generate(agent=agent_tool, conversation=messages)

    result = response.message
    # result = StreamAccumulator()
    # for part in response:
    #     result.append_iter(chunk=part)

    messages.add_assistant(content=str(result.content), tool_calls=result.tool_calls)

    for tool in result.tool_calls:
        if function_to_calls := available_functions.get(tool.function.name):
            output_tool = function_to_calls(**tool.function.arguments)
            messages.add_tool(content=str(output_tool), func_name=tool.function.name, func_arg=tool.function.arguments)
            
            
    if any(msg.get("role") == "tool" for msg in messages.messages):
        print("\n")
        print("-" * 20, "Sending back to agent", "-" * 20, "\n")
        agent_response = Agent(
            temp=0.2,
            system_prompt="you are a helpfull assistant",
            thingking_mode=False,
            stream_mode=True,
            llm="qwen3.5:2b",
        )
        res = OllamaClient().generate(agent=agent_response, conversation=messages)
        final_result = StreamAccumulator()
        for part in res:
            final_result.append_iter(chunk=part)
        messages.add_assistant(content=final_result.content, tool_calls=final_result.tool_calls) 
            # if part.message.thinking:
            #     print(part.message.thinking, end="", flush=True)
            # if part.message.content:
            #     if not done_thinking:
            #         print("\n\n-------------- Final result --------------\n")
            #         done_thinking = True
            #     print(part.message.content, end="", flush=True)
            # if part.message.tool_calls:
            #     print("")
            #     print("Model returned tool calls")
            #     print(part.message.tool_calls)
                
        # print("\n", "=" * 20, "final messages", "=" * 20)
        # print(messages.messages)
        print("\n-------------- Final Conversation -------------- ")
        print(messages.messages)
        print("\nPanjang conversation: ",len(messages.messages))
    else:
        print("No tool calls returned")


if __name__ == "__main__":
    main()
