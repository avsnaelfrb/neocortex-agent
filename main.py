from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Any
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


# class ToolRegister():
#     def __init__(self, tool_name: str, description: str, ) -> None:
#         pass


async def list_vault() -> list[str]:
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

    response_intent_agent = context_agent.ollama_chat(
        user_input=user_chat,
        stream_mode=False,
        thinking_mode=False,
        tools=[list_vault]
    )
    response_main_agent = agent.ollama_chat(user_input=response_intent_agent.message.content, stream_mode=True)
    for part in response_main_agent:
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


if __name__ == "__main__":
    main()
