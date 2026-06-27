from collections.abc import Iterator, Sequence
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.tools import BaseTool, tool
from langchain_ollama import ChatOllama
from pydantic.json_schema import JsonSchemaValue

from agent_tools import AgentTools


class Conversation:
    def __init__(self) -> None:
        self.messages: list[BaseMessage] = []

    def add_user(self, content: str) -> None:
        self.messages.append(HumanMessage(content=content))

    def add_assistant(self, message: AIMessage) -> None:
        self.messages.append(message)

    def add_tool(self, content: Any, tool_call_id: str, tool_name: str) -> None:
        self.messages.append(
            ToolMessage(
                content=str(content),
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        )

    def dump(self) -> list[dict[str, Any]]:
        return [message.model_dump() for message in self.messages]


class StreamAccumulator:
    def __init__(self) -> None:
        self.chunk: Optional[AIMessageChunk] = None
        self.content = ""

    def append_iter(self, chunk: AIMessageChunk) -> None:
        self.chunk = chunk if self.chunk is None else self.chunk + chunk

        if chunk.content:
            text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            self.content += text
            print(text, end="", flush=True)

    def to_message(self) -> AIMessage:
        if self.chunk is None:
            return AIMessage(content="")
        return AIMessage(
            content=self.chunk.content,
            tool_calls=self.chunk.tool_calls,
            invalid_tool_calls=self.chunk.invalid_tool_calls,
            response_metadata=self.chunk.response_metadata,
            usage_metadata=self.chunk.usage_metadata,
        )


class LangChainClient:
    def generate(self, agent: "Agent", conversation: Conversation) -> AIMessage | Iterator[AIMessageChunk]:
        messages: list[BaseMessage] = [SystemMessage(content=agent.system_prompt)]
        messages.extend(conversation.messages)

        llm = ChatOllama(
            model=agent.llm,
            temperature=agent.temp,
            reasoning=agent.thinking_mode,
            format=agent.format,
        )

        if agent.tools:
            llm = llm.bind_tools(agent.tools)

        if agent.stream_mode:
            return llm.stream(messages)

        return llm.invoke(messages)


class Agent:
    def __init__(
        self,
        temp: float,
        system_prompt: str,
        thinking_mode: bool,
        stream_mode: bool,
        tools: Optional[Sequence[BaseTool]] = None,
        format: Optional[Literal["", "json"] | JsonSchemaValue] = None,
        llm: str = "lfm2.5:latest",
    ) -> None:
        self.temp = temp
        self.system_prompt = system_prompt
        self.thinking_mode = thinking_mode
        self.stream_mode = stream_mode
        self.tools = tools
        self.format = format
        self.llm = llm


def build_tools(agent_tools: AgentTools) -> tuple[list[BaseTool], dict[str, BaseTool]]:
    @tool
    def list_vault() -> dict[str, list[str] | int]:
        """List all visible files in the user's Obsidian vault."""
        return agent_tools.list_vault()

    @tool
    def search_file(keyword: str) -> list[str]:
        """Search files in the user's Obsidian vault by one keyword."""
        return agent_tools.search_file(keyword=keyword)

    tools = [list_vault, search_file]
    return tools, {item.name: item for item in tools}


def collect_stream(response: Iterator[AIMessageChunk]) -> AIMessage:
    result = StreamAccumulator()
    for part in response:
        result.append_iter(part)
    return result.to_message()


def main() -> None:
    while True:
        try:
            raw_tools = AgentTools()
            langchain_tools, available_tools = build_tools(raw_tools)

            agent_tool = Agent(
                temp=0.1,
                system_prompt="You are a tool calling agent. Use tools when the user asks about vault files.",
                thinking_mode=False,
                stream_mode=True,
                tools=langchain_tools,
                llm="qwen3.5:2b",
            )

            conversation = Conversation()

            print("-" * 64)
            user_chat = str(input("Send question :> "))
            print("-" * 64, "\n")

            conversation.add_user(user_chat)

            response = LangChainClient().generate(agent=agent_tool, conversation=conversation)
            assistant_message = collect_stream(response) if agent_tool.stream_mode else response
            conversation.add_assistant(assistant_message)

            for tool_call in assistant_message.tool_calls:
                selected_tool = available_tools.get(tool_call["name"])
                if selected_tool is None:
                    continue

                output = selected_tool.invoke(tool_call["args"])
                conversation.add_tool(
                    content=output,
                    tool_call_id=tool_call["id"],
                    tool_name=tool_call["name"],
                )

            if any(isinstance(message, ToolMessage) for message in conversation.messages):
                print("\n")
                print("-" * 20, "Sending back to agent", "-" * 20, "\n")

                agent_response = Agent(
                    temp=0.2,
                    system_prompt="You are a response assistant, answer with simple sentences.",
                    thinking_mode=False,
                    stream_mode=True,
                    llm="qwen3.5:0.8b",
                )

                final_response = LangChainClient().generate(agent=agent_response, conversation=conversation)
                final_message = collect_stream(final_response) if agent_response.stream_mode else final_response
                conversation.add_assistant(final_message)

                print("\n\n------------------- Final Conversation ------------------- \n")
                print(conversation.dump())
                print("\nPanjang conversation: ", len(conversation.messages))
            else:
                print("No tool calls returned")
        except KeyboardInterrupt:
            print("\n\nbyee -----")
            break


if __name__ == "__main__":
    main()
