from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union

from ollama import Tool
from pydantic.json_schema import JsonSchemaValue


class InferenceProfile:
    def __init__(
        self,
        temp: float,
        system_prompt: str,
        thingking_mode: bool,
        stream_mode: bool,
        tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
        format: Optional[Union[Literal["", "json"], JsonSchemaValue]] = None,
        llm: str = "smollm2:1.7b",
    ) -> None:
        self.temp = temp
        self.system_prompt = system_prompt
        self.thinking_mode = thingking_mode
        self.stream_mode = stream_mode
        self.tools = tools
        self.format: JsonSchemaValue | Literal["", "json"] | None = format
        self.llm = llm
