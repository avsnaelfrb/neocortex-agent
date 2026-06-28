from collections.abc import Callable
from pathlib import Path
import subprocess
import inspect
from typing import Any

HOME_DIR = Path().home()
OBSIDIAN_PATH_DIR = HOME_DIR / "Documents" / "Exocortex"

class AgentTools:
    def list_vault(self) -> dict[str, list[str] | int]:
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

    def search_file(self, keyword: str) -> list[str]:
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

    def current_datetime(self):
        """
        desc: get current time and date time in user location
        output: string date day and time
        args: no arguments
        """
        return subprocess.run(['date'], capture_output=True, text=True).stdout
        

class ToolRegistry:
    def __init__(self, tool_object: Any):
        self._tools: dict[str, Callable] = {}

        for _, func in inspect.getmembers(
            tool_object,
            predicate=callable
        ):
            if func.__name__.startswith("_"):
                continue
            self._tools[func.__name__] = func

    def get(self, name: str):
        return self._tools.get(name)
    
    def ollama_tools(self) -> list[Any]:
        return list(self._tools.values())