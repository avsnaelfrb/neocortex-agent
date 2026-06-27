from pathlib import Path
import subprocess

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
        