"""Input output functions for the project."""
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Union

import yaml  # type: ignore

# Global variable to save program call time.
CALL_TIME = None


def write_dict_to_yml(yml_file: Union[str, Path], d: dict) -> bool:
    """Write a dictionary to a file in yml format.

    Args:
        yml_file:
            Path to the yml file.
        d:
            Dictionary to be written to the yml file.

    Returns:
        True if successful.

    """
    yml_file = Path(yml_file)
    p = Path(yml_file.parent)
    p.mkdir(parents=True, exist_ok=True)

    with open(yml_file, "w+") as yml_f:
        yml_f.write(yaml.dump(d, Dumper=yaml.Dumper))

    return True


def get_dict_from_yml(yml_file: Path) -> dict:
    """Read a dictionary from a file in yml format."""
    with open(str(yml_file)) as yml_f:
        d = yaml.safe_load(yml_f)

    if not isinstance(d, dict):
        raise TypeError("Yaml file %s invalid!" % str(yml_file))

    return d


def create_path_recursively(path: Union[str, Path]) -> bool:
    """Create a path. Creates missing parent folders.

    Args:
        path:
            Path to be created.

    Returns:
        True if successful.

    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    return True


def list_files_recursively(
        path: Path,
        root: Optional[Union[Path]] = None,
        relative: bool = False,
        endswith: Optional[str] = None,
) -> List[Path]:
    """List all files in a repository recursively."""
    if not root:
        root = path
    files_list = []

    for cur_root, dirs, files in os.walk(str(path)):
        cur_root = str(Path(cur_root))

        for d in dirs:
            files_list += list_files_recursively(
                Path(cur_root).joinpath(d), root, relative, endswith
            )
        for fi in files:
            if endswith:
                if not fi.endswith(endswith):
                    continue
            if relative:
                files_list.append(Path(cur_root).joinpath(fi).relative_to(root))
            else:
                files_list.append(Path(cur_root).joinpath(fi))
        break

    return files_list


def get_doc_file_prefix() -> str:
    """Get the time when the program was called.

    Returns:
        Time when the program was called.

    """
    global CALL_TIME

    if not CALL_TIME:
        CALL_TIME = time.strftime("%Y%m%d_%H-%M-%S")

    call_time = CALL_TIME

    return "run_%s" % call_time


def copy(file: Union[str, Path], path_to: Union[str, Path]) -> Path:
    """Copy a file A to either folder B or file B. Makes sure folder structure for target exists.

    Args:
        file:
            Path to the file to be copied.
        path_to:
            Path to the target folder or file.

    Returns:
        Path to the copied file.

    """
    file = Path(file)
    path_to = Path(path_to)

    if os.path.exists(path_to) and os.path.samefile(file, path_to):
        return path_to

    create_path_recursively(path_to.parent)

    return Path(shutil.copy(file, path_to))
