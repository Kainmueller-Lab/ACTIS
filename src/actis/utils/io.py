"""Input output functions for the project."""
import errno
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import List, Optional, Union

import yaml  # type: ignore

from actis.actis_logging import get_logger


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


def copy_folder(folder_to_copy, destination, copy_root_folder=True, force_copy=False):
    """Copies a folder to a destination.

    Args:
        folder_to_copy:
            The folder to copy
        destination:
            The destination folder to copy to
        copy_root_folder:
            boolean value. if true copies the root folder in the target destination.
            Else all files in the folder to copy.
        force_copy:
            boolean value. If true, removes the destination folder before copying.

    Returns:

    """
    folder_to_copy = Path(folder_to_copy)
    destination = Path(destination)

    if os.path.exists(destination) and os.path.samefile(folder_to_copy, destination):
        return destination

    if copy_root_folder:
        destination = destination.joinpath(folder_to_copy.name)

    if force_copy:
        force_remove(destination)

    create_path_recursively(destination)

    for root, dirs, files in os.walk(folder_to_copy):
        root = Path(root)

        for d in dirs:
            copy_folder(
                root.joinpath(d), destination.joinpath(d), copy_root_folder=False
            )
        for fi in files:
            copy(root.joinpath(fi), destination)
        break

    return destination


def force_remove(path, warning=True):
    path = Path(path)
    if path.exists():
        try:
            if path.is_file():
                try:
                    path.unlink()
                except PermissionError:
                    handle_remove_readonly(os.unlink, path, sys.exc_info())
            else:
                shutil.rmtree(
                    str(path), ignore_errors=False, onerror=handle_remove_readonly
                )
        except PermissionError as e:
            get_logger().warn("Cannot delete %s." % str(path))
            if not warning:
                raise e


def handle_remove_readonly(func, path, exc):
    """Changes readonly flag of a given path."""
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
        func(path)
    else:
        raise
