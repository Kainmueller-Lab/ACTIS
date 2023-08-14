import os
from collections.abc import Sequence
from pathlib import Path

import toml


class Parameter(Sequence):
    """Base class for all parameters."""

    def __init__(self, *args, **kwargs):
        """Initialize the parameter class with its default values."""
        # init with default values from resources
        current_path = Path(os.path.dirname(os.path.realpath(__file__)))
        param_base_file = Path(current_path).joinpath(
            "resources", "parameters.toml"
        )
        args_init = toml.load(str(param_base_file))

        for key in args_init:
            self.__setitem__(key, args_init[key])

        if args != ():
            for dictionary in args:
                for key in dictionary:
                    self._setattr(key, dictionary[key])

        if kwargs != {}:
            for key in kwargs:
                self._setattr(key, kwargs[key])

    def _setattr(self, key, val):
        if hasattr(self, key):
            setattr(self, key, val)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        for key in self.__dict__:
            yield key

    def __len__(self):
        return len(self.__dict__)

    def reset(self):
        """Reset all parameters to None."""
        for key in self.__dict__:
            self._setattr(key, None)

    @classmethod
    def from_toml(cls, path: str):
        """Create a parameter object from a yml file."""
        params = toml.load(path)

        return cls(params)

    def __str__(self, indent=1):
        """Return a string representation of the parameter object."""
        s = "%s:  \n" % self.__class__.__name__
        for attr in self.__dict__:
            s += f"{attr:<30}{str(getattr(self, attr)):<40}\n"
        return s
