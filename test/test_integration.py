import os
from pathlib import Path
import sys

import pytest
import toml
import tempfile

from actis.argument_parsing import startup
from actis.utils.io import copy_folder

cur_path = Path(os.path.dirname(os.path.realpath(__file__)))


def load_resources():
    resources_path = cur_path.parent.joinpath("src", "actis", "utils", "resources")
    template_path = resources_path.joinpath("parameters.toml")

    return template_path


def copy_data(tmp_path):
    data_path = cur_path.parent.joinpath("data")
    _ = copy_folder(data_path, tmp_path, copy_root_folder=True)


@pytest.fixture()
def train_supervised():
    print("Setup train_supervised output")
    # create tmp_path
    tmp_path_dir = tempfile.TemporaryDirectory(dir=tempfile.gettempdir())
    tmp_path = Path(tmp_path_dir.name)
    tmp_path.mkdir(parents=True, exist_ok=True)

    param_file = str(load_resources())

    # set test parameters
    params = toml.load(param_file)
    params["base_dir"] = str(tmp_path)
    params["training_steps"] = 10
    params["validation_interval"] = 10
    params["num_workers"] = 0

    # write parameters to file
    with open(tmp_path.joinpath("test_train_super.toml"), "w") as f:
        toml.dump(params, f)

    # copy data to tmp_path to be able to use relative path
    copy_data(str(tmp_path))

    # build arguments
    sys.argv = [sys.argv[0]] + [
        "train_super",
        str(tmp_path.joinpath("test_train_super.toml")),
        "--log-out=%s" % str(tmp_path.joinpath("log")),
    ]

    # call
    startup()

    yield tmp_path
    tmp_path_dir.cleanup()


class TestIntegrationSuite:

    def test_train_super_and_semi(self, train_supervised):
        # create a checkpoint
        #self.train_supervised(tmp_path)
        tmp_path = train_supervised

        param_file = str(load_resources())

        # set test parameters
        params = toml.load(param_file)
        params["training_steps"] = 24  # todo: can't be < 24
        params["base_dir"] = str(tmp_path)
        params["validation_interval"] = 12
        params["num_workers"] = 0
        params["checkpoint_path"] = str(
            tmp_path.joinpath("experiments", "myexperiment", "train", "checkpoints", "best_model.pth"))

        with open(tmp_path.joinpath("test_train_semi.toml"), "w") as f:
            toml.dump(params, f)

        # build arguments
        sys.argv = [sys.argv[0]] + [
            "train_semi",
            str(tmp_path.joinpath("test_train_semi.toml")),
            "--log-out=%s" % str(tmp_path.joinpath("log")),
        ]

        # call
        startup()

    def test_evaluate(self, train_supervised):
        tmp_path = train_supervised

        sys.argv = [sys.argv[0]] + [
            "evaluate",
            "--base_dir=%s" % str(tmp_path),
            "--experiment=myexperiment",
            "--checkpoint=best_model.pth",
            "--fg_thresh_linspace_num=2",
            "--seed_thresh_linspace_num=2",
            "--log-out=%s" % str(tmp_path.joinpath("log")),
        ]

        # call
        startup()
