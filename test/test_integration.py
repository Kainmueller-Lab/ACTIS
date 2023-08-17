import os
from pathlib import Path
import sys

import toml

from actis.argument_parsing import startup
from actis.utils.io import copy_folder

class TestIntegrationSuite:
    cur_path = Path(os.path.dirname(os.path.realpath(__file__)))

    def load_resources(self):
        resources_path = self.cur_path.parent.joinpath("src", "actis", "utils", "resources")
        template_path = resources_path.joinpath("parameters.toml")

        return template_path

    def copy_data(self, tmp_path):
        data_path = self.cur_path.parent.joinpath("data")
        _ = copy_folder(data_path, tmp_path, copy_root_folder=True)

    def train_supervised(self, tmp_path):
        param_file = str(self.load_resources())

        # set test parameters
        params = toml.load(param_file)
        params["base_dir"] = str(tmp_path)
        params["training_steps"] = 1000

        # write parameters to file
        with open(tmp_path.joinpath("test_train_super.toml"), "w") as f:
            toml.dump(params, f)

        # copy data to tmp_path to be able to use relative path
        self.copy_data(tmp_path)

        # build arguments
        sys.argv = [sys.argv[0]] + [
            "train_super",
            str(tmp_path.joinpath("test_train_super.toml")),
            "--log-out=%s" % str(tmp_path.joinpath("log")),
        ]

        # call
        startup()

    def test_train_super_and_semi(self, tmp_path):
        # create a checkpoint
        self.train_supervised(tmp_path)

        param_file = str(self.load_resources())

        # set test parameters
        params = toml.load(param_file)
        params["training_steps"] = 80
        params["checkpoint_path"] = str(tmp_path.joinpath("experiments", "myexperiment", "train", "checkpoints", "best_model.pth"))

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

    def test_train_super(self, tmp_path):
        self.train_supervised(tmp_path)

    def test_evaluate(self, tmp_path):
        # create a checkpoint
        self.train_supervised(tmp_path)

        sys.argv = [sys.argv[0]] + [
            "evaluate",
            "--base_dir=%s" % str(tmp_path),
            "--experiment=myexperiment",
            "--checkpoint=best_model.pth"
            "--log-out=%s" % str(tmp_path.joinpath("log")),
        ]

        # call
        startup()
