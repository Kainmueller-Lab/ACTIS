import argparse
import sys
from typing import Callable

import S3
from S3.evaluate_experiment import evaluate
from S3.s3_logging import get_log_file, get_logger, configure_logger, close_logger
from S3.train_semi import train_semi
from S3.train_supervised import train_supervised
from S3.utils.parameter import Parameter


def startup():
    """Entry points of `polarityjam`."""
    parser = create_parser()
    args = parser.parse_args()

    __run_subcommand(args, parser)


def __run_subcommand(args, parser):
    """Calls a specific subcommand."""
    command = ""
    try:
        command = sys.argv[1]  # command always expected at second position
    except IndexError:
        parser.error("Please provide a valid action!")

    log_file = get_log_file(args.log_out)
    configure_logger('INFO', logfile_name=log_file)


    get_logger().info("S3 Version %s" % S3.__version__)

    get_logger().debug("loading parameter file...")

    args.param = Parameter.from_toml(args.param)

    get_logger().debug("Running %s subcommand..." % command)

    args.func(args)  # execute entry point function

    close_logger()


def create_parser():
    """Creates the parser for the command line interface."""
    parser = S3Parser()

    # train_semi action
    p = parser.create_file_command_parser('train_semi', train_semi, '')
    parser.add_argument(
        "--wandb", dest="wandb", action="store_true", default=False,
        help="Use weights and biases for the analysis. Environment variable \"WANDB_API_KEY\" has to be configured! "
             "You can find this in your user settings (e.g. https://wandb.ai/settings)"
    )
    parser.add_argument(
        "--wandb_project", dest="wandb_project", type=str, default="S3",
    )

    # train super action
    p = parser.create_file_command_parser('train_super', train_supervised,
                                          'Train your network based on your parameters')
    parser.add_argument(
        "--wandb", dest="wandb", action="store_true", default=False,
        help="Use weights and biases for the analysis. Environment variable \"WANDB_API_KEY\" has to be configured! "
             "You can find this in your user settings (e.g. https://wandb.ai/settings)"
    )
    parser.add_argument(
        "--wandb_project", dest="wandb_project", type=str, default="S3",
    )

    # evaluate experiment action
    p = parser.create_command_parser(
        'evaluate', evaluate, 'Evaluate your experiment.'
    )
    p.add_argument("--experiment", type=str, default="exp_0_mouse_seed1_samples10_DINO_L1_loss_highLR")
    p.add_argument("--checkpoint", type=str, default="best_model.pth")

    p.add_argument('in_file', type=str, help='Path to the input tif file.')  # todo: copy args

    return parser.parser


class ArgumentParser(argparse.ArgumentParser):
    """Override default error method of all parsers to show help of sub-command."""

    def error(self, message: str):
        self.print_help()
        self.exit(2, '%s: error: %s\n' % (self.prog, message))


class S3Parser(ArgumentParser):
    def __init__(self):
        super().__init__()
        self.parent_parser = self.create_parent_parser()
        self.parser = self.create_parser()
        self.subparsers = self.parser.add_subparsers(title='actions', help='sub-command help')

    @staticmethod
    def create_parent_parser() -> ArgumentParser:
        """Parent parser for all subparsers to have the same set of arguments."""
        parent_parser = ArgumentParser(add_help=False)
        parent_parser.add_argument('--version', '-V', action='version', version="%s " % S3.__version__)
        # parse logging
        parent_parser.add_argument(
            '--log',
            required=False,
            help='Logging level for your command. Choose between %s' %
                 ", ".join(['INFO', 'DEBUG']),
            default='INFO'
        )
        parent_parser.add_argument(
            '--log-out',
            required=False,
            help='Points to a folder where the log file will be stored. If not specified, the log file will be stored '
                 'in the current working directory.',
            default=None
        )
        return parent_parser

    def create_parser(self) -> ArgumentParser:
        """Creates the main parser for the pipeline."""
        parser = ArgumentParser(
            add_help=True,
            description='S3 - Semi Supervised Training for Instance and Semantic Segmentation.',
            parents=[self.parent_parser])
        return parser

    def create_command_parser(self, command_name: str, command_function: Callable, command_help: str) -> ArgumentParser:
        """Creates a parser for a S3 command, specified by a name, a function and a help description."""
        parser = self.subparsers.add_parser(command_name, help=command_help, parents=[self.parent_parser])
        parser.set_defaults(func=command_function)
        return parser

    def create_file_command_parser(self, command_name: str, command_function: Callable,
                                   command_help: str) -> ArgumentParser:
        """Creates a parser for a Polarityjam command dealing with a file.

        Parser is specified by a name, a function and a help description.
        """
        parser = self.create_command_parser(command_name, command_function, command_help)
        parser.add_argument('param', type=str, help='Path to the parameter file.')
        return parser
