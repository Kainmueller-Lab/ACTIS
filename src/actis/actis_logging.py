import logging
import sys
from pathlib import Path
import time

import actis

LOGGER_NAME = actis.__name__

# Global variable to save program call time.
CALL_TIME = None

def get_logger():
    return logging.getLogger(LOGGER_NAME)  # root logger


def get_log_file(out_folder):
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    log_file = Path(out_folder).joinpath("%s.log" % get_doc_file_prefix())
    log_file.touch()

    return log_file


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

def configure_logger(loglevel=None, logfile_name=None, formatter_string=None):
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(loglevel)

    # create formatter
    if not formatter_string:
        formatter = get_default_formatter()
    else:
        formatter = logging.Formatter(formatter_string)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if logfile_name:
        ch = logging.FileHandler(logfile_name, mode='a', encoding=None, delay=False)
        ch.setLevel(loglevel)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def close_logger():
    for h in logging.getLogger(LOGGER_NAME).handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
    logging.getLogger(LOGGER_NAME).handlers.clear()


def get_default_formatter():
    return logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')


def wandb_init(args):
    import wandb
    try:
        run = wandb.init(project=args.wandb_project, config=args.__dict__, sync_tensorboard=True, dir=args.log_out)
    except wandb.errors.UsageError:
        get_logger().error(
            "You chose to run the experiment logging to wandb.ai. Environment variable \"WANDB_API_KEY\" most"
            "likely not configured. Switch off wandb logging by specifying not passing --wandb to the call."
        )
        raise
    run.name = run.id

    return run
