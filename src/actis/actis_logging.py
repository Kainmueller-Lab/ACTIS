import logging
import sys
from pathlib import Path

import actis
from actis.utils.io import get_doc_file_prefix

LOGGER_NAME = actis.__name__


def get_logger():
    return logging.getLogger(LOGGER_NAME)  # root logger


def get_log_file(out_folder):
    log_file = Path(out_folder).joinpath("%s.log" % get_doc_file_prefix())
    log_file.touch()

    return log_file


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
        print(
            "You chose to run the experiment logging to wandb.ai. Environment variable \"WANDB_API_KEY\" most"
            "likely not configured. Switch off wandb logging by specifying not passing --wandb to the call."
        )
        raise
    run.name = run.id

    return run
