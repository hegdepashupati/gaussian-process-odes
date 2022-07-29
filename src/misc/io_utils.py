import logging
import os
import json


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def saveargs(args, filename=None):
    with open(os.path.join(args.save, 'train_args.json') if filename is None else filename, 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


def get_logger(logpath, displaying=True, saving=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger
