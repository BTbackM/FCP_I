from __future__ import annotations
from beater.utils.disk import ABS_PATH
from logging import config, getLogger
from os import path
from yaml import safe_load

def setup_logger(name: str):
    with open(path.join(ABS_PATH, 'beater', 'utils','logger.yml'), 'r') as file:
        config.dictConfig(safe_load(file.read()))
    logger = getLogger(name)

    return logger

Logger = setup_logger('BT')
