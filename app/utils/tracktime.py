""" This module is a decorator that tracks time on methods.
"""
import time
import logging
from typing import Any

from app.config import BaseConfig as app_config

logger = logging.getLogger(app_config.APP_NAME)

def tracktime(func: Any) -> Any:
    """Used as a decorator for tracking time for
    methods. Prints the time used.

    Args:
        func (_type_): a method
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        logger.info("%s executed in: %s seconds", func.__name__, elapsed_time)
        return result

    return wrapper
