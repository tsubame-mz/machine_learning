import logging
import logging.handlers

# import os


def setup_logger(name: str, level: int = logging.DEBUG, filename: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # formatter = logging.Formatter(
    #     "[%(name)s][%(levelname)s][%(asctime)s.%(msecs)03d][%(module)s][%(filename)s:%(funcName)s:%(lineno)d]: %(message)s",
    #     datefmt="%Y/%m/%d %H:%M:%S",
    # )
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file
    # if filename is not None:
    #     if not os.path.exists(LOG_FILE_DIR):
    #         os.mkdir(LOG_FILE_DIR)

    #     fh = logging.handlers.RotatingFileHandler(
    #         os.path.join(LOG_FILE_DIR, filename), maxBytes=LOG_ROTATE_SIZE, backupCount=LOG_ROTATE_NUM
    #     )
    #     fh.setLevel(logging.DEBUG)
    #     fh.setFormatter(formatter)
    #     logger.addHandler(fh)

    return logger
