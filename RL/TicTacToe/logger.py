import logging
import logging.handlers
import os

LOG_ROTATE_SIZE: int = 10 * 1000 * 1000  # byte
LOG_ROTATE_NUM: int = 10  # ログファイル数


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
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file
    if filename is not None:
        fh = logging.handlers.RotatingFileHandler(
            os.path.join("./logs/", filename), maxBytes=LOG_ROTATE_SIZE, backupCount=LOG_ROTATE_NUM
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
