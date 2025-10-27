import logging
import colorlog

# Setting up color logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)s] - [%(name)s] - %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white'
    }
))

def get_logger(name, level = "DEBUG"):
    logger = logging.getLogger(name)

    if level == "DEBUG":
        logger.setLevel(level=logging.DEBUG)
    elif level == "INFO":
        logger.setLevel(level=logging.INFO)
    elif level == "WARNING":
        logger.setLevel(level=logging.WARNING)
    elif level == "ERROR":
        logger.setLevel(level=logging.ERROR)
    elif level == "CRITICAL":
        logger.setLevel(level=logging.CRITICAL)
    else:
        logger.setLevel(level=logging.DEBUG)

    logger.addHandler(handler)

    return logger
