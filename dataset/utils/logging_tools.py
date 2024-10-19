import logging
from colorlog import ColoredFormatter
from pytz import timezone
from datetime import datetime
from dataset.configs.history_data_crawlers_config import root_path
from pathlib import Path

logs_path = f"{root_path}/logs/"
Path(logs_path).mkdir(parents=True, exist_ok=True)


# ? change timezone to iran:
def timetz(*args):
    return datetime.now(tz).timetuple()


tz = timezone("Asia/Tehran")  # UTC, Asia/Shanghai, Europe/Berlin
logging.Formatter.converter = timetz


def logger_v1(
    logger_name,
    log_file="data_pipeline.log",
):
    """ """
    # ? Create top level logger
    log = logging.getLogger(logger_name)

    # ? Add console handler using our custom ColoredFormatter
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    LOGFORMAT = "  %(log_color)s[%(name)s][%(levelname)s]%(reset)s:%(log_color)s%(message)s%(reset)s"
    formatter = ColoredFormatter(LOGFORMAT, "%Y-%m-%d")
    ch.setFormatter(formatter)
    log.addHandler(ch)

    # ? Add file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ff = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s]: %(message)s", "%Y-%m-%d %H:%M"
    )
    fh.setFormatter(ff)
    log.addHandler(fh)

    # ? Set log level
    log.setLevel(logging.DEBUG)
    return log

    # log.debug("A quirky message only developers care about")
    # log.info("Curious users might want to know this")
    # log.warning("Something is wrong and any user should be informed")
    # log.error("Serious stuff, this is red for a reason")
    # log.critical("OH NO everything is on fire")

log_file = f"{root_path}/logs/General_jobs.log"
default_logger = logger_v1("test run", log_file=log_file)


