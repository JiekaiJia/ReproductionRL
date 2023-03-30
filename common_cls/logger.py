import logging
import traceback


class Logger:
    def __init__(self, file_name: str):
        self.logger = logging.getLogger(file_name)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        self.logger.setLevel(level=logging.DEBUG)

    def set_level(self, level: str = logging.DEBUG):
        self.logger.setLevel(level=level)

    def add_stream_handler(self, level: str = logging.DEBUG):
        sh = logging.StreamHandler()
        sh.setFormatter(self.formatter)
        sh.setLevel(level)
        self.logger.addHandler(sh)

    def add_file_handler(self, log_file_name: str, level: str = logging.DEBUG):
        fh = logging.FileHandler(log_file_name)
        fh.setFormatter(self.formatter)
        fh.setLevel(level)
        self.logger.addHandler(fh)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        self.logger.error(msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)