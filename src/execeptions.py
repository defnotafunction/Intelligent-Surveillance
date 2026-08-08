import logging
from logging.handlers import TimedRotatingFileHandler

# LOGGING
file_handler = TimedRotatingFileHandler(
    "logs/camera.log", 
    when="midnight", 
    interval=1, 
    backupCount=7,
)

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler], format='%(asctime)s - %(levelname)s - %(message)s')

# CUSTOM EXECEPTIONS
class ControllerNotFoundError(Exception):
    def __init__(self, controller_id) -> None:
        super().__init__(f'Controller with the ID {controller_id} was not found.')
