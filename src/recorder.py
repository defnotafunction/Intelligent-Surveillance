import cv2
from datetime import datetime
from os import path, makedirs
import time

SRC_DIR = path.dirname(path.abspath(__file__)) 
BASE_DIR = path.dirname(SRC_DIR)

class Recorder:
    def __init__(self, cap: cv2.VideoCapture) -> None:
        """
        The initalization of a Recorder object.

        Args:
            cap: A cv2 VideoCapture object.
        """
        makedirs(path.join(BASE_DIR, 'data', 'videos'), exist_ok=True)  # Create videos folder in case it isn't there

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = 20.0

        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = None
        self.minimum_length = 30  # 30 Seconds
        self.current_time = time.time()
        self.called_write = False

    def write(self, frame, prefix: str) -> None:
        """
        Calls the cv2 VideoWriter write method

        Args:
            frame: A frame from the cv2 VideoCapture read method.
            prefix: A string that'll be used at the start at the filename of the video file created on the first call of this method after using the reset method.
        """
        if self.video_writer is None:  # First write call or after the writer has been resetted.
            self.called_write = True

            str_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            filename = f'{prefix}_sc_footage_{str_datetime}.mp4'  # To make every filename unique
            
            self.video_writer = cv2.VideoWriter(
                path.join(BASE_DIR, 'data', 'videos', f"{filename}"),
                self.fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )

        self.video_writer.write(frame)

    def get_done_recording(self) -> bool:
        """
        Returns a boolean that indicates whether the video_writer attribute can be assigned to None based on if the amount of seconds since creating a video file is greater than the reset_cooldown attribute.
        This ensures that the current video file will be at least the number of seconds the minimum_length attribute is assigned.
        """
        if not self.called_write:
            self.current_time = time.time()

        return (time.time() - self.current_time >= self.minimum_length) and self.called_write

    def reset(self) -> None:
        """
        Reassigns the video_writer and called_write attributes to None and resets current_time attribute.
        Only use this method when the current video file is done being created!
        """
        self.video_writer = None
        self.called_write = False
        self.current_time = time.time()