from datetime import datetime, timedelta
import pygame
from os import path

pygame.mixer.init()

SRC_DIR = path.dirname(path.abspath(__file__)) 
BASE_DIR = path.dirname(SRC_DIR)

class AlarmManager:
    def __init__(self, curfew_start_hour: int, curfew_duration_in_hours: int) -> None:
        """
        The initalization of an AlarmManager object.

        Args:
            curfew_start_hour: An integer that represents start of the curfew hour, must be a number between 0 and 23.
            curfew_duration_in_hours: How long the curfew the curfew lasts in hour units.
        
        """
        self.curfew_start_hour = curfew_start_hour
        self.curfew_duration_in_hours = curfew_duration_in_hours

        self.start_of_curfew_datetime = datetime.now()
        self.start_of_curfew_datetime = self.start_of_curfew_datetime.replace(hour=self.curfew_start_hour, minute=0, second=0)
        self.end_of_curfew_datetime = self.start_of_curfew_datetime + timedelta(hours=self.curfew_duration_in_hours)

    def update_curfew_datetimes(self) -> None:
        """
        Reassigns datetime attributes.
        Used when to readjust for timezones or entering the next day.
        """
        self.start_of_curfew_datetime = datetime.now()
        self.start_of_curfew_datetime = self.start_of_curfew_datetime.replace(hour=self.curfew_start_hour)
        self.end_of_curfew_datetime = self.start_of_curfew_datetime + timedelta(hours=self.curfew_duration_in_hours)

    def get_now_is_curfew(self) -> bool:
        """Returns true if the current time is during curfew."""
        return self.start_of_curfew_datetime <= datetime.now() < self.end_of_curfew_datetime

    def get_busy(self) -> bool:
        """Returns a boolean value that indicates if the pygame's music stream is currently playing."""
        return pygame.mixer.music.get_busy()

    def play_alarm(self) -> None:
        """Plays the security alarm file stored in assets/audios/."""
        alarm_path = path.join(BASE_DIR, 'assets', 'audios', 'security-alarm.mp3')
        
        if not self.get_busy():
            pygame.mixer.music.load(alarm_path)
            pygame.mixer.music.play()

    def kill(self) -> None:
        """Uninitializes Pygame audio mixer module."""
        pygame.mixer.quit()