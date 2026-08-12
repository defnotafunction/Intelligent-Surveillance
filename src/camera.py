import cv2
from .controller import ControllerManager, get_any_controllers_connected
from .facial_recognition import FaceAnalyzer
from .alarm import AlarmManager
from.recorder import Recorder
import logging
import random
from os import path, makedirs
import speech_recognition as sr
import pyttsx3
from datetime import datetime

class Camera:
    def __init__(self, curfew_start_hour: int, curfew_duration_in_hours: int) -> None:
        """
        The initalization of a Camera object.

        Args:
            curfew_start_hour: An integer that represents start of the curfew hour, must be a number between 0 and 23.
            curfew_duration_in_hours: How long the curfew the curfew lasts in hour units.

        """
        # Initialization of AlarmManager
        self._alarm_manager = AlarmManager(
            curfew_start_hour=curfew_start_hour,
            curfew_duration_in_hours=curfew_duration_in_hours
            )

        # Initialize Pedestrian detection
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self._cap = cv2.VideoCapture(0)  # Will capture frames from camera

        # FACE RECOGNITION AND DETECTION
        self._face_casacde = cv2.CascadeClassifier(path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        self._face_analyzer = FaceAnalyzer()

        if get_any_controllers_connected():
            self._controller = ControllerManager()  # Handles any inputs from game controllers
        else:
            self._controller = None

        self._current_controller_input = None
        self._person_is_visible = False
        self.face_remembering_enabled = False  # Determines whether program will begin remembering faces or not
        self._speech_recognizer = sr.Recognizer()  # SPEECH TO TEXT
        self._tts_engine = pyttsx3.init()  # TEXT TO SPEECH
        self._recorder = Recorder(self._cap)
        self._controller_activated_record = False 

    def talk(self, text) -> None:
        """
        Runs all necessary logic for text to speech.
        
        Args:
            text: The text for the engine to say.
        """
        self._tts_engine.say(text)
        self._tts_engine.runAndWait()
        self._tts_engine = pyttsx3.Engine()  # Engine can't run say more than once for some reason

    def _get_controller(self) -> tuple[ControllerManager | None]:
        """Returns a ControllerManager object if a controller is detected, otherwise it returns None. Used to adapt to disconnections and reconnections."""
        return ControllerManager() if get_any_controllers_connected() else None

    def _handle_controller_events(self) -> None:
        """Executes all controller-related events."""
        if self._controller is not None:
            controller_input = self._controller.listen()
            self._current_controller_input = controller_input

            if self._current_controller_input == 'REMEMBER':
                self.face_remembering_enabled = not self.face_remembering_enabled  # Toggles between True and False
            elif self._current_controller_input == 'RECORD':
                self._controller_activated_record = not self._controller_activated_record

        self._controller = self._get_controller()

    def _handle_pedestrian_detection(self, frame, confidence_threshold: float) -> None:
        """
        Runs all pedestrian dectection related events.
        
        Args:
            frame: A numpy array deriving from the VideoCapture's read method.
            confidence_threshold: A float value within an exclusive range between 0 and 1. The threshold for the weights of the detector that determines whether a detected object is classified as a pedestrian or not.
        """
        boxes, weights = self._hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

        if len(boxes) > 0:
            self._person_is_visible = True
        
        for box, weight in zip(boxes, weights):
            if weight > confidence_threshold:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def _handle_face_detecting(self, frame, frames_ran: int) -> None:
        """
        Runs all face dectection related events.
        
        Args:
            frame: A numpy array deriving from the VideoCapture's read method.
            frames_ran: An integer that holds how many frames have been captured since running.
        """
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self._face_casacde.detectMultiScale(grayscaled_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(detected_faces) > 0:
            self._person_is_visible = True

        for (x, y, w, h) in detected_faces:
            colored_face = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if self.face_remembering_enabled:
                with sr.Microphone() as source:
                    self._speech_recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio = self._speech_recognizer.listen(source)
                    text = self._speech_recognizer.recognize_vosk(audio)
                    text = text.lower()

                    if 'my name is' in text:
                        person_name = text.replace('my name is', '')
                        self._face_analyzer.remember_face(colored_face, person_name)

            if self._current_controller_input == 'COUNT':
                count = self._face_analyzer.get_unknown_face_count(colored_face)
                self.talk(f"This face has been tracked in unknown faces {count} {'time' if count == 1 else 'times'}.")

            # Below includes computationally expensive methods that uses Deepface
            if frames_ran % 100 == 0:
                if self._face_analyzer.get_face_is_in_known_faces(colored_face):
                    person_name = self._face_analyzer.get_name_of_known_face(colored_face)
                    self.talk(f'Hello {person_name}')

                elif self._face_analyzer.get_face_is_in_unknown_faces(colored_face): 
                    # Track unknown faces every once in a while
                    chance_to_capture = 0.1  # Ten percent (random.random returns a float value between 0 and 1)
                    random_float = random.random()
                    if random_float <= chance_to_capture:
                        self._face_analyzer.save_unknown_face(colored_face)

                else:
                    # If detected face isn't stored at all, automatically store it in unknown faces.
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 128, 128), 2)
                    self._face_analyzer.save_unknown_face(colored_face)

    def _handle_recording_and_alarm(self, frame) -> None:
        """
        Handles all necessary logic related to the recording and alarm system.
        
        Args:
            frame: A numpy array deriving from the VideoCapture's read method.
        """
        # Play alarm and write video if person is visible during curfew or recorder isn't done recording during curfew
        now_is_curfew = self._alarm_manager.get_now_is_curfew()
        done_recording = self._recorder.get_done_recording()
        if (
            (self._person_is_visible and now_is_curfew)
            or (not done_recording and now_is_curfew)
            ):
            self._alarm_manager.play_alarm()
            self._recorder.write(frame)
            return  # Prevents controller input statement from messing with the segments of this statement.
        elif done_recording:
            self._recorder.reset()

        if self._controller_activated_record:
            self._recorder.write(frame)
        else:
            self._recorder.reset()

    def run(self) -> None:
        """Captures live video frames and detects pedistrians."""
        frames_ran = 0
        
        while True:
            ret, frame = self._cap.read()

            if not ret:
                logging.error('Frame failed!')
                break

            self._handle_controller_events()

            self._handle_pedestrian_detection(frame, confidence_threshold=0.8)

            self._handle_face_detecting(frame, frames_ran=frames_ran)

            self._handle_recording_and_alarm(frame)
             
            cv2.putText(
                frame,
                f'Remembering Faces: {self.face_remembering_enabled}',
                org=(50, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=1
                )
            cv2.imshow('Original Live Feed', frame)
            if (
                cv2.waitKey(1) % 0xFF == ord('q')
                or self._current_controller_input == 'QUIT'
                ):
                break

            frames_ran += 1
            self._person_is_visible = False  # Resets attribute, if person is still visible it'll become True again and will be processed as true.

        self._controller.kill()
        self._alarm_manager.kill()
        self._cap.release()
        cv2.destroyAllWindows()
