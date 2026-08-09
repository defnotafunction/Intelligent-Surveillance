import cv2
from .controller import ControllerManager, get_any_controllers_connected
from .facial_recognition import FaceAnalyzer
import logging
import random
import threading
from os import path

class Camera:
    def __init__(self) -> None:
        # Initialize Pedestrian detection
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self._cap = cv2.VideoCapture(0)  # Will capture frames from camera

        # FACE RECOGNITION AND DETECTION
        logging.debug(cv2.data.haarcascades)
        self._face_casacde = cv2.CascadeClassifier(path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
        self._face_analyzer = FaceAnalyzer()

        if get_any_controllers_connected():
            self._controller = ControllerManager()  # Handles any inputs from game controllers
        else:
            self._controller = None

        self._current_controller_input = None

    def _get_controller(self) -> tuple[ControllerManager | None]:
        """Returns a ControllerManager object if a controller is detected, otherwise it returns None. Used to adapt to disconnections and reconnections."""
        return ControllerManager() if get_any_controllers_connected() else None

    def _handle_controller_events(self) -> None:
        """Executes all controller-related events."""
        if self._controller is not None:
            controller_input = self._controller.listen()
            self._current_controller_input = controller_input

        self._controller = self._get_controller()

    def _handle_pedestrian_detection(self, frame, confidence_threshold: float) -> None:
        """
        Runs all pedestrian dectection related events.
        
        Args:
            frame: A numpy array deriving from the VideoCapture's read method.
            confidence_threshold: A float value within an exclusive range between 0 and 1. The threshold for the weights of the detector that determines whether a detected object is classified as a pedestrian or not.
        """
        boxes, weights = self._hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            
        for box, weight in zip(boxes, weights):
            if weight > confidence_threshold:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def _handle_face_detecting(self, frame) -> None:
        """
        Runs all pedestrian dectection related events.
        
        Args:
            frame: A numpy array deriving from the VideoCapture's read method.
        """
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self._face_casacde.detectMultiScale(grayscaled_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in detected_faces:
            colored_face = frame[y:y+h, x:x+w]

            if self._face_analyzer.get_face_is_in_known_faces(colored_face):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            elif self._face_analyzer.get_face_is_in_unknown_faces(colored_face):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Every once in a while keep collecting data for already stored unknown face
                if random.random() <= 0.01:
                    self._face_analyzer.save_unknown_face(colored_face)

            else:
                # If detected face isn't stored at all, automatically store it in unknown faces.
                cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 128, 128), 2)
                self._face_analyzer.save_unknown_face(colored_face)

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

            # TODO: Multithreading because Deepface face verification can take time and delay video
            self._handle_face_detecting(frame)
            
            cv2.imshow('Original Live Feed', frame)
            if (
                cv2.waitKey(1) % 0xFF == ord('q')
                or self._current_controller_input == 'QUIT'
                ):
                break

            frames_ran += 1

        self._controller.kill()
        self._cap.release()
        cv2.destroyAllWindows()
