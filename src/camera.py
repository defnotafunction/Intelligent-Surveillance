import cv2
from .controller import ControllerManager, get_any_controllers_connected

class Camera:
    def __init__(self) -> None:
        # Initialize Pedestrian detection
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self._cap = cv2.VideoCapture(0)  # Will capture frames from camera

        if get_any_controllers_connected():
            self._controller = ControllerManager()  # Handles any inputs from game controllers
        else:
            self._controller = None

    def _get_controller(self) -> tuple[ControllerManager | None]:
        """Returns a ControllerManager object if a controller is detected, otherwise it returns None. Used to adapt to disconnections and reconnections."""
        return ControllerManager() if get_any_controllers_connected() else None

    def _handle_controller_events(self) -> None:
        """Executes all controller-related events."""
        if self._controller is not None:
            self._controller.listen()

        self._controller = self._get_controller()


    def run(self) -> None:
        """Captures live video frames and detects pedistrians."""

        while True:
            ret, frame = self._cap.read()

            if not ret:
                print('Frame failed')
                break

            boxes, weights = self._hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            confidence_threshold = 0.7

            for box, weight in zip(boxes, weights):
                if weight > confidence_threshold:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self._handle_controller_events()
            
            cv2.imshow('Original Live Feed', frame)
            if cv2.waitKey(1) % 0xFF == ord('q'):
                break

        self._controller.kill()
        self._cap.release()
        cv2.destroyAllWindows()
