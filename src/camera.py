import cv2
from .controller import ControllerManager, get_any_controllers_connected
from .facial_recognition import FaceAnalyzer
from .alarm import AlarmManager
from .recorder import Recorder
from .microphone import SoundAnalyzer, SpeechRecognition
from .sender import GmailSender
import logging
import random
import numpy as np
from os import path, listdir
import pyttsx3
from sklearn.exceptions import NotFittedError
from concurrent.futures import ThreadPoolExecutor
from groq import APIConnectionError

SRC_DIR = path.dirname(path.abspath(__file__)) 
BASE_DIR = path.dirname(SRC_DIR)

class Camera:
    def __init__(self, curfew_start_hour: int, curfew_duration_in_hours: int) -> None:
        """
        The initalization of a Camera object.

        Args:
            curfew_start_hour: An integer that represents start of the curfew hour, must be a number between 0 and 23.
            curfew_duration_in_hours: How long the curfew the curfew lasts in hour units.

        """
        self._wake_word = 'Paper'
        # Initialization of AlarmManager
        self._alarm_manager = AlarmManager(
            curfew_start_hour=curfew_start_hour,
            curfew_duration_in_hours=curfew_duration_in_hours
            )
        self._sound_analyzer = SoundAnalyzer()

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
        self._current_known_face = None

        self._speech_recognizer = SpeechRecognition()  # SPEECH TO TEXT
        self._listening_pool = ThreadPoolExecutor(max_workers=1)
        self._current_words_spoken = {'words': None}

        self._tts_engine = pyttsx3.init()  # TEXT TO SPEECH
        self._recorder = Recorder(self._cap)
        self._sender = GmailSender()

        # BOOLEAN ATTRIBUTES
        self._known_face_spotted = False
        self._person_is_visible = False
        self.face_remembering_enabled = False  # Determines whether program will begin remembering faces or not
        self._controller_activated_record = False 
        self._past_controller_activated_record = False
        self._detected_sound_anamoly = False
        self._training_sound_analyzer = False

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

    def _send_email_of_unknown_faces_graph(self) -> None:
        """Uses yagmail to send an email that holds the graph of clusters of unknown faces."""
        try:
            graph_path = self._face_analyzer.create_graph_of_unknown_faces()
            self._sender.send(
                'Unknown Faces Graph',
                message="Here's a graph of clusters of unknown faces!",
                file_paths=[graph_path]
        ) 
        except ValueError:
            self.talk("There aren't enough unknown faces stored in order to do that.")

    def _handle_controller_events(self) -> None:
        """Executes all controller-related events."""
        if self._controller is not None:
            controller_input = self._controller.listen()
            self._current_controller_input = controller_input

            if self._current_controller_input == 'REMEMBER':
                self.face_remembering_enabled = not self.face_remembering_enabled  # Toggles between True and False
            elif self._current_controller_input == 'RECORD':
                self._past_controller_activated_record = self._controller_activated_record
                self._controller_activated_record = not self._controller_activated_record

            # SOUND ANALYZER METHODS
            elif self._current_controller_input == 'SOUND TRAIN':
                self._training_sound_analyzer = not self._training_sound_analyzer
            elif self._current_controller_input == 'SOUND RESET':
                self._sound_analyzer.reset_model()
            elif self._current_controller_input == 'GRAPH UNKNOWN FACES':
                # T-SNE uses perplexity of 5, if # of unknown faces is lower it raises exception.
                self._send_email_of_unknown_faces_graph()

                self._current_controller_input = None


        self._controller = self._get_controller()

    def _handle_pedestrian_detection(self, frame: np.ndarray, confidence_threshold: float) -> None:
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

    def _handle_face_detecting(self, frame: np.array, frames_ran: int) -> None:
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

        can_analyze_faces = frames_ran % 100 == 0

        if can_analyze_faces:
            self._known_face_spotted = False
            self._current_known_face = None

        for (x, y, w, h) in detected_faces:
            colored_face = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if self.face_remembering_enabled:
                    text = self._speech_recognizer.listen()  

                    if 'my name is' in text:
                        person_name = text.replace('my name is', '')
                        self._face_analyzer.remember_face(colored_face, person_name)   

            if self._current_controller_input == 'COUNT':
                count = self._face_analyzer.get_unknown_face_count(colored_face)
                self.talk(f"This face has been tracked in unknown faces {count} {'time' if count == 1 else 'times'}.")

            # Below includes computationally expensive methods that uses Deepface
            if can_analyze_faces:
                if self._face_analyzer.get_face_is_in_known_faces(colored_face):
                    self._known_face_spotted = True
                    self._current_known_face = colored_face
                    person_name = self._face_analyzer.get_name_of_known_face(colored_face)

                    if frames_ran % 500 == 0:
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

    def _handle_recording_and_alarm(self, frame: np.ndarray, frames_ran: int) -> None:
        """
        Handles all necessary logic related to the recording and alarm system.
        
        Args:
            frame: A numpy array deriving from the VideoCapture's read method.
            frames_ran: An integer that holds how many frames have been captured since running.
        """
        last_video = listdir(path.join(BASE_DIR, 'data', 'videos'))[-1]
        last_video_path = path.join(BASE_DIR, 'data', 'videos', last_video)
        # Play alarm and write video if person is visible during curfew or sound anamoly was detected
        now_is_curfew = self._alarm_manager.get_now_is_curfew()
        done_recording = self._recorder.get_done_recording()
        if (
            (self._person_is_visible and now_is_curfew)
            or (not done_recording and now_is_curfew and self._recorder.called_write)  # done_recording is False for the first "Recorder.reset_cooldown" seconds even if write hasn't been called
            or self._detected_sound_anamoly
            ):
            self._alarm_manager.play_alarm()
            prefix = 'sound_anamoly' if self._detected_sound_anamoly else 'human_spotted_curfew'
            self._recorder.write(frame, prefix=prefix)

        elif done_recording:
            self._recorder.reset()
            self._sender.send(
                email_subject='ALARM BLARED!',
                message='The following attachment is a recording of when the alarm blared (Person during curfew or sound anamoly):',
                file_paths=[last_video_path]
            )
            

        if self._controller_activated_record:
            self._recorder.write(frame, prefix='controller')
        elif self._past_controller_activated_record and not self._controller_activated_record:  # If the user currently stopped recording after they were just recording
            self._recorder.reset()
            self._past_controller_activated_record = self._controller_activated_record
            self._sender.send(
                email_subject='Manual Recording',
                message='The following attachment is a recording:',
                file_paths=[last_video_path]
                                    )
                                    
        # Every 10 frames, check sound anamolies since predictions take time
        try:
            if frames_ran % 10 == 0:
                # Prevents infinite loop where analyzer can detect alarm as anamoly
                if ((self._sound_analyzer.detect_sound_anamoly() and not self._alarm_manager.get_busy())
                    # Keeps detected_sound_anamoly attribute as True if the Recorder object isn't done recording.
                or (self._detected_sound_anamoly and not done_recording)):  
                    self._detected_sound_anamoly = True
                else:
                    self._detected_sound_anamoly = False
        except NotFittedError:
            pass

    def _handle_sound_analysis(self) -> None:
        """Execute certain methods of the SoundAnalyzer class"""
        if self._training_sound_analyzer:
            self._sound_analyzer.listen_for_training()

        if self._sound_analyzer.sound_samples and not self._training_sound_analyzer:  # If samples were recorded and training was turned off
            self._sound_analyzer.train()

    def _handle_recognizing_commands(self) -> None:
        """Execute methods of the SpeechRecognition class relating to mapping voice inputs to functions."""
        # FOR LLM API REQUESTS
        method_map = {
            'get_name_of_known_face': lambda: self._face_analyzer.get_name_of_known_face(self._current_known_face),
            'create_graph_of_unknown_faces': self._send_email_of_unknown_faces_graph
        }

        # FOR PRESET COMMANDS
        command_map = {
            'say my name': lambda: self._face_analyzer.get_name_of_known_face(self._current_known_face),
            'email unknown faces cluster': self._send_email_of_unknown_faces_graph
        }

        if not hasattr(self, "_active_listening_future") or self._active_listening_future is None:
            def on_listening_done(future):
                result = future.result()
                
                self._current_words_spoken.update({'words': result})
                self._active_listening_future = None

            # Listens concurrently
            if self._known_face_spotted:
                self._active_listening_future = self._listening_pool.submit(self._speech_recognizer.listen)
                self._active_listening_future.add_done_callback(lambda f: on_listening_done(f))

        if self._current_words_spoken['words'] is not None:
            spoken_text = self._current_words_spoken['words']
            self._current_words_spoken['words'] = None
            clean_words = [word.lower().strip(",.?!") for word in spoken_text.split()]  # Remove punctuation
            
            if self._wake_word.lower() in clean_words:
                try: 
                    response = self._speech_recognizer.get_llm_response(spoken_text)

                    if response in method_map:
                        method_to_call = method_map[response]
                        returned_value = method_to_call()
    
                        if isinstance(returned_value, str):
                            self.talk(returned_value)
                    else:
                        self.talk(response)  

                except:  # Manual Phrases are used as a fallback
                    selected_command = None
                    string_clean_words = ' '.join(clean_words)

                    for phrase, method in command_map.items():
                        if phrase in string_clean_words:
                            selected_command = method

                    if selected_command is not None:
                        returned_value = selected_command()

                        if isinstance(returned_value, str):
                            self.talk(returned_value)
        
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

            self._handle_recording_and_alarm(frame, frames_ran=frames_ran)

            self._handle_sound_analysis()

            self._handle_recognizing_commands()

            # Send emails that failed to send
            if frames_ran % 1000 == 0:
                self._sender.send_pending_emails()
            
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
