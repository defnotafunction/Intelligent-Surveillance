import numpy as np
import pyaudio
from sklearn.linear_model import SGDOneClassSVM 
import joblib
from os import path

SRC_DIR = path.dirname(path.abspath(__file__)) 
BASE_DIR = path.dirname(SRC_DIR)

class SoundAnalyzer:
    # ARGUMENTS FOR STREAM
    CHUNK = 1024
    RATE = 44100
    CHANNELS=1

    def __init__(self) -> None:
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        self.model = SGDOneClassSVM(nu=0.01)  # Intended to be trained on normal and frequent sounds, used for sound anamolies
        self.sound_samples = []
 
    def save_model(self) -> None:
        """Uses joblib to save the model used for sound anamolies into the models folder."""

        model_path = path.join(BASE_DIR, 'data', 'models', 'oneclasssvm.joblib')
        joblib.dump(self.model, model_path)

    def reset_model(self) -> None:
        """Reassigns the model attribute and saves it."""
        self.model = SGDOneClassSVM(nu=0.01)
        self.save_model()

    def listen_for_training(self) -> None:
        """Extracts audio snapshots and stores it in list to use for training the model for anamoly detection."""
        raw_bytes = self.stream.read(self.CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(raw_bytes, dtype=np.float32)

        self.sound_samples.append(audio_array)

    def detect_sound_anamoly(self) -> bool:
        """
        Feeds the model for detecting sound anamolies the current audio chunk.
        This method should be called after calling the listen_for_training and train methods!

        Returns:
            A boolean that indicates if the current audio chunk is an outlier compared to the data it has been trained on.
        """
        raw_bytes = self.stream.read(self.CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(raw_bytes, dtype=np.float32)

        prediction = self.model.predict([audio_array])

        return prediction == -1  # ONECLASSSVM RETURNS -1 IF INPUT IS OUTLIER
        
    def train(self) -> None:
        """
        Fits a model for anamoly detection on a list containing audio snapshots.
        The listen method must be called before this one, this method also reassigns the sound_samples to an empty list.
        """
        self.model.partial_fit(self.sound_samples)
        self.sound_samples = []

    def kill(self) -> None:
        """Terminates pyaudio."""
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()