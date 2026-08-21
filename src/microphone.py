import numpy as np
import pyaudio
from sklearn.linear_model import SGDOneClassSVM 
import joblib
from os import path, getenv
import speech_recognition as sr
from groq import Groq
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import io

load_dotenv()

SRC_DIR = path.dirname(path.abspath(__file__)) 
BASE_DIR = path.dirname(SRC_DIR)

class SoundAnalyzer:
    # ARGUMENTS FOR STREAM
    CHUNK = 1024
    RATE = 44100
    CHANNELS=1
    MODEL_PATH = path.join(BASE_DIR, 'data', 'models', 'oneclasssvm.joblib')

    def __init__(self) -> None:
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        if path.exists(self.MODEL_PATH):
            self.model = joblib.load(self.MODEL_PATH)
        else:
            self.model = SGDOneClassSVM(nu=0.01)  # Intended to be trained on normal and frequent sounds, used for sound anamolies

        self.sound_samples = []
 
    def save_model(self) -> None:
        """Uses joblib to save the model used for sound anamolies into the models folder."""
        joblib.dump(self.model, self.MODEL_PATH)

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

        prediction = self.model.predict([audio_array])[0]

        return prediction == -1  # ONECLASSSVM RETURNS -1 IF SAMPLE IS OUTLIER
        
    def train(self) -> None:
        """
        Fits a model for anamoly detection on a list containing audio snapshots.
        The listen method must be called before this one, this method also reassigns the sound_samples to an empty list.
        """
        self.model.partial_fit(self.sound_samples)
        self.sound_samples = []
        self.save_model()

    def kill(self) -> None:
        """Terminates pyaudio."""
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

class SpeechRecognition:
    GROQ_API_KEY = getenv('GROQ_API_KEY')

    def __init__(self) -> None:
        self._speech_recognizer = sr.Recognizer()
        self._speech_recognizer.pause_threshold = 1.5
        self._local_whisper = WhisperModel(
            "base.en", 
            device="cpu", 
            compute_type="int8",
            cpu_threads=4
        )

        self.client = Groq(
            api_key=self.GROQ_API_KEY
        )


    def listen(self) -> str:
        with sr.Microphone() as source:
            self._speech_recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio = self._speech_recognizer.listen(source)

        wav_bytes = audio.get_wav_data()
        wav_stream = io.BytesIO(wav_bytes)

        segments, info = self._local_whisper.transcribe(wav_stream, beam_size=5)
            
            
        text = "".join([segment.text for segment in segments])
            
        return text.lower().strip()

    def get_llm_response(self, user_prompt: str) -> str:
        """
        Sends a request containing methods at the LLM's disposal alongside the user prompt to an LLM, calls a function if the LLM's response included one, otherwise it returns a response.
        
        Args:
            user_prompt: A string containing the prompt of the user.
        
        Returns:
            A string containing the LLM's response.
        """
        prompt_preset = """
                        Your name is paper, assist the user.
                        User prompt:
                                
                        """
        full_prompt = prompt_preset + user_prompt
        completion = self.client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                    ],
                    temperature=1,
                    max_completion_tokens=2048,
                    top_p=1,
                    reasoning_effort="medium",
                    stream=False,
                    stop=None
                )

        response_text = completion.choices[0].message.content.strip()

        return response_text


        