# Intelligent Surveillance
- An ML-powered Surveillance program that adapts to faces and environments, automatically records and plays alarms, and reports alarming information via email.

## Devices Required
- Camera
- Microphone
- Game Controller (Preferably Xbox Series or Dualsense)
- Speaker (Optional but recommended)

## Controls
- Controller (Xbox Series Controller)
    - A: Quit
    - B: Enable remembering faces
        - Directions: Show your face on-screen and say the phrase "my name is" with your name after.
    - X: Get count of how many times the unknown face currently detected by the camera was tracked.
    - Y: Begin / Stop recording.
    - LB: Train Sound Analyzer's model on normal sounds for anomaly detection.
    - RB: Reset Sound Analyzer's model.
    - View: Send a graph containing grouped data points that represent unknown faces along with the number of times that each face was tracked.

- Speech Recognition
    - LLM / Command wake up call: "Paper"
    - Preset Commands:
        - "Say my name": Announces the name of the known face detected by the camera.
        - "Email unknown faces cluster": Emails a graph containing grouped data points that represent faces.

# Core Features
- Live Camera Surveillance using `opencv-python` (`OpenCV`).
- Human detection, facial recognition using `DeepFace` and `OpenCV`.
- Unknown face tracking powered by `scikit-learn`'s `DBSCAN` and `TSNE`.
- Automatic Recording + Alarm.
- Sound anomaly detection powered by `scikit-learn`'s `SGDOneClassSVM`.
- Controller input handling using `pygame`.
- Speech recognition using `SpeechRecognition`.
- Text-to-speech feedback using `pyttsx3`.
- Email delivery using `Yagmail`.
- Conversational responses and command calling using `Groq`. (Only when a known face is recognized)

## Features (in-depth)
- Human Detection
- Facial Recognition, Detection, and Memory:
    - Storing Faces:
        1. Instantly stores a face into a folder of unknown faces if that face isn't recognized or spotted in its data.
        2. If a face is already recognized in a folder containing unknown faces, it gets stored again randomly.
        3. Once a face and their name are remembered and rendered as a "known face", their data is stored, and every instance of that face is removed from the folder containing unknown faces.
    
    - Tracking:
        - Uses density clustering (`StandardScaler` -> `PCA` -> `DBSCAN`) to identify and group unknown faces, and dimensionality reduction (`TSNE`/t-SNE) for graphing them.

- Recording + Alarm:
    - Activated once a person is detected during defined curfew hours or a sound anomaly is spotted.
    - Recording can be activated and the resulting video is sent via email.
    - An email with the recorded video is sent to the Gmail account specified in the environment file.

- Sound Anomaly Detection:
    - An unsupervised SVM (`SGDOneClassSVM` from `scikit-learn`) can be trained on audio data through a microphone to detect anomalies.
    - The model can be reset and retrained using controller inputs.
    - The trained model is saved and reused between program runs.

- Email Alerts:
    - Emails are sent only through Gmail using credentials from the environment file.
    - Alerts include alarm recordings, manual recordings, or graphs for identifying unknown faces.
    - If there isn't an internet connection, emails and their contents are stored, and are then reconstructed and sent periodically.

- Voice Feedback and Speech Recognition:
    - Known faces are announced by name periodically.
    - The number of times an unknown face, visible to the camera, has been captured can be announced through the speaker.
    - `Groq` is used once speech including the wake up call for commands is recognized, and if a known face was recently detected by the camera.
        - If there isn't an internet connection and thus sending API requests isn't possible, the program uses preset commands if the wake up call is recognized and a known face was recently detected by the camera.

- Runtime Management:
    - Required data folders and metadata files are created automatically when missing.
    - Camera, microphone, audio, controller, and display resources are released on exit.

## File Structure
```
|-- README.md
|-- assets/
|   `-- audios/
|       `-- security-alarm.mp3
|-- data/
|   |-- emaildata/
|   |-- face_arrays/
|   |   |-- known_faces/
|   |   `-- unknown_faces/
|   |    
|   |-- face_to_data/
|   |   `-- known_face_data.json
|   |-- graphs/                  
|   |-- models/
|   `-- videos/
|
|-- docs/
|   |-- dev-log.md
|   |-- notes.md
|   `-- PROJECT_PLAN.md
|-- logs/
`-- src/
    |-- __init__.py
    |-- alarm.py                    # Handles alarm system.
    |-- camera.py                   # Captures frames from camera, uses methods from other classes.
    |-- controller.py               # Includes methods for tracking controller inputs using pygame.
    |-- execeptions.py              # Includes custom exceptions for special cases.
    |-- facial_recognition.py       # Includes methods for recognizing faces, density clustering unknown faces, and storing faces.
    |-- main.py                     # Main entry point.
    |-- microphone.py               # Contains methods for listening to audio data and recognizing speech.
    |-- recorder.py                 # Handles logic for recording and saving videos.
    `-- sender.py                   # Handles logic for sending emails containing content.
```
