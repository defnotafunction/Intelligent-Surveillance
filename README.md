# Intelligent Surveillance (W.I.P)
- An AI-powered Surveillance program that adapts to faces and environments, automatically records and plays alarms, and reports alarming information via email.

## Devices Required
- Camera
- Microphone
- Game Controller (Preferably Xbox Series or Dualsense)
- Speaker (Optional)

## Controls (Xbox Series Controller)
- A: Quit
- B: Enable remembering faces
    - Directions: Show your face on-screen and say the phrase "my name is" with your name after.
- X: Get count of unknown face on screen.
- Y: Begin / Stop recording.
- LB: Train Sound Analyzer's model on normal sounds for anomaly detection.
- RB: Reset Sound Analyzer's model.
- View: Send a graph of unknown faces.

# Core Features
- Live Camera Surveillance using `opencv-python` (`OpenCV`).
- Human detection, facial recognition, and face memory using `DeepFace` and `OpenCV`.
- Face tracking powered by `scikit-learn`'s `DBSCAN` and `TSNE`.
- Automatic Recording + Alarm.
- Sound anomaly detection powered by `scikit-learn`'s `SGDOneClassSVM`.
- Controller input handling using `pygame`.
- Speech recognition using `SpeechRecognition`.
- Text-to-speech feedback using `pyttsx3`.
- Email delivery using `Yagmail`.

## Features (in-depth)
- Human Detection
- Facial Recognition, Detection, and Memory:
    - Storing Faces:
        1. Instantly stores a face into a folder of unknown faces if that face isn't recognized or spotted in its data.
        2. If a face is already recognized in a folder containing unknown faces, it gets stored again randomly.
        3. Remembers a face and their name, and removes its instances in a folder of unknown faces.
    
    - Tracking:
        - Uses density clustering (`DBSCAN`) to identify and group unknown faces, and dimensionality reduction (`TSNE`/t-SNE) for graphing them.

- Recording + Alarm:
    - Activated once a person is detected during defined curfew hours or a sound anomaly is spotted.
    - Recording can be activated and the resulting video is sent via email.
    - An email with the recorded video is sent to the Gmail account specified in the environment file.

- Sound Anomaly Detection:
    - An unsupervised SVM (`SGDOneClassSVM` from `scikit-learn`) can be trained on current sounds through a microphone to detect anomalies.
    - The model can be reset and retrained from controller inputs.
    - The trained model is saved and reused between program runs.

- Email Alerts:
    - Emails are sent only through Gmail using credentials from the environment file.
    - Alerts include alarm recordings, manual recordings, or generated unknown-face graphs.

- Voice Feedback:
    - Recognized faces are announced by name.
    - The amount of times an unknown face, visible to the camera, has been captured can be announced through the speaker.

- Runtime Management:
    - Required data folders and metadata files are created automatically when missing.
    - Camera, microphone, audio, controller, and display resources are released on exit.

## File Structure
```
Security Camera/
|-- README.md
|-- assets/
|   `-- audios/
|       `-- security-alarm.mp3
|-- data/
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
    |-- alarm.py
    |-- camera.py
    |-- controller.py
    |-- execeptions.py
    |-- facial_recognition.py
    |-- main.py
    |-- microphone.py
    |-- recorder.py
    `-- sender.py
```
