# Developer Log

## [8/7/2026]
-  **Work done**: Initialized Git repository, created project layout, begined video capturing in camera.py.
- **Challenges**: None. (We're just getting started)
- **Future steps**: This project will initially consist of sensors such as a camera, microphone, and a game controller (for quick configuration). It will output data using a speaker. 
    - CORE features to implement (Roughly in order):
        - Human detection and facial recognition
        - Sound irregularity detection
        - Recording and alarm feature that activates during certain events

## [8/8/2026]
- **Work done**: Used functions and classes to abstract code, implemented logging, and added files like main.py, execeptions.py, and camera.log. I also added a data folder containing known and unknown faces for the program to use. I created the base for detecting controller inputs. I implemented face detection, and started on facial recognition and tracking unknown faces.
- **Challenges**: Deepface is computationally expensive to run. I planned to use a multithreading library in order for this project to be able to be ran on less powerful devices, but separating methods that uses Deepface from regular face detection logic and calling those methods with Deepface periodically using a looping counter proved to work.
- **Future steps**: Now that the basics of the controller handling is finished, I will further develop facial recognition features and storing data for each authenticated face, and unknown faces will be tracked using density clustering.

## [8/9/2026]
- **Work done**: Implemented face recognition and storing names for each face, the controller is used to enable the ability to start remembering faces. Faces are stored in known faces once the person says the phrase 'My name is (their name)', then every instance of that face is removed from unknown faces. I also implemented text-to-speech so remembered faces will have their names called.
- **Challenges**: Deepface's method "verify" initially identified every face as the one in the known_faces folder, it then identified every face as a brand new one as a result of my try-except block that returned None if an exception was raised, which exceptions were actually being raised on every call. Finally, Deepface's default model has shown to exhibit racial bias, so I used SFace and assigned a strict threshold to it which worked much better.
- **Future steps**: Tracking and identifying unknown faces using density clustering, implementing a recording and alarm feature.

## [8/10/2026]
- **Work done**: Created alarm.py, recorder.py, the Recorder and AlarmManager class, the program now plays an alarm and records a video when someone is spotted during user-defined curfew hours.
- **Challenges**: ...
- **Future steps**: Tracking and identifying unknown faces using density clustering

## [8/12/2026]
- **Work done**: Implemented attributes and methods in Recorder that dictate when a video file can stop being written to. It ensures that all video files are about the same length and that a new video file isn't created every time the face and pedistrian detectors randomly toggle on and off. Add a method that returns the frequency of a face appearing in unknown faces using DBSCAN clustering.
- **Challenges**: ...
- **Future steps**: Activate recording when decibels are unusually high, this will lead into environmental adaptation. Continue working on clustering unknown faces.

## [8/16/2026]
- **Work done**: Created the SoundAnalyzer class and methods to detect anamolies in sound using OneClassSVM.
- **Challenges**: Alarm blares when sound anamoly is detected, SoundAnalyzer can consider an alarm as an anamoly, creating an infinite loop. Fixed by only considering a sound as a anamoly when alarm isn't playing.
- **Future steps**: Implement LLM features and finish up the facial recognition part of the project.

## [8/17/2026]
- **Work done**: Fixed recording after detecting anamolies in sound. Created sender.py and the GmailSender class, so after recording, the video is sent via email.
- **Challenges**: Problems kept popping up with the recording mechanisms.
- **Future steps**: Implement LLM features, finish up the facial recognition part of the project, especially the density clustering part.

## [8/18/2026]
- **Work done**: Created a method for creating graphs of the result of density clustering unknown faces. The program also sends them after detecting a specific controller input.
- **Challenges**: Fixing a problem with the email sending feature, emails were sent every frame after a video was finished recording.
- **Future steps**: Make program listen to pre-defined phrases instead of implementing LLM features.

## [8/19/2026]
- **Work done**: Implemented PCA for density clustering unknown faces to reduce noise, increased epsilon for DBSCAN. Created SpeechRecognition class in microphone.py to prepare for adding pre-defined commands and communication with an LLM.
- **Challenges**: Selecting the right amount of principle components and epsilon size.
- **Future steps**: Make program have an open ear for pre-defined commands or prompts for an LLM, may use multithreading; if it's too computationally expensive, I'll map the feature to an input on the controller instead.

## [8/20/2026]
- **Work done**: Improved unknown faces graph design, and added a legend that contains the number of instances of each distinct face (according to DBSCAN). Used faster_whisper for speech recognition, Groq for requests to LLM, and threading for the program to listen for speech in the background.
- **Challenges**: ...
- **Future steps**: Allow the LLM to call functions or methods, and add pre-defined phrases for the program to listen for. Map a controller input to toggle between using the LLM or pre-defined phrases for commands.