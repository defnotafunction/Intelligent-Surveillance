# Developer Log

## [8/7/2026]
-  **Work done**: Initialized Git repository, created project layout, begin video capturing in camera.py.
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
- **Work done**: Implement face recognition and storing names for each face, the controller is used to enable the ability to start remembering faces. Faces are stored in known faces once the person says the phrase 'My name is (their name)', then every instance of that face is removed from unknown faces. I also implemented text-to-speech so remembered faces will have their names called.
- **Challenges**: Deepface's method "verify" initially identified every face as the one in the known_faces folder, it then identified every face as a brand new one as a result of my try-except block that returned None if an exception was raised, which exceptions were actually being raised on every call. Finally, Deepface's default model has shown to exhibit racial bias, so I used SFace and assigned a strict threshold to it which worked much better.
- **Future steps**: Tracking and identifying unknown faces using density clustering, implementing a recording and alarm feature.

## [8/10/2026]
- **Work done**: Create alarm.py, recorder.py, the Recorder and AlarmManager class, the program now plays an alarm and records a video when someone is spotted during user-defined curfew hours.
- **Challenges**: ...
- **Future steps**: Tracking and identifying unknown faces using density clustering

## [8/12/2026]
- **Work done**: Implemented attributes and methods in Recorder that dictate when a video file can stop being written to. It ensures that all video files are about the same length and that a new video file isn't created every time the face and pedistrian detectors randomly toggle on and off. Add a method that returns the frequency of a face appearing in unknown faces using DBSCAN clustering.
- **Challenges**: ...
- **Future steps**: Activate recording when decibels are unusually high, this will lead into environmental adaptation. Continue working on clustering unknown faces.