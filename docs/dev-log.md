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
- **Challenges**: Deepface is computationally expensive to run. I planned to use a multithreading library in order for this project to be able to be ran on less powerful devices, but separating methods that uses Deepface from regular face detection logic and calling those methods with Deepface every once and a while proved to work.
- **Future steps**: Now that the basics of the controller handling is finished, I will further develop facial recognition features and storing data for each authenticated face, and unknown faces will be tracked using density clustering.

## [8/9/2026]
- **Work done**: Implement face recognition and storing names for each face, the controller is used to enable the ability to start remembering faces. Faces are remembered once the person says the phrase 'My name is (their name)', and every instance of that face is removed from unknown faces. I also implemented text-to-speech so remembered faces will have their names called.
- **Challenges**: ...
- **Future steps**: ...