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
- **Challenges**: Deepface is computationally expensive to run, I am planning to use a multithreading library in order for this project to be able to be ran on less powerful devices.
- **Future steps**: Now that the basics of the controller handling is finished, I will further develop facial recognition features and storing data for each authenticated face, and unknown faces will be tracked using density clustering.