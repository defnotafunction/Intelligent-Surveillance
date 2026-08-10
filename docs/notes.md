## [8/8/2026]
- **Thoughts**: 
    - I started on the Density-Based clustering for unknown faces, I decided to assign the min_samples (number of samples in a neighborhood for a point to be considered a core point) to the value one since it might not get the chance to capture faces of unknown people more than once. I could use DeepFace to create lists of where each held faces that matched each other but it isn't always accurate and wouldn't provide as much insight. With DBSCAN, I could create graphs or other forms of data visualization.
    - Flow of identifying images: OpenCV's frame array -> DeepFace verification (embeds images under the hood)
    - Faces will be split into two groups, known faces, and unknown faces, faces that are brand new (not in unknown faces) will be instantly assigned to unknown faces.
- **Problems and Fixes**: 
    - Couldn't install the facial-recognition library, installed deepface instead. 
    - OpenCv's installation folder didn't include the haarcascade model, had to manually install it myself.
    - Deepface's verification can take some time so I'm planning on using the threading library for methods that use it.
        - Multithreading was not needed, just calling methods with Deepface once every period should be enough.

## [8/9/2026]
- **Thoughts**:    
    - Face recognition is close to being finished, once enabling the option to remember faces, the program will listen for the phrase "My name is" and extract the text after it then store that face into known_faces then store the name into known_face_data.json, the face and name are linked by a shared number. This was a huge step in this project.
        - e.g "known_face1.npy" in known_faces and "Face1" in known_face_data.json
    
- **Problems and Fixes**:
    - Pyttsx3's engine object can't run its say method more than once, I don't know why, but I'll have to reassign Camera's tts_engine attribute every time after it's used.
    - Deepface's verify method returned true, every face it saw looked like the one stored in the folder known_faces apparently, so I used FaceAnalyzer's get_face_match method, the only difference between the two is that get_face_match doesn't skip face detection (although it isn't needed). The result was it apparently didn't recognize me at all, it kept running the block of code that's reserved for faces that don't match with any in the data.
        - Error was being thrown, which I made return None using try-except.
        - Fixed it by assiging the enforce_detection argument to False and the detector_backend argument to "skip", it now matches faces successfully; however, interestingly and concerningly, it does confuse dark-skinned individuals to be the same person, I might change the model it uses.
        - I tried using different, more expensive face recognition models. ArcFace identified every one as me, and FaceNet512 identified nobody as me, not even me. Also, using those models will cost me speed and processing power, and since this project is intended to be ran on less powerful devices, I will use SFace, it has proved itself to be able to distinguish between faces despite them having similar skin tones.

# [8/10/2026]
- **Thoughts**: 
    - Renamed the empty microphone.py file to alarm.py and created an AlarmManager class to handle curfew times, recording, and playing alarms in specific cases such as when someone is spotted during curfew.
    - Grabbed a random sound pixabay to be used as the security alarm, so far it'll play when somebody is spotted during curfew.

- **Problems and Fixes**:
    - ...