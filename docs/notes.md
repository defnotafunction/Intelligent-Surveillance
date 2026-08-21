# Notes

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

## [8/10/2026]
- **Thoughts**: 
    - Renamed the empty microphone.py file to alarm.py and created an AlarmManager class to handle curfew times, recording, and playing alarms in specific cases such as when someone is spotted during curfew.
    - Grabbed a random sound pixabay to be used as the security alarm, so far it'll play when somebody is spotted during curfew.

- **Problems and Fixes**:
    - ...

## [8/12/2026]
- **Thoughts**:
    - Implemented attributes and methods in Recorder that dictate when a video file can stop being written to.
    - I'm going to start adding features to identify tracked unknown faces using density clustering, the arrays from unknown_faces will be loaded and embedded, then the model will fit on it. The first method that uses DBSCAN will be get_unknown_face_count, it will simply return how many points a cluster of a face has.
    - Controller can now be used to start recording.

- **Problems and Fixes**:
    - VideoWriter creates a video file as soon as its initalized, leaving an empty file if I don't write any frames to it. I'll try fixing it by only initalizing the VideoWriter object when the Recorder's write method is called.
        - The problem was fixed, however, the face and pedestrian detection can randomly toggle on and off, leaving random, short clips in the video folder. To fix it, I'll give the Recorder class a limit on how quickly it can stop recording or reset.

## [8/16/2026]
- **Thoughts**:
    - Creating the SoundAnalyzer class, it will detect anamolies in sound using PyAudio and a OneClassSVM, the "nu" argument, which acts as the maximum allowed percentage of misclassifications, should be set to a low value since the model is intended to be trained on sounds that are frequent and normal. 

- **Problems and Fixes**:
    - An alarm is blared whenever a sound anamoly is detected; however, the microphone can pick up on the alarm and consider it an anamoly, resulting in an infinite loop.
        - Fixed it by only assigning the detected_sound_anamoly attribute of the Camera class to True if the AlarmManager's get_busy method returned False. In other words I used the pygame's mixer module to check if the alarm was still blaring so sound anamolies won't be considered detected when the alarm plays.

## [8/17/2026]
- **Thoughts**:
    - ...

- **Problems and Fixes**:
    - The recorder object uses the reset method if the controller didn't activate recording, which means whenever an alarm blares, the video file will stop having frames added because it's going through that if block. To fix this, I added a boolean variable that captures the past input of the controller's recording attribute, I changed the if statement leading to the reset method to only execute its nested segment if the controller didn't activate recording and its past input was recording. In other words, it only resets when the user just stopped recording, not when the controller recording attribute is False in general.
    - get_done_recording method from the Recorder class returns True before the first write call; however, doing this makes sending emails messy. I added a called_write attribute with the boolean False to the Recorder class that is assigned True after the write method is called and False when the Recorder's reset method is called. This allows the program to have more context about the get_done_recording method since now it returns False during the period of time before the Recorder's cooldown ends even if a video file hasn't been written at all.

## [8/18/2026]
- **Thoughts**:
    - Scratch of plans. I'm not adding the LLM feature because I want to keep most of the project light, local, and deterministic. I'll simply use the speech recognition library to listen for pre-defined commands. Boring, yes. Complicated, no.

- **Problems and Fixes**:
    - Tens of emails are sent, the problem was that Recorder's current time attribute, the one that dictates whether the get_done_recording method is true, is only evaluated and resetted in the write method. This means that between a video being done recording, and a Recorder object's write method being called, an email was being sent every frame.

## [8/19/2026]
- **Thoughts**: 
    - Just learned that os.listdir() returns files in an arbitrary order, I have to change all methods that uses that listdir but rely on consistent order.
    - Change of plans again, we're implementing LLM features, but we're also keeping the recognition to pre-defined commands as a fallback feature (e.g User has no internet connection or LLM API Key)

- **Problems and Fixes**:
    - DBSCAN returned one cluster for each face, I have to increase the epilson argument. I will also use PCA to reduce noise before passing it into DBSCAN for interpretability.


## [8/20/2026]
- **Thoughts**: ... 
    
- **Problems and Fixes**:
    - The vosk model isn't good at speech reconition, I'm switching to faster whisper.