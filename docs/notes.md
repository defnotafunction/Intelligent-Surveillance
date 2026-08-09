## [8/8/2026]
- **Thoughts**: 
    - I started on the Density-Based clustering for unknown faces, I decided to assign the min_samples (number of samples in a neighborhood for a point to be considered a core point) to the value one since it might not get the chance to capture faces of unknown people more than once. I could use DeepFace to create lists of where each held faces that matched each other but it isn't always accurate and wouldn't provide as much insight. With DBSCAN, I could create graphs or other forms of data visualization.
    - Flow of identifying images: OpenCV's frame array -> DeepFace verification (embeds images under the hood)
    - Faces will be split into two groups, known faces, and unknown faces, faces that are brand new either will be automatically assigned to unknown faces.
- **Problems and Fixes**: 
    - Couldn't install the facial-recognition library, installed deepface instead. 
    - OpenCv's installation folder didn't include the haarcascade model, had to manually install it myself.
    - Deepface's verification can take some time so I'm planning on using the threading library for methods that use it.
        - Multithreading was not needed, just calling methods with Deepface once every period should be enough.

## [8/9/2026]
- **Thoughts**:
    - Face recognition is close to being finished, once enabling the option to remember faces, the program will listen for the phrase "My name is" and extract the text after it then store that face into known_faces then store the name into known_face_data.json, the face and name are linked by a shared number. This was a huge step in this project.
        - e.g "known_face1.npy" in known_faces and "Face1" in known_face_data.json
    
- **Problems and Fixes**: ...