## [8/8/2026]
- **Thoughts**: 
    - I started on the Density-Based clustering for unknown faces, I decided to assign the min_samples (number of samples in a neighborhood for a point to be considered a core point) to the value one since it might not get the chance to capture faces of unknown people more than once. I could use DeepFace to create lists of where each held faces that matched each other but it isn't always accurate and wouldn't provide as much insight. With DBSCAN, I could create graphs or other forms of data visualization.
    - Flow of identifying images: OpenCV's frame array -> DeepFace verification (embeds images under the hood)
    - Deepface's verification can take some time so I'm planning on using the threading library for methods that use it.
    - Faces will be split into two groups, known faces, and unknown faces, faces that are brand new either will be automatically assigned to unknown faces.
- **Mishaps and Fixes**: 
    - Couldn't install the facial-recognition library, installed deepface instead. 
    - OpenCv's installation folder didn't include the haarcascade model, had to manually install it myself.