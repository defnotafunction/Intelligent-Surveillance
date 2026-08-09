from deepface import DeepFace
from numpy import ndarray
from os import path, listdir
from numpy import save, load
from sklearn.cluster import DBSCAN


SRC_DIR = path.dirname(path.abspath(__file__)) 
BASE_DIR = path.dirname(SRC_DIR)

class FaceAnalyzer:
    def __init__(self):
        self.model = DBSCAN(
            min_samples=1,  # One face can be its own cluster
            n_jobs=-1  # Uses all processors
        )

    def get_faces_match(self, img1: ndarray, img2: ndarray) -> bool:
        """
        Checks if the two images include the same face.

        Args:
            img1: An array representing an image to compare with the second.
            img2: An array representing image to compare with the first.
        
        Returns:
            A boolean indicating whether the two images include the same face
        """

        try:
            return DeepFace.verify(img1.copy(), img2.copy())['verified']  # Use copies of the images to prevent memory locking
        except:
            return None

    def save_unknown_face(self, unknown_face: ndarray) -> None:
        """
        Saves an embedded array of an unknown face. 
        This method should only be used for unknown faces!

        Args:
            unknown_face: An array that represents the image including the unknown face.
        """
        folder_path = path.join(BASE_DIR, 'data', 'faces', 'unknown_faces')

        file_number = len(listdir(folder_path)) + 1
        file_path = path.join(folder_path, f'unknown_face{file_number}.npy')

        save(file_path, unknown_face)

    def get_face_is_in_unknown_faces(self, face: ndarray) -> bool:
        """
            Checks if a face matches with any in the unknown_faces folder.

            Args:
                face: An array that represents the image including the unknown face.
            
            Returns:
                A boolean indicating whether the unknown face was already stored.
        """
        folder_path = path.join(BASE_DIR, 'data', 'faces', 'unknown_faces')
        
        for file_name in listdir(folder_path):
            file_path = path.join(folder_path, file_name)
            face_array = load(file_path)

            if DeepFace.verify(face_array, face, detector_backend='skip'):  # Skip face detecting since face should be an array already.
                return True

        return False

    def get_face_is_in_known_faces(self, face: ndarray) -> bool:
        """
            Checks if a face matches with any in known_faces.

            Args:
                face: An array that represents the image including the known face.
            
            Returns:
                A boolean indicating whether a face is known.
        """
        folder_path = path.join(BASE_DIR, 'data', 'faces', 'known_faces')
        
        for file_name in listdir(folder_path):
            file_path = path.join(folder_path, file_name)
            face_array = load(file_path)

            if DeepFace.verify(face_array, face, detector_backend='skip'):
                return True

        return False

    def fit_clustering_model(self) -> None:
        """Fits a DBSCAN model on unknown faces."""

        folder_path = path.join(BASE_DIR, 'data', 'faces', 'unknown_faces')

        face_arrays = []
        for file_name in listdir(folder_path):
            face_arrays.append(load(path.join(folder_path, file_name)))

        self.model.fit(face_arrays)
            

