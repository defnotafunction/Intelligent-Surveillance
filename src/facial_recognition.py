from deepface import DeepFace
from numpy import ndarray
from os import path, listdir, remove
from numpy import save, load
from sklearn.cluster import DBSCAN
import json

SRC_DIR = path.dirname(path.abspath(__file__)) 
BASE_DIR = path.dirname(SRC_DIR)

class FaceAnalyzer:
    def __init__(self):
        self.model = DBSCAN(
            min_samples=1,  # One face can be its own cluster
            n_jobs=-1  # Uses all processors
        )

        known_face_data_path = path.join(BASE_DIR, 'data', 'face_to_data', 'known_face_data.json')
        with open(known_face_data_path, "r") as file:
            self.data_of_known_faces = json.load(file)
         
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
        Saves an array of an unknown face into the folder containing unknown faces. 
        This method should only be used for unknown faces!

        Args:
            unknown_face: An array that represents the image including the unknown face.
        """
        folder_path = path.join(BASE_DIR, 'data', 'face_arrays', 'unknown_faces')

        file_number = len(listdir(folder_path)) + 1
        file_path = path.join(folder_path, f'unknown_face{file_number}.npy')

        save(file_path, unknown_face)

    def store_face_into_known_faces(self, face: ndarray, name: str) -> None:
        """
        Saves an array of a face into the folder containing known faces and saves data of the person in the folder linking data to known faces.

        Args:
            face: An array that represents the image including the face.
            name: Name of the person with the face.
        """
        known_faces_path = path.join(BASE_DIR, 'data', 'face_arrays', 'known_faces')

        known_face_number = len(listdir(known_faces_path)) + 1
        known_face_file_path = path.join(known_faces_path, f'known_face{known_face_number}.npy')

        save(known_face_file_path, face)

        self.data_of_known_faces[f'Face{known_face_number}'] = name  # Faces and data will be linked using a shared number
        known_face_data_path = path.join(BASE_DIR, 'data', 'face_to_data', 'known_face_data.json')

        with open(known_face_data_path, 'w') as file:
            json.dump(self.data_of_known_faces, file, indent=4)


    def remove_face_from_unknown_faces(self, face: ndarray) -> None:
        """
        Removes a face from every image included in the folder containing unknown faces. 
        This method should only be used for unknown faces!

        Args:
            unknown_face: An array that represents the image including the unknown face.
        """

        folder_path = path.join(BASE_DIR, 'data', 'face_arrays', 'unknown_faces')

        for file_name in listdir(folder_path):
            file_path = path.join(folder_path, file_name)
            face_array = load(file_path)

            if DeepFace.verify(face_array, face, detector_backend='skip'):  # Skip face detecting since face should be an array already.
                remove(file_path)

    def get_face_is_in_unknown_faces(self, face: ndarray) -> bool:
        """
        Checks if a face matches with any in the folder containing unknown faces.

        Args:
            face: An array that represents the image including the unknown face.
        
        Returns:
            A boolean indicating whether the unknown face was already stored.
        """
        folder_path = path.join(BASE_DIR, 'data', 'face_arrays', 'unknown_faces')
        
        for file_name in listdir(folder_path):
            file_path = path.join(folder_path, file_name)
            face_array = load(file_path)

            if DeepFace.verify(face_array, face, detector_backend='skip'):  # Skip face detecting since face should be an array already.
                return True

        return False

    def get_face_is_in_known_faces(self, face: ndarray) -> bool:
        """
        Checks if a face matches with any in the folder containing known faces.

        Args:
            face: An array that represents the image including the known face.
        
        Returns:
            A boolean indicating whether a face is known.
        """
        folder_path = path.join(BASE_DIR, 'data', 'face_arrays', 'known_faces')
        
        for file_name in listdir(folder_path):
            file_path = path.join(folder_path, file_name)
            face_array = load(file_path)

            if DeepFace.verify(face_array, face, detector_backend='skip'):
                return True

        return False

    def remember_face(self, face: ndarray, name_of_person: str) -> None:
        """
        Saves an array of a face into the folder containing known faces and saves data of the person in the folder linking data to known faces.
        Also removes every image that includes face from unknown faces.

        Args:
            face: An array that represents the image including the face.
            name_of_person: Name of the person with the face.
        """

        if not self.get_face_is_in_known_faces(face):
            self.remove_face_from_unknown_faces(face)
            self.store_face_into_known_faces(face, name=name_of_person)

    def get_name_of_known_face(self, face: ndarray) -> str:
        """
        Gets the name of a known face.
        Must only be used with faces that are known!

        Args:
            face: An array that holds the image with a face.
        
        Returns:
            The name of the person with the face as a string.
        """
        folder_path = path.join(BASE_DIR, 'data', 'face_arrays', 'known_faces')
                
        for file_name in listdir(folder_path):
            file_path = path.join(folder_path, file_name)
            face_array = load(file_path)

            if DeepFace.verify(face_array, face, detector_backend='skip'):
                face_number = file_name[10:-4]
                break

        return self.data_of_known_faces[f'Face{face_number}']

    def fit_clustering_model(self) -> None:
        """Fits a DBSCAN model on the folder containing unknown faces."""

        folder_path = path.join(BASE_DIR, 'data', 'face_arrays', 'unknown_faces')

        face_arrays = []
        for file_name in listdir(folder_path):
            face_arrays.append(load(path.join(folder_path, file_name)))

        self.model.fit(face_arrays)
            

