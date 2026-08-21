from deepface import DeepFace
import numpy as np
from os import path, listdir, remove, makedirs
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import cv2
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import logging

SRC_DIR = path.dirname(path.abspath(__file__)) 
BASE_DIR = path.dirname(SRC_DIR)

class FaceAnalyzer:
    def __init__(self):
        self.clustering_model = DBSCAN(
            eps=10,
            min_samples=1,  # One face can be its own cluster
            n_jobs=-1  # Uses all processors
        )
        self.create_missing_folders_and_files()
        self.deepface_model_name = 'SFace'
        known_face_data_path = path.join(BASE_DIR, 'data', 'face_to_data', 'known_face_data.json')

        # IN CASE FILE IS EMPTY
        try:
            with open(known_face_data_path, "r", encoding="utf-8") as file:
                self.data_of_known_faces = json.load(file)
        except json.JSONDecodeError:
            self.data_of_known_faces = {}

    def create_missing_folders_and_files(self) -> None:
        """Creates folders and files that the FaceAnalyzer class needs to use."""

        required_folders = [
            path.join(BASE_DIR, 'data', 'face_arrays'),
            path.join(BASE_DIR, 'data', 'face_arrays', 'known_faces'),
            path.join(BASE_DIR, 'data', 'face_arrays', 'unknown_faces'),
            path.join(BASE_DIR, 'data', 'face_to_data'),
            path.join(BASE_DIR, 'data', 'models'),
            path.join(BASE_DIR, 'data', 'graphs')

        ]

        required_files = [
            path.join(BASE_DIR, 'data', 'face_to_data', 'known_face_data.json')
        ]

        for folder in required_folders:
            makedirs(folder, exist_ok=True)

        for file in required_files:
            if not path.exists(file):
                with open(file, "w", encoding="utf-8") as file:
                    pass
        
    def get_faces_match(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.35) -> bool:
        """
        Checks if the two images include the same face.

        Args:
            img1: An array representing an image to compare with the second.
            img2: An array representing image to compare with the first.
            threshold: An integer between 0 and 1, if the distance between the images are less than the threshold, then the function will return True.
        
        Returns:
            A boolean indicating whether the two images include the same face
        """

        # CONVERT BGR ARRAYS INTO RGB
        rgb_img1 = cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2RGB)
        rgb_img2 = cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2RGB)

        distance = DeepFace.verify(
            rgb_img1,
            rgb_img2,
            enforce_detection=False,
            detector_backend='skip',  # Skips detecting face because image is already an array of a cropped face
            model_name=self.deepface_model_name
            )['distance']

        return distance <= threshold
        

    def save_unknown_face(self, unknown_face: np.ndarray) -> None:
        """
        Saves an array of an unknown face into the folder containing unknown faces. 
        This method should only be used for unknown faces!

        Args:
            unknown_face: An array that represents the image including the unknown face.
        """
        folder_path = path.join(BASE_DIR, 'data', 'face_arrays', 'unknown_faces')

        file_number = len(listdir(folder_path)) + 1
        file_path = path.join(folder_path, f'unknown_face{file_number}.npy')

        np.save(file_path, unknown_face)

    def store_face_into_known_faces(self, face: np.ndarray, name: str) -> None:
        """
        Saves an array of a face into the folder containing known faces and saves data of the person in the folder linking data to known faces.

        Args:
            face: An array that represents the image including the face.
            name: Name of the person with the face.
        """
        known_faces_path = path.join(BASE_DIR, 'data', 'face_arrays', 'known_faces')

        known_face_number = len(listdir(known_faces_path)) + 1
        known_face_file_path = path.join(known_faces_path, f'known_face{known_face_number}.npy')

        np.save(known_face_file_path, face)

        self.data_of_known_faces[f'Face{known_face_number}'] = name  # Faces and data will be linked using a shared number
        known_face_data_path = path.join(BASE_DIR, 'data', 'face_to_data', 'known_face_data.json')

        with open(known_face_data_path, 'w') as file:
            json.dump(self.data_of_known_faces, file, indent=4)


    def remove_face_from_unknown_faces(self, face: np.ndarray) -> None:
        """
        Removes a face from every image included in the folder containing unknown faces. 
        This method should only be used for unknown faces!

        Args:
            unknown_face: An array that represents the image including the unknown face.
        """

        folder_path = path.join(BASE_DIR, 'data', 'face_arrays', 'unknown_faces')

        for file_name in listdir(folder_path):
            file_path = path.join(folder_path, file_name)
            face_array = np.load(file_path)

            if self.get_faces_match(face_array, face):  # Skip face detecting since face should be an array already.
                remove(file_path)

    def get_face_is_in_unknown_faces(self, face: np.ndarray) -> bool:
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
            face_array = np.load(file_path)

            if self.get_faces_match(face_array, face): 
                return True

        return False

    def get_face_is_in_known_faces(self, face: np.ndarray) -> bool:
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
            face_array = np.load(file_path)

            if self.get_faces_match(face_array, face):
                return True

        return False

    def remember_face(self, face: np.ndarray, name_of_person: str) -> None:
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

    def get_name_of_known_face(self, face: np.ndarray) -> str:
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
            face_array = np.load(file_path)

            if self.get_faces_match(face_array, face):
                face_number = file_name[10:-4]
                break

        return self.data_of_known_faces[f'Face{face_number}']

    def get_embeddings_of_files_in_directory(self, folder_path: str) -> list[np.ndarray]:
        """
        Uses Deepface to embed the array files in a folder.

        Args:
            folder_path: A string that represents the path to the folder containing the files.

        Returns:
            A list of the embeddings.
        """
        embeddings = []
        
        for file_name in sorted(listdir(folder_path)):
            face_array = np.load(path.join(folder_path, file_name))
            objs = DeepFace.represent(
                img_path=face_array,
                model_name=self.deepface_model_name,
                enforce_detection=False,
                detector_backend='skip'
            )
            vector = objs[0]['embedding']

            embeddings.append(vector)

        return embeddings

    def get_cluster_predictions_of_unknown_faces(self) -> np.ndarray:
        """Uses Standardization and PCA to fit a DBSCAN model on the folder containing unknown faces and returns its predictions."""
        folder_path = path.join(BASE_DIR, 'data', 'face_arrays', 'unknown_faces')
        pca = PCA(n_components=7)
        embeddings = self.get_embeddings_of_files_in_directory(folder_path)
        
        embeddings_scaled = StandardScaler().fit_transform(X=embeddings)
        projected_embeddings = pca.fit_transform(embeddings_scaled)
        clusters = self.clustering_model.fit_predict(projected_embeddings)

        return clusters

    def get_unknown_face_count(self, face: np.ndarray) -> int:
        """
        Returns an approximation of how many times a certain face appears in unknown_faces based off the predictions of a clustering model.

        Args:
            face: An array that holds the image with a face.

        Returns:
            An integer that represents how many times a face appears in the folder unknown_faces.
        """
        face_idx = None
        folder_path = path.join(BASE_DIR, 'data', 'face_arrays', 'unknown_faces')
        for idx, file_name in enumerate(sorted(listdir(folder_path))):
            unknown_face = np.load(path.join(folder_path, file_name))

            if self.get_faces_match(unknown_face, face):
                face_idx = idx

        if face_idx is None:
            return 0
        
        labels = self.get_cluster_predictions_of_unknown_faces()

        # Labels should be a parallel list to the list of the files in unknown_faces
        label_of_face = labels[face_idx]
        count = np.count_nonzero(labels == label_of_face)

        return count

    def create_graph_of_unknown_faces(self) -> str:
        """
        Use t-sne and density clustering to create a file with a graph with every unknown face as a point and color-coded clusters.

        Returns:
            A file path leading to a png of the newly created graph.
        """
        unknown_face_directory = path.join(BASE_DIR, 'data', 'face_arrays', 'unknown_faces')
        unknown_faces_files = [path.join(unknown_face_directory, filename) for filename in sorted(listdir(unknown_face_directory))]
        embeddings = self.get_embeddings_of_files_in_directory(unknown_face_directory)
        clusters = self.get_cluster_predictions_of_unknown_faces()

        # T-SNE, or most dimensionality reduction techniques in general, need scaling to perform well.
        embeddings_scaled = StandardScaler().fit_transform(embeddings)

        tsne = TSNE(n_components=2, perplexity=5, random_state=42)

        embeddings_2d = tsne.fit_transform(embeddings_scaled)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=clusters,
            cmap='Set2',
            s=35,
            alpha=0.7
              )

        unique_clusters = np.unique(clusters)
        cmap = scatter.cmap
        norm = scatter.norm

        # Lists for creating the legend for face counts
        count_labels = []
        
        for cluster_id in unique_clusters:
            cluster_color = cmap(norm(cluster_id))
            cluster_count = np.sum(clusters == cluster_id)
            count_labels.append(f"Face {cluster_id}: Captured {cluster_count} times")

            # Position for face image
            points_in_cluster = embeddings_2d[clusters == cluster_id]
            center_x = np.mean(points_in_cluster[:, 0])
            center_y = np.mean(points_in_cluster[:, 1])

            # Picture to use for cluster will be first instance of face according to DBSCAN
            index_for_face = list(clusters).index(cluster_id)
            first_file_of_face = unknown_faces_files[index_for_face]
            bgr_face = np.load(first_file_of_face)
            rgb_face = bgr_face[:, :, ::-1]
            face_image = Image.fromarray(rgb_face)
            face_image = face_image.resize((50, 50), Image.Resampling.LANCZOS)
            
            imagebox = OffsetImage(face_image, zoom=.4)
            ab = AnnotationBbox(imagebox, (center_x, center_y), frameon=True)
            
            # Set the frame line color to match the color of its cluster
            ab.patch.set_edgecolor(cluster_color)
            ab.patch.set_linewidth(2)

            plt.gca().add_artist(ab)

        # Legend for face counts
        handles, _ = scatter.legend_elements()
        plt.legend(
            handles, 
            count_labels, 
            title="Face Legend", 
            bbox_to_anchor=(1.05, 1), 
            loc='upper left'
        )

        plt.title('Clusters of Unknown faces', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Axis 1')
        plt.ylabel('t-SNE Axis 2')
        plt.grid(True, linestyle='--', alpha=0.3)

        graphs_path = path.join(BASE_DIR, 'data', 'graphs')
        graph_file_name = f'unknown_faces_graph{len(listdir(graphs_path)) + 1}.png'

        graph_file_path = path.join(graphs_path, graph_file_name)
        plt.savefig(graph_file_path, dpi=300, bbox_inches='tight')

        return graph_file_path
