
import os
import pandas as pd
import face_recognition
import numpy as np

# --- Constants ---
# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths relative to the script's location for robustness
STUDENT_DB_CSV = os.path.join(script_dir, 'students_db.csv')
STUDENT_IMAGES_DIR = os.path.join(script_dir, 'student_images/')

class FaceManager:
    """
    Loads, stores, and matches known student faces.
    """
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_metadata = []
        self._load_student_database()
        self._encode_known_faces()

    def _load_student_database(self):
        """Loads student information from the CSV database."""
        try:
            self.student_db = pd.read_csv(STUDENT_DB_CSV)
            print(f"INFO: Successfully loaded {len(self.student_db)} students from {STUDENT_DB_CSV}.")
        except FileNotFoundError:
            print(f"ERROR: Student database '{STUDENT_DB_CSV}' not found. Please create it.")
            exit()

    def _encode_known_faces(self):
        """Loads student images and creates facial encodings."""
        print("INFO: Encoding student faces from images...")
        if not os.path.exists(STUDENT_IMAGES_DIR):
             print(f"ERROR: Image directory '{STUDENT_IMAGES_DIR}' not found. Please create it.")
             exit()

        for _, row in self.student_db.iterrows():
            roll_no = str(row['roll_no'])
            image_file = next((f for f in os.listdir(STUDENT_IMAGES_DIR) if f.startswith(roll_no + '.')), None)

            if image_file:
                image_path = os.path.join(STUDENT_IMAGES_DIR, image_file)
                try:
                    student_image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(student_image)
                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_metadata.append({'roll_no': roll_no, 'name': row['name'], 'elective': row['elective']})
                    else:
                        print(f"WARNING: No face found in image for Roll No: {roll_no} ({image_file}).")
                except Exception as e:
                    print(f"ERROR: Could not process image {image_path}: {e}")
            else:
                print(f"WARNING: No image file found for Roll No: {roll_no}.")
        print(f"INFO: Encoded {len(self.known_face_encodings)} faces.")

    def find_match(self, face_encoding, tolerance=0.6):
        """
        Compares a given face encoding to all known faces.
        Returns:
            tuple: (bool, dict) indicating if a match was found and the corresponding student metadata.
        """
        if not self.known_face_encodings:
            return False, None
            
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return True, self.known_face_metadata[best_match_index]
            
        return False, None

