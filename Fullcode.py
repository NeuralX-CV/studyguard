import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import argparse
import random

script_dir = os.path.dirname(os.path.abspath(__file__))

STUDENT_DB_CSV = os.path.join(script_dir, 'students_db.csv')
FRAME_PROCESS_INTERVAL = 10
FACE_MATCH_TOLERANCE = 0.6
BEHAVIOR_ANALYSIS_INTERVAL = 90


class BehaviorAnalyzer:
    def __init__(self):
        self.possible_actions = ['sitting', 'standing', 'raising_hand', 'engaged', 'distracted']
        print("INFO: Behavior Analyzer (Simulated) initialized.")

    def analyze_actions(self, frame_sequence):
        if len(frame_sequence) > 0:
            return random.choice(self.possible_actions)
        return 'unknown'


class StudyGuardController:
    def __init__(self, video_source):
        print("INFO: Initializing StudyGuard Controller...")
        self.video_source = video_source
        self.known_face_encodings = []
        self.known_face_metadata = []
        self.student_db = None
        self.student_tracker = {}
        self.behavior_analyzer = BehaviorAnalyzer()
        self.frame_buffer = []
        self.student_images_path = None

        self._load_student_database()
        self._find_and_set_images_path()
        self._encode_known_faces()

    def _load_student_database(self):
        try:
            self.student_db = pd.read_csv(STUDENT_DB_CSV)
            print(f"INFO: Successfully loaded {len(self.student_db)} students from {STUDENT_DB_CSV}.")
        except FileNotFoundError:
            print(f"ERROR: Student database '{STUDENT_DB_CSV}' not found. Please create it.")
            exit()

    def _find_and_set_images_path(self):
        parent_dir = script_dir
        expected_dir_name = 'student_images'
        try:
            for item_name in os.listdir(parent_dir):
                item_path = os.path.join(parent_dir, item_name)
                if os.path.isdir(item_path) and item_name.strip() == expected_dir_name:
                    self.student_images_path = item_path
                    print(f"INFO: Found image directory at '{item_path}' (handling potential whitespace).")
                    return
        except FileNotFoundError:
            pass
        self.student_images_path = os.path.join(parent_dir, expected_dir_name)

    def _encode_known_faces(self):
        print("INFO: Encoding student faces from images...")

        if not os.path.exists(self.student_images_path):
            expected_path = os.path.join(script_dir, 'student_images')
            print(f"ERROR: Image directory '{expected_path}' not found. Please create it.")
            print("TIP: Please ensure the folder is named *exactly* 'student_images' with no typos or extra spaces.")
            exit()

        for index, row in self.student_db.iterrows():
            roll_no = str(row['roll_no'])
            image_file = next((f for f in os.listdir(self.student_images_path) if f.startswith(roll_no + '.')), None)

            if image_file:
                image_path = os.path.join(self.student_images_path, image_file)
                try:
                    student_image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(student_image)
                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_metadata.append({
                            'roll_no': roll_no,
                            'name': row['name'],
                            'elective': row['elective']
                        })
                    else:
                        print(f"WARNING: No face found in image for Roll No: {roll_no} ({image_file}).")
                except Exception as e:
                    print(f"ERROR: Could not process image {image_path}: {e}")
            else:
                print(f"WARNING: No image file found for Roll No: {roll_no}.")
        print(f"INFO: Encoded {len(self.known_face_encodings)} faces.")

    def run_monitoring(self):
        video_capture = cv2.VideoCapture(self.video_source)
        if not video_capture.isOpened():
            print(f"ERROR: Cannot open video source: {self.video_source}")
            return

        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("INFO: End of video stream.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            self.frame_buffer.append(rgb_small_frame)

            if frame_count % FRAME_PROCESS_INTERVAL == 0:
                self._process_frame(rgb_small_frame)

            if frame_count % BEHAVIOR_ANALYSIS_INTERVAL == 0 and len(self.frame_buffer) > 0:
                self._analyze_behavior_in_frame()

            display_frame = self._visualize_data(frame)
            cv2.imshow('StudyGuard - AI Classroom Monitoring', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("INFO: 'q' pressed. Shutting down...")
                break

            frame_count += 1

        video_capture.release()
        cv2.destroyAllWindows()
        self.generate_report()

    def _process_frame(self, frame):
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        current_frame_roll_nos = set()

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding,
                                                     tolerance=FACE_MATCH_TOLERANCE)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) == 0: continue

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                metadata = self.known_face_metadata[best_match_index]
                roll_no = metadata['roll_no']
                current_frame_roll_nos.add(roll_no)

                if roll_no not in self.student_tracker:
                    self.student_tracker[roll_no] = {
                        'metadata': metadata,
                        'entry_time': datetime.now(),
                        'last_seen': datetime.now(),
                        'present': True,
                        'behaviors': set()
                    }
                else:
                    self.student_tracker[roll_no]['last_seen'] = datetime.now()
                    self.student_tracker[roll_no]['present'] = True

        all_tracked_roll_nos = set(self.student_tracker.keys())
        absent_roll_nos = all_tracked_roll_nos - current_frame_roll_nos
        for roll_no in absent_roll_nos:
            self.student_tracker[roll_no]['present'] = False

    def _analyze_behavior_in_frame(self):
        action = self.behavior_analyzer.analyze_actions(self.frame_buffer)
        for roll_no, data in self.student_tracker.items():
            if data['present']:
                data['behaviors'].add(action)
        self.frame_buffer.clear()

    def _visualize_data(self, frame):
        for roll_no, data in self.student_tracker.items():
            if data['present']:
                text = f"{data['metadata']['name']} ({roll_no}) - Present"
                cv2.putText(frame, text, (10, 30 + 20 * list(self.student_tracker.keys()).index(roll_no)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    def generate_report(self):
        report_data = []
        current_time = datetime.now()

        print("INFO: Generating final report...")

        for roll_no, data in self.student_tracker.items():
            entry_time = data['entry_time']
            duration = current_time - entry_time
            duration_str = str(timedelta(seconds=round(duration.total_seconds())))

            report_data.append({
                'roll_no': roll_no,
                'name': data['metadata']['name'],
                'class_elective': data['metadata']['elective'],
                'date': entry_time.strftime('%Y-%m-%d'),
                'entry_time': entry_time.strftime('%H:%M:%S'),
                'time_in_classroom': duration_str,
                'behavior': ', '.join(sorted(list(data['behaviors']))) if data['behaviors'] else 'NA'
            })

        if not report_data:
            print("WARNING: No student data was tracked. Report will be empty.")
            return

        report_df = pd.DataFrame(report_data)
        report_filename = f"attendance_report_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        report_df.to_csv(report_filename, index=False)
        print(f"SUCCESS: Report generated successfully: {report_filename}")


if __name__ == "__main__":
    source = 0
    print(f"INFO: Using video source: {source} (Default Webcam)")
    controller = StudyGuardController(video_source=source)
    controller.run_monitoring()
