
import cv2
from datetime import datetime
import face_recognition
import numpy as np
from face_manager import FaceManager
from behavior_analyzer import BehaviorAnalyzer
from student_tracker import StudentTracker
from utils import generate_csv_report

# --- Constants ---
FRAME_PROCESS_INTERVAL = 10  # Process every 10th frame for performance
BEHAVIOR_ANALYSIS_INTERVAL = 90  # Analyze behavior every 90 frames (approx 3 seconds)
FACE_MATCH_TOLERANCE = 0.6  # Lower is stricter

class StudyGuardController:
    """
    Orchestrates the entire monitoring process.
    """
    def __init__(self, video_source):
        print("INFO: Initializing StudyGuard Controller...")
        self.video_source = video_source
        self.face_manager = FaceManager()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.student_tracker = StudentTracker()
        self.frame_buffer = []

    def run_monitoring(self):
        """Starts the main monitoring loop."""
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

            if frame_count % BEHAVIOR_ANALYSIS_INTERVAL == 0 and self.frame_buffer:
                self._analyze_behavior_in_frame()

            display_frame = self._visualize_data(frame)
            cv2.imshow('StudyGuard - AI Classroom Monitoring', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("INFO: 'q' pressed. Shutting down...")
                break

            frame_count += 1

        video_capture.release()
        cv2.destroyAllWindows()
        generate_csv_report(self.student_tracker.get_all_students_data())

    def _process_frame(self, frame):
        """Processes a single frame for face detection and recognition."""
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        current_frame_roll_nos = set()

        for face_encoding in face_encodings:
            match_found, metadata = self.face_manager.find_match(face_encoding, FACE_MATCH_TOLERANCE)
            
            if match_found:
                roll_no = metadata['roll_no']
                current_frame_roll_nos.add(roll_no)
                self.student_tracker.update_student(roll_no, metadata)

        self.student_tracker.mark_absent(current_frame_roll_nos)

    def _analyze_behavior_in_frame(self):
        """Analyzes behavior for present students."""
        action = self.behavior_analyzer.analyze_actions(self.frame_buffer)
        self.student_tracker.record_behavior_for_present_students(action)
        self.frame_buffer.clear()

    def _visualize_data(self, frame):
        """Draws information on the display frame."""
        y_offset = 30
        for roll_no, data in self.student_tracker.get_all_students_data().items():
            if data.get('present', False):
                text = f"{data['metadata']['name']} ({roll_no}) - Present"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 20
        return frame
