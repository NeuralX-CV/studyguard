StudyGuard – AI Classroom Monitoring System
A real-time classroom monitoring system using computer vision and deep learning to automate attendance logging and analyze student behavior. StudyGuard provides educators with valuable insights into classroom dynamics, ensuring an engaging and efficient learning environment.

Features
Real-time Face Recognition: Accurately identifies multiple students (tested for 60-70+) in a live video stream for automated entry/exit tracking.

Automated Attendance Logging: Generates a detailed CSV report with student roll number, name, elective, date, entry time, and total time spent in the classroom.

Student Behavior Analysis: A modular, simulated 3D-CNN analyzes student actions like hand-raising and standing. The design allows for a real deep learning model to be easily integrated.

High Performance: Optimized to process frames at intervals, ensuring smooth operation even on less powerful hardware like a Raspberry Pi 5.

Modular and Extendable: The codebase is broken down into logical components (controller, face_manager, utils, etc.), making it easy to maintain, test, and extend.

How It Works
The system follows a continuous pipeline for each frame of the video stream:

Video Capture: The Controller captures frames from the specified video source (webcam or file).

Face Detection: For performance, every Nth frame is processed. The system detects all faces present in the frame.

Face Recognition: Each detected face is compared against a pre-encoded database of known student faces managed by the FaceManager.

Student Tracking: The StudentTracker updates the status of each recognized student, logging their entry time and marking them as "present."

Behavior Analysis: Periodically, a sequence of frames is passed to the BehaviorAnalyzer, which simulates a 3D-CNN model to classify the behavior of present students.

Reporting: When the process is terminated, the utils module generates a comprehensive .csv report with the session's attendance and behavior data.

Project Structure
The project is organized into a clean, modular structure:

studyguard/
├── main.py                 # Main entry point to run the application
├── controller.py           # Core controller that orchestrates the system
├── face_manager.py         # Handles loading student data and face encodings
├── student_tracker.py      # Manages the real-time state of each student
├── behavior_analyzer.py    # Simulates student behavior analysis
├── utils.py                # Contains helper functions (e.g., report generation)
|
├── student_images/         # Directory for student images (e.g., 101.jpg)
│   ├── 101.jpg
│   └── ...
|
├── students_db.csv         # CSV database with student information
└── requirements.txt        # Required Python packages

Getting Started
Follow these steps to get StudyGuard running on your local machine.

Prerequisites
Python 3.8 or higher

pip package installer

CMake must be installed on your system for dlib to build correctly.

A webcam or a video file of a classroom.

Installation
Clone the Repository:

git clone [https://github.com/your-username/StudyGuard.git](https://github.com/your-username/StudyGuard.git)
cd StudyGuard

Create and Activate a Virtual Environment (Recommended):

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate

Install Dependencies:

pip install -r requirements.txt

Usage
1. Prepare Student Data
Before running the system, you must provide the data for the students you want to track.

Student Database: Open students_db.csv and populate it with your student roster. The file requires the following columns: roll_no,name,elective.

Student Images: In the student_images/ directory, add at least one clear, front-facing photo for each student. Crucially, name each image file to match the student's roll_no (e.g., 101.jpg, 102.png).

2. Run the System
You can run the application from your terminal.

To use the default webcam:

python main.py

To use a different camera index or a video file:

python main.py --video_source /path/to/your/video.mp4

To stop the program, ensure the video window is active and press the 'q' key. The final attendance report will be saved in the root directory.

Configuration
You can fine-tune the system's performance and accuracy by adjusting the constants at the top of the controller.py file:

FRAME_PROCESS_INTERVAL: The number of frames to skip between processing. Increase for better performance, decrease for higher detection frequency.

BEHAVIOR_ANALYSIS_INTERVAL: The number of frames to buffer before running behavior analysis.

FACE_MATCH_TOLERANCE: The strictness for face matching. Lower values (0.5) are stricter, while higher values (0.7) are more lenient. The default is 0.6.

Contributing
Contributions are welcome! If you'd like to improve StudyGuard, please feel free to fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

Future Work
[ ] Implement Real 3D-CNN: Train and integrate a real 3D-CNN model for accurate behavior analysis.

[ ] Database Integration: Replace the CSV database with a more robust database system like SQLite or PostgreSQL.

[ ] Web-Based Dashboard: Create a web interface to display real-time statistics and view reports.

[ ] Alerting System: Implement an alerting mechanism for specific behaviors (e.g., a student being absent for too long).

License
This project is licensed under the MIT License - see the LICENSE.md file for details.
