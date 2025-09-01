
from datetime import datetime

class StudentTracker:
    """
    Tracks the state and data for all students.
    """
    def __init__(self):
        self._students = {}

    def update_student(self, roll_no, metadata):
        """
        Updates a student's state or logs a new entry.
        """
        now = datetime.now()
        if roll_no not in self._students:
            # New student entry
            self._students[roll_no] = {
                'metadata': metadata,
                'entry_time': now,
                'last_seen': now,
                'present': True,
                'behaviors': set()
            }
        else:
            # Update last seen time for existing student
            self._students[roll_no]['last_seen'] = now
            self._students[roll_no]['present'] = True

    def mark_absent(self, roll_nos_in_frame):
        """
        Marks students not in the current frame as 'absent'.
        """
        all_tracked_roll_nos = set(self._students.keys())
        absent_roll_nos = all_tracked_roll_nos - roll_nos_in_frame
        for roll_no in absent_roll_nos:
            self._students[roll_no]['present'] = False

    def record_behavior_for_present_students(self, action):
        """
        Adds a behavior to all students currently marked as present.
        """
        for roll_no, data in self._students.items():
            if data['present']:
                data['behaviors'].add(action)

    def get_all_students_data(self):
        """
        Returns the dictionary containing all tracked student data.
        """
        return self._students
