
import pandas as pd
from datetime import datetime, timedelta

def generate_csv_report(student_data):
    """
    Generates the final CSV attendance and behavior report.
    
    Args:
        student_data (dict): The dictionary of tracked student data.
    """
    report_data = []
    current_time = datetime.now()
    
    print("INFO: Generating final report...")

    for roll_no, data in student_data.items():
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
            'behavior': ', '.join(sorted(list(data['behaviors']))) if data['behaviors'] else 'N/A'
        })
    
    if not report_data:
        print("WARNING: No student data was tracked. Report will be empty.")
        return

    report_df = pd.DataFrame(report_data)
    report_filename = f"attendance_report_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    try:
        report_df.to_csv(report_filename, index=False)
        print(f"SUCCESS: Report generated successfully: {report_filename}")
    except Exception as e:
        print(f"ERROR: Could not save report file: {e}")
