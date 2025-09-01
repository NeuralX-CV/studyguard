
import argparse
from controller import StudyGuardController

def main():
    """
    Parses arguments, initializes the controller, and runs the monitoring system.
    """
    parser = argparse.ArgumentParser(description="StudyGuard AI Classroom Monitoring System")
    parser.add_argument(
        "--video_source",
        type=str,
        default='0',
        help="Path to a video file or camera index (e.g., '0' for the default webcam)."
    )
    args = parser.parse_args()

    # Convert video source to integer if it's a number (for webcam index)
    try:
        source = int(args.video_source)
    except ValueError:
        source = args.video_source

    print(f"INFO: Using video source: {source}")
    controller = StudyGuardController(video_source=source)
    controller.run_monitoring()

if __name__ == "__main__":
    main()
