import os
import argparse
from people_detector import PeopleDetector
from app import app, monitor  # Import the Flask app and monitor instance
import threading
import time
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pedestrian Detection using YOLO")
    parser.add_argument('--video', type=str, required=True, help="Path to the video file.")
    parser.add_argument('--model', type=str, default='yolov8x.pt', help="Path to the YOLO model.")
    parser.add_argument('--output', type=str, default='output.avi', help="Path to save the processed video.")
    parser.add_argument('--web', action='store_true', help="Run with web interface")
    return parser.parse_args()

def run_detector(detector, monitor):
    """Run the detector in a separate thread and update the monitor"""
    while detector.cap.isOpened() and monitor.is_running:
        ret, frame = detector.cap.read()
        if not ret:
            detector.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # Use detector's original process_frame method
        detections = detector.process_frame(frame)
        
        # Update monitor with detection results
        monitor.current_count = detector.count
        
        # Update zone counts based on detections
        monitor.zone_counts = {'platform_1': 0, 'platform_2': 0}
        for box, _ in detections:
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            
            # Check which platform the detection belongs to
            if center_x < detector.frame_width // 2:
                monitor.zone_counts['platform_1'] += 1
            else:
                monitor.zone_counts['platform_2'] += 1
        
        # Draw frame with detections
        frame_copy = frame.copy()
        
        # Draw detections
        for box, _ in detections:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add total count to frame
        cv2.putText(frame_copy, f'Total Count: {detector.count}', (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Store the frame for web display
        with monitor.frame_lock:
            monitor.last_frame = frame_copy
        
        # Update analytics
        monitor.update_analytics()
        
        # Show detection window
        cv2.imshow('Detector', frame_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.01)  # Small delay to prevent excessive CPU usage
    
    cv2.destroyAllWindows()

def main():
    args = parse_arguments()
    if not os.path.exists(args.video):
        raise ValueError(f"Error: The video source '{args.video}' does not exist.")

    if args.web:
        # Create detector instance
        detector = PeopleDetector(model_path=args.model, video_source=args.video)
        monitor.detector = detector
        monitor.is_running = True
        
        # Start detector thread
        detector_thread = threading.Thread(target=run_detector, args=(detector, monitor))
        detector_thread.daemon = True
        detector_thread.start()
        
        # Run Flask app
        app.run(debug=False, threaded=True, use_reloader=False)
    else:
        # Run standalone without web interface
        detector = PeopleDetector(model_path=args.model, video_source=args.video)
        detector.process_video(output_path=args.output)

if __name__ == "__main__":
    main()
