import cv2
import numpy as np
from ultralytics import YOLO
import logging
import time
import csv
from pathlib import Path
import torch

class PeopleDetector:
    def __init__(self, model_path: str, video_source: str, confidence_threshold=0.2):
        self.count = 0
        self.current_detections = []
        self.video_source = video_source
        self.confidence_threshold = confidence_threshold
        self.cap = cv2.VideoCapture(video_source)
        
        # Set video resolution to full HD
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.model = self.initialize_model(model_path)
        self.log_file = Path('detections.csv')
        self.setup_logging()
        self.initialize_log_file()
        
        # Pre-calculate frame regions
        _, first_frame = self.cap.read()
        if first_frame is not None:
            self.frame_height = first_frame.shape[0]
            self.frame_width = first_frame.shape[1]
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def initialize_model(self, model_path: str):
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path}: {e}")

    def enhance_frame(self, frame):
        """Enhance frame for better detection of distant features"""
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_enhanced = self.clahe.apply(l)
        
        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]]) / 9.0
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced

    def get_head_region(self, person_box, is_far=False):
        """Extract head region from full person detection with adjustments for distance"""
        x1, y1, x2, y2 = person_box
        height = y2 - y1
        width = x2 - x1
        
        if is_far:
            # For far people, use a larger portion of the body as head might be less distinct
            head_height = height / 4  # Use top 1/4 for far people
            head_width = width * 0.9  # Wider box for far people to account for uncertainty
        else:
            # For closer people, use more precise head region
            head_height = height / 6  # Use top 1/6 for close people
            head_width = width * 0.7  # Narrower box for close people
        
        head_x1 = x1 + (width - head_width) / 2
        head_x2 = head_x1 + head_width
        head_y2 = y1 + head_height
        
        return np.array([head_x1, y1, head_x2, head_y2])

    def process_frame(self, frame):
        """Process a single frame and return detections"""
        # Process frame with model using original parameters
        results = self.model(frame, verbose=False, conf=0.1)  # Lower threshold to detect far people
        
        detections = []
        for box in results[0].boxes:
            if box.cls == 0:  # person class
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy().item()
                # Accept lower confidence for smaller (potentially far) detections
                box_height = xyxy[3] - xyxy[1]
                if box_height < 100:  # Far person
                    if conf > 0.1:
                        detections.append((xyxy, conf))
                else:  # Close person
                    if conf > 0.15:
                        detections.append((xyxy, conf))
        
        self.current_detections = detections
        self.count = len(detections)  # Update count
        return detections

    def process_video(self, output_path="output.avi"):
        frame_number = 0
        frame_skip = 1  # Process every frame
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_writer = None

        # Create resizable window
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if frame_number % frame_skip != 0:
                frame_number += 1
                continue
            if not ret:
                break

            # Process frame for people detection
            detections = self.process_frame(frame)
            
            # Draw detections
            frame_copy = frame.copy()
            for box, _ in detections:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add total count to frame
            cv2.putText(frame_copy, f'Total Count: {self.count}',
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if output_writer is None:
                output_writer = cv2.VideoWriter(output_path, fourcc, 20.0,
                                             (frame_copy.shape[1], frame_copy.shape[0]))
            output_writer.write(frame_copy)
            
            cv2.imshow('output', frame_copy)
            self.log_detections(frame_number, self.count)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_number += 1

        self.cap.release()
        if output_writer:
            output_writer.release()
        cv2.destroyAllWindows()

    def calculate_iou(self, box1, box2):
        # Calculate intersection over union between two boxes
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1]))
        box2_area = (float(box2[2]) - float(box2[0])) * (float(box2[3]) - float(box2[1]))
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def initialize_log_file(self):
        with self.log_file.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Frame', 'Count'])

    def log_detections(self, frame_number, count):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.log_file.open('a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, frame_number, count])
        logging.info(f"Frame {frame_number}: Detected {count} people")

    def get_current_detections(self):
        """Get the current detections and count"""
        return self.current_detections, self.count

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
