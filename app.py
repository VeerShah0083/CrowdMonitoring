from flask import Flask, render_template, jsonify, Response, request
from people_detector import PeopleDetector
import cv2
import threading
import numpy as np
from datetime import datetime
import json
from collections import deque
import time
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['LOCAL_VIDEO_PATH'] = 'D:/Desktop/IPD'  # Local video directory

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# CCTV Configuration
CCTV_SOURCES = {
    'camera1': {
        'name': 'Platform 1 CCTV',
        'url': 'rtsp://admin:password@192.168.1.100:554/stream1',
        'enabled': False
    },
    'camera2': {
        'name': 'Platform 2 CCTV',
        'url': 'rtsp://admin:password@192.168.1.101:554/stream1',
        'enabled': False
    }
}

# Allowed video file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class MonitoringSystem:
    def __init__(self):
        self.detector = None
        self.current_count = 0
        self.crowd_history = deque(maxlen=100)
        self.zone_counts = {
            'platform_1': 0,
            'platform_2': 0
        }
        self.analytics = {
            'hourly_trends': {},  # Store hourly averages
            'peak_times': [],     # Store peak crowd times with counts
            'platform_usage': {   # Track platform utilization
                'platform_1': {'total': 0, 'average': 0, 'peak': 0, 'counts': deque(maxlen=50)},
                'platform_2': {'total': 0, 'average': 0, 'peak': 0, 'counts': deque(maxlen=50)}
            },
            'alerts_history': []  # Track alert frequency
        }
        self.alerts = []
        self.is_running = False
        self.last_frame = None
        self.frame_lock = threading.Lock()
        self.video_source = None
        self.source_type = None
        self.crowd_threshold = 24  # Threshold for crowd alerts
        
        # Updated zone boundaries (x1, y1, x2, y2) to cover more of the frame
        self.zones = {
            'platform_1': [(0, 200, 960, 1080)],    # Left half of the frame
            'platform_2': [(960, 200, 1920, 1080)]  # Right half of the frame
        }
        
        self.zone_thresholds = {
            'platform_1': 24,
            'platform_2': 24
        }

    def initialize_detector(self, source_type, source_path):
        if self.detector is not None:
            self.cleanup()
        
        self.source_type = source_type
        self.video_source = source_path
        self.detector = PeopleDetector(model_path="yolov8x.pt", video_source=source_path)
        
        # Set video properties based on source type
        if source_type == 'cctv':
            # RTSP buffer size optimization for CCTV
            self.detector.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        # Set resolution for both types
        self.detector.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.detector.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def cleanup(self):
        if self.detector is not None:
            if self.detector.cap.isOpened():
                self.detector.cap.release()
            self.detector = None
        self.is_running = False
        self.last_frame = None
        
        # Reset analytics
        for platform in ['platform_1', 'platform_2']:
            stats = self.analytics['platform_usage'][platform]
            stats['total'] = 0
            stats['average'] = 0
            stats['peak'] = 0
            stats['counts'].clear()
        
        self.crowd_history.clear()
        self.analytics['peak_times'].clear()
        self.analytics['hourly_trends'].clear()
        self.alerts.clear()
        
        # Clean up only uploaded files from the uploads directory
        if self.source_type == 'file' and self.video_source:
            try:
                # Only delete files from the uploads directory
                if os.path.dirname(self.video_source) == app.config['UPLOAD_FOLDER'] and os.path.exists(self.video_source):
                    os.remove(self.video_source)
            except Exception as e:
                print(f"Error cleaning up file: {e}")

    def update_analytics(self):
        """Update analytics data"""
        current_time = datetime.now()
        hour_key = current_time.strftime('%H:00')
        
        # Update hourly trends
        if hour_key not in self.analytics['hourly_trends']:
            self.analytics['hourly_trends'][hour_key] = []
        self.analytics['hourly_trends'][hour_key].append(self.current_count)
        
        # Update platform usage
        for platform in ['platform_1', 'platform_2']:
            count = self.zone_counts[platform]
            stats = self.analytics['platform_usage'][platform]
            
            # Update peak if current count is higher
            stats['peak'] = max(stats['peak'], count)
            
            # Add current count to the counts deque
            stats['counts'].append(count)
            
            # Calculate average from the counts deque
            stats['average'] = sum(stats['counts']) / len(stats['counts'])
        
        # Check for peak times (when crowd > 75% of threshold)
        if self.current_count > 18:  # 75% of 24
            peak_entry = {
                'time': current_time.strftime('%H:%M:%S'),
                'count': self.current_count,
                'platform1': self.zone_counts['platform_1'],
                'platform2': self.zone_counts['platform_2']
            }
            self.analytics['peak_times'].append(peak_entry)
            if len(self.analytics['peak_times']) > 10:  # Keep last 10 peak times
                self.analytics['peak_times'].pop(0)
        
        # Update crowd history
        self.crowd_history.append({
            'timestamp': current_time.strftime('%H:%M:%S'),
            'total': self.current_count,
            'zones': self.zone_counts.copy()
        })

monitor = MonitoringSystem()

def generate_frames():
    while True:
        if not monitor.is_running:
            break
            
        with monitor.frame_lock:
            frame = monitor.last_frame
            if frame is not None:
                try:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    print(f"Error in generate_frames: {e}")
                    break
        time.sleep(0.03)  # Reduce frame rate to prevent memory buildup

def process_frame(frame, detections):
    """Process frame and update monitoring data"""
    # Clear previous counts
    for zone in monitor.zone_counts:
        monitor.zone_counts[zone] = 0
    
    # Process each detection and update zone counts
    for box, conf in detections:
        x1, y1, x2, y2 = map(int, box)
        
        # Calculate center point of the detection
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Check each zone
        for zone_name, zone_areas in monitor.zones.items():
            for zone_box in zone_areas:
                # Check if the center point is inside the zone
                if (zone_box[0] <= center_x <= zone_box[2] and 
                    zone_box[1] <= center_y <= zone_box[3]):
                    monitor.zone_counts[zone_name] += 1
                    break  # Once counted in a zone, move to next detection
    
    # Update current total count
    monitor.current_count = len(detections)
    
    # Check for alerts
    check_alerts()
    
    # Update crowd history with current counts
    monitor.crowd_history.append({
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'total': monitor.current_count,
        'zones': monitor.zone_counts.copy()
    })
    
    # Update analytics immediately
    monitor.update_analytics()

def check_overlap(box1, box2):
    """Check if two boxes overlap"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return x1 < x2 and y1 < y2

def check_alerts():
    """Check for alert conditions"""
    total_count = monitor.current_count
    
    # Check total crowd alert
    if total_count > 35:  # Critical overcrowding
        alert = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'message': f'CRITICAL: Total crowd ({total_count} people) exceeds safe limit!',
            'severity': 'high',
            'type': 'total'
        }
        # Only add if it's different from the last alert
        if not monitor.alerts or monitor.alerts[-1] != alert:
            monitor.alerts.append(alert)
            monitor.analytics['alerts_history'].append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'type': 'critical',
                'count': total_count
            })
    elif total_count > 25:  # Warning level
        alert = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'message': f'WARNING: High crowd level ({total_count} people)',
            'severity': 'medium',
            'type': 'total'
        }
        # Only add if it's different from the last alert
        if not monitor.alerts or monitor.alerts[-1] != alert:
            monitor.alerts.append(alert)
            monitor.analytics['alerts_history'].append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'type': 'warning',
                'count': total_count
            })
    
    # Check platform-specific alerts
    for platform, count in monitor.zone_counts.items():
        if count > 20:  # Platform overcrowding threshold
            severity = 'high' if count > 25 else 'medium'
            message = f'Overcrowding on {platform.replace("_", " ").title()}: {count} people'
            
            alert = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'zone': platform,
                'count': count,
                'message': message,
                'severity': severity,
                'type': 'zone'
            }
            # Only add if it's different from the last alert for this platform
            if not any(a.get('zone') == platform and a['message'] == message for a in monitor.alerts[-5:]):
                monitor.alerts.append(alert)
    
    # Keep only recent alerts (last 50)
    monitor.alerts = monitor.alerts[-50:]

def monitoring_thread():
    """Background thread for continuous monitoring"""
    try:
        while monitor.is_running:
            if monitor.detector is None or not monitor.detector.cap.isOpened():
                print("Detector not initialized or video capture closed")
                break

            ret, frame = monitor.detector.cap.read()
            if not ret:
                monitor.detector.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            # Use the same detection parameters as the main detector
            results = monitor.detector.model(frame, verbose=False)
            detections = []
            
            for box in results[0].boxes:
                if box.cls == 0:  # person class
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf.cpu().numpy().item()
                    # Use the same detection logic as in PeopleDetector
                    box_height = xyxy[3] - xyxy[1]
                    if box_height < 100:  # Far person
                        if conf > 0.1:
                            detections.append((xyxy, conf))
                    else:  # Close person
                        if conf > 0.15:
                            detections.append((xyxy, conf))
            
            # Process detections and update counts
            process_frame(frame, detections)
            
            # Draw frame with detections
            frame_copy = frame.copy()
            
            # Draw person detections
            for box, _ in detections:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Add total count to frame
            cv2.putText(frame_copy, f'Total Count: {monitor.current_count}',
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            with monitor.frame_lock:
                monitor.last_frame = frame_copy
            
            # Reduce sleep time for more frequent updates
            time.sleep(0.01)  # ~100 FPS max
            
    except Exception as e:
        print(f"Error in monitoring thread: {e}")
    finally:
        monitor.cleanup()

@app.route('/')
def index():
    # Get list of local video files
    local_videos = []
    if os.path.exists(app.config['LOCAL_VIDEO_PATH']):
        for file in os.listdir(app.config['LOCAL_VIDEO_PATH']):
            if allowed_file(file):
                local_videos.append(file)
    
    return render_template('index.html', 
                         cctv_sources=CCTV_SOURCES,
                         local_videos=local_videos)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    if not monitor.is_running:
        try:
            source_type = request.form.get('source_type')
            source_path = None
            
            if source_type == 'cctv':
                camera_id = request.form.get('camera_id')
                if camera_id not in CCTV_SOURCES:
                    return jsonify({'status': 'error', 'message': 'Invalid camera selected'})
                source_path = CCTV_SOURCES[camera_id]['url']
            
            elif source_type == 'file':
                if 'video_file' in request.files:
                    file = request.files['video_file']
                    if file.filename == '':
                        return jsonify({'status': 'error', 'message': 'No file selected'})
                    
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(filepath)
                        source_path = filepath
                    else:
                        return jsonify({'status': 'error', 'message': 'Invalid file type'})
                else:
                    # Check for local file selection
                    local_file = request.form.get('local_file')
                    if local_file:
                        filepath = os.path.join(app.config['LOCAL_VIDEO_PATH'], local_file)
                        if os.path.exists(filepath) and allowed_file(local_file):
                            source_path = filepath
                        else:
                            return jsonify({'status': 'error', 'message': 'Selected local file not found'})
                    else:
                        return jsonify({'status': 'error', 'message': 'No video source provided'})
            
            else:
                return jsonify({'status': 'error', 'message': 'Invalid source type'})
            
            monitor.initialize_detector(source_type, source_path)
            monitor.is_running = True
            threading.Thread(target=monitoring_thread, daemon=True).start()
            return jsonify({'status': 'success'})
            
        except Exception as e:
            monitor.cleanup()
            return jsonify({'status': 'error', 'message': str(e)})
    return jsonify({'status': 'success'})

@app.route('/update_cctv_config', methods=['POST'])
def update_cctv_config():
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        url = data.get('url')
        enabled = data.get('enabled', False)
        
        if camera_id in CCTV_SOURCES:
            CCTV_SOURCES[camera_id]['url'] = url
            CCTV_SOURCES[camera_id]['enabled'] = enabled
            return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'Invalid camera ID'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_monitoring')
def stop_monitoring():
    monitor.cleanup()
    return jsonify({'status': 'success'})

@app.route('/get_stats')
def get_stats():
    current_hour = datetime.now().strftime('%H:00')
    hourly_avg = 0
    if current_hour in monitor.analytics['hourly_trends']:
        hourly_data = monitor.analytics['hourly_trends'][current_hour]
        if hourly_data:
            hourly_avg = sum(hourly_data) / len(hourly_data)
    
    # Ensure we're sending the most current data
    return jsonify({
        'current_count': monitor.current_count,
        'zone_counts': {
            'platform_1': monitor.zone_counts['platform_1'],
            'platform_2': monitor.zone_counts['platform_2']
        },
        'alerts': monitor.alerts[-5:],
        'crowd_history': list(monitor.crowd_history),
        'analytics': {
            'platform_usage': {
                'platform_1': {
                    'total': monitor.analytics['platform_usage']['platform_1']['total'],
                    'average': round(monitor.analytics['platform_usage']['platform_1']['average'], 1),
                    'peak': monitor.analytics['platform_usage']['platform_1']['peak']
                },
                'platform_2': {
                    'total': monitor.analytics['platform_usage']['platform_2']['total'],
                    'average': round(monitor.analytics['platform_usage']['platform_2']['average'], 1),
                    'peak': monitor.analytics['platform_usage']['platform_2']['peak']
                }
            },
            'peak_times': monitor.analytics['peak_times'][-10:],
            'hourly_average': round(hourly_avg, 1),
            'alerts_count': len(monitor.analytics['alerts_history'])
        }
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True) 