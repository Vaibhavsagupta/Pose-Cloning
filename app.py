import sys
import traceback
import time
import math
import numpy as np
import cv2
import argparse

print("Starting app...", flush=True)

try:
    # Parse arguments
    parser = argparse.ArgumentParser(description='Clone Tracker App')
    parser.add_argument('--fullscreen', action='store_true', help='Start in fullscreen mode')
    # Use parse_known_args to avoid crashing if unknown args are passed (though usually strict is fine)
    args, unknown = parser.parse_known_args()

    from ultralytics import YOLO
    
    # Load YOLOv8 Pose model
    print("Loading YOLOv8 model...", flush=True)
    model = YOLO('yolov8n-pose.pt') 
    print("Model loaded.", flush=True)

    cap = cv2.VideoCapture(0)
    # Use a reasonable default resolution for processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Setup Window
    window_name = "CLONE TRACKER"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # COCO Keypoint Connections
    # 5: Shoulders, 7/9: L Arm, 6/8/10: R Arm, 11/12: Hips, 13/15: L Leg, 14/16: R Leg
    POSE_CONNECTIONS = [
        (5, 6),   # Shoulders
        (5, 7), (7, 9), # Left Arm
        (6, 8), (8, 10), # Right Arm
        (11, 12), # Hips
        (5, 11),  # Left Body
        (6, 12),  # Right Body
        (11, 13), (13, 15), # Left Leg
        (12, 14), (14, 16)  # Right Leg
    ]

    start_time = time.time()
    frame_count = 0
    
    # Track trails per person ID
    # key: track_id, value: list of frames (each frame is list of points)
    person_trails = {} 
    max_trail_length = 8

    def get_gradient_color(t):
        hue = (t * 120) % 360
        c = 1
        x = c * (1 - abs((hue / 60) % 2 - 1))
        
        if 0 <= hue < 60:
            r, g, b = c, x, 0
        elif 60 <= hue < 120:
            r, g, b = x, c, 0
        elif 120 <= hue < 180:
            r, g, b = 0, c, x
        elif 180 <= hue < 240:
            r, g, b = 0, x, c
        elif 240 <= hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (int(r * 255), int(g * 255), int(b * 255))

    def draw_smooth_line(img, pt1, pt2, color, thickness):
        cv2.line(img, pt1, pt2, color, thickness)
        cv2.line(img, pt1, pt2, tuple(c//2 for c in color), thickness+2)

    def draw_smooth_circle(img, center, radius, color):
        cv2.circle(img, center, radius, color, -1)
        cv2.circle(img, center, radius//2, (255, 255, 255), -1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame", flush=True)
            break
        
        # Flip frame horizontally for mirror effect (standard webcam view)
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        # Run YOLO inference
        # persist=True handles tracking (assigns IDs)
        results = model.track(frame, persist=True, verbose=False)
        
        current_time = time.time()
        wave_time = current_time - start_time
        
        gradient_color = get_gradient_color(wave_time)
        pulse_intensity = 0.8 + 0.2 * math.sin(wave_time * 6)
        neon_color = tuple(int(c * pulse_intensity) for c in gradient_color)
        
        glow_layer = np.zeros_like(frame, dtype=np.uint8)
        
        has_detections = False
        
        if results and len(results) > 0 and results[0].keypoints is not None:
            # Iterate through each detected person
            # keypoints is a tensor of shape (N, 17, 3) -> (x, y, conf) or (N, 17, 2)
            # .xy gives pixels
            
            keypoints_all = results[0].keypoints.xy.cpu().numpy() # list of (17, 2)
            ids = results[0].boxes.id
            if ids is not None:
                track_ids = ids.cpu().numpy().astype(int)
            else:
                track_ids = range(len(keypoints_all))

            if len(keypoints_all) > 0:
                has_detections = True
                
                for idx, kpts in enumerate(keypoints_all):
                    track_id = track_ids[idx]
                    
                    person_points = []
                    clone_points = []
                    
                    # Extract valid points
                    valid_kpts = {}
                    for i, (px, py) in enumerate(kpts):
                        if px == 0 and py == 0: continue # Invalid
                        valid_kpts[i] = (int(px), int(py))
                        
                        # Clone/Mirror coordinate
                        cx = int(width - px)
                        cy = int(py)
                        clone_points.append((cx, cy))
                        
                        # Draw joints
                        radius = 6
                        # Draw on Clone
                        draw_smooth_circle(glow_layer, (cx, cy), radius, neon_color)
                    
                    # Draw connections
                    for i, j in POSE_CONNECTIONS:
                        if i in valid_kpts and j in valid_kpts:
                            px1, py1 = valid_kpts[i]
                            px2, py2 = valid_kpts[j]
                            
                            # Mirror them
                            cx1, cy1 = width - px1, py1
                            cx2, cy2 = width - px2, py2
                            
                            thickness = 4
                            draw_smooth_line(glow_layer, (cx1, cy1), (cx2, cy2), neon_color, thickness)
                    
                    # Update Trails
                    if track_id not in person_trails:
                        person_trails[track_id] = []
                    
                    person_trails[track_id].append(clone_points)
                    if len(person_trails[track_id]) > max_trail_length:
                        person_trails[track_id].pop(0)
                        
                    # Draw trails
                    for t_idx, trail_frame in enumerate(person_trails[track_id][:-1]):
                        trail_alpha = (t_idx / len(person_trails[track_id])) * 0.4
                        trail_color = tuple(int(c * trail_alpha) for c in neon_color)
                        for tx, ty in trail_frame:
                            cv2.circle(glow_layer, (tx, ty), 3, trail_color, -1)
                
                # Cleanup old trails
                active_ids = set(track_ids)
                to_remove = [tid for tid in person_trails if tid not in active_ids]
                for tid in to_remove:
                    del person_trails[tid]

        # Blend layers
        frame = cv2.addWeighted(frame, 0.7, glow_layer, 0.6, 0)
        
        if not has_detections:
            # Simplified scanning message
            search_text = "SCANNING..."
            font_scale = 1.0
            thickness = 2
            text_size = cv2.getTextSize(search_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            
            scan_color = get_gradient_color(wave_time * 2)
            cv2.putText(frame, search_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, scan_color, thickness)
        
        # Simple FPS counter
        if frame_count % 30 == 0:
            fps = int(1.0 / (time.time() - current_time + 0.001))
            fps_text = f"FPS: {fps}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(window_name, frame)
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            # Toggle fullscreen
            prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            if prop == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap.release()
    cv2.destroyAllWindows()

except BaseException as e:
    with open('error_log.txt', 'w') as f:
        f.write(str(e))
        f.write('\n')
        traceback.print_exc(file=f)
    print(f"An error occurred: {e}", flush=True)
 