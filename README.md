# Clone Tracker - Multi-Person Real-time Pose Mirroring

A visually stunning real-time pose tracking application that creates a "clone" or mirror effect of body movements using computer vision. Now upgraded with **YOLOv8 Pose** for robust multi-person tracking and responsive functionality.

**Author**: [Vaibhav S A Gupta](https://github.com/Vaibhavsagupta)

## Features

- **Multi-Person Tracking**: Detects and creates effects for all people in the frame using `ultralytics` YOLOv8.
- **Responsive Design**: 
  - Works on any resolution.
  - **Fullscreen support** via `--fullscreen` arg or 'f' hotkey.
  - Resizable window with automatic content scaling.
- **Dynamic Visual Effects**: 
  - Color-changing neon skeletal overlay.
  - Pulsing glow effects.
  - Motion trails that follow joint movements.
  - Gradient color transitions over time.
- **Performance Optimized**: Includes FPS monitoring and efficient rendering.

## Requirements

```bash
pip install ultralytics opencv-contrib-python numpy lap
```

### Dependencies

- `ultralytics` - YOLOv8 for state-of-the-art multi-person pose estimation.
- `opencv-contrib-python` - Robust camera capture and image processing.
- `numpy` - Numerical computations.
- `lap` - Linear Assignment Problem solver for object tracking.

## How It Works

### 1. Pose Detection
The application uses the **YOLOv8n-pose** model to detect 17 key body landmarks for multiple people in real-time. Unlike previous versions, this supports tracking multiple users simultaneously with persistent IDs.

### 2. Clone Creation
The detected pose is mirrored to create a "clone" effect relative to the screen center. The specific keypoints (shoulders, elbows, knees, etc.) are connected to form a neon skeleton.

### 3. Visual Effects Engine

#### Color Gradient System
- Implements HSV to RGB color space conversion.
- Creates smooth color transitions over time.
- Applies pulsing effects using sine wave functions.

#### Neon Glow Effects
- Dual-layer line drawing for glow appearance.
- Smooth circles with white centers for joints.
- Additive blending for realistic glow.

#### Motion Trails
- Maintains history of joint positions per tracked person ID.
- Applies alpha blending for fade-out effect.
- Creates smooth motion visualization for everyone in the scene.

## Usage

1. **Run the Application**
   ```bash
   python app.py
   ```
   
   **Fullscreen Mode:**
   ```bash
   python app.py --fullscreen
   ```

2. **Controls**
   - **'f'**: Toggle fullscreen mode.
   - **'q'**: Quit the application.
   - Resize the window manually to fit your needs.

3. **Expected Output**
   - Main window: "CLONE TRACKER" showing camera feed with effects.
   - Console: FPS metrics and status.

## Technical Details

- **Model**: YOLOv8n-pose (Nano model) for speed.
- **Tracking**: BoT-SORT / ByteTrack (via Ultralytics `track` mode).
- **Coordinate System**: COCO Keypoint format (17 points).

## License

This project is for educational and personal use. YOLOv8 is subject to AGPL-3.0 license.

## Contributing

Feel free to submit issues and enhancement requests!
