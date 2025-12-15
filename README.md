# Vegetable Detector

A real-time vegetable detection system using YOLOv8 and OpenCV. This project captures video from your webcam, detects vegetables, draws bounding boxes around them, and displays a live count of detected items.

## Features
- **Real-time Detection**: Uses YOLOv8 for fast and accurate object detection.
- **Live Counting**: Displays a dynamic count of each detected vegetable type on the screen.
- **Visual Feedback**: Draws bounding boxes and confidence scores around detected vegetables.

## Prerequisites
- Python 3.8 or higher
- Webcam

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ChaitanyaPesitm/Vegetable-Detector.git
   cd Vegetable-Detector
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure your webcam is connected.
2. Run the detection script:
   ```bash
   python vegetable/vegdetector.py
   ```
3. The application window will open showing the live feed.
   - Detected vegetables will have bounding boxes.
   - A count summary is displayed in the top-left corner.
4. Press **'q'** to quit the application.

## Project Structure
- `vegetable/vegdetector.py`: Main script for detection and display.
- `vegetable/yolov8n.pt`: Pre-trained YOLOv8 model (ensure this file exists or is downloaded automatically).
- `requirements.txt`: List of Python dependencies.

## Technologies Used
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
