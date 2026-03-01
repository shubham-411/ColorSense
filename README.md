# ColourSense

**ColourSense** is a real-time computer vision project that detects objects from a webcam feed, identifies their dominant color, and announces what it sees using voice feedback.

This project combines object detection, color recognition, and text-to-speech into a simple but powerful assistive vision system.

## What This Project Does

* Captures live video from your webcam
* Detects objects using a trained YOLO model
* Extracts the dominant color from the center of each detected object
* Displays object name + color on screen
* Speaks out what it detects (e.g., *"I see a red bottle"*)


## Technologies Used

This project directly uses:

* **OpenCV (`cv2`)** – Webcam handling, image processing, drawing bounding boxes
* **NumPy** – Numerical operations
* **Pandas** – Reading and processing the `colors.csv` dataset
* **pyttsx3** – Offline text-to-speech engine
* **threading** – Non-blocking voice execution
* **Ultralytics YOLO** – Object detection model loading and inference


## Installation

Make sure you have Python 3.8+ installed.

Install required packages:

```bash
pip install opencv-python numpy pandas pyttsx3 ultralytics
```

---

## How To Run-

1. Ensure:

   * `colors.csv` is in the same directory as your script
   * Your trained YOLO model exists at:

     ```
     runs/detect/train/weights/best.pt
     ```

2. Run the script:

```bash
python main.py
```

3. Press **Q** to quit.


##  How It Works

### Object Detection

The script loads a trained YOLO model:

```python
model = YOLO("runs/detect/train/weights/best.pt")
```

Each webcam frame is passed into the model to detect objects.



### Color Detection

For every detected object:

* The center region (20×20 crop) is extracted
* The average RGB value is calculated
* The closest matching color from `colors.csv` is selected

Color matching is done using absolute RGB difference.



### Voice Feedback

The detected result is spoken using `pyttsx3` in a background thread to prevent blocking the video stream.

Example:

```
I see a blue phone
```

## Purpose of This Project

* Learning real-time computer vision
* Integrating detection + color analysis
* Building an assistive AI demo
* Strengthening practical AI project skills

