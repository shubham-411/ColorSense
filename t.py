import cv2
import numpy as np
import pandas as pd
import pyttsx3
import threading
from ultralytics import YOLO

index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
df = pd.read_csv('colors.csv')
mouse_x, mouse_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

cv2.namedWindow("ColorSense")
cv2.setMouseCallback("ColorSense", mouse_callback)

def getColorName(R, G, B):
    minimum = 10000
    cname = "Undefined"
    for i in range(len(df)):
        d = abs(R - int(df.loc[i, "R"])) + abs(G - int(df.loc[i, "G"])) + abs(B - int(df.loc[i, "B"]))
        if d < minimum:
            minimum = d
            cname = df.loc[i, "color_name"]
    return cname

def speak(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()


#YOLO Model
model = YOLO("runs/detect/train/weights/best.pt")

#Main Webcam Loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Camera not found!")
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            if x1 < mouse_x < x2 and y1 < mouse_y < y2:
                
            # Get average color in object region (ROI)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                crop_size = 20
                roi = frame[cy-crop_size:cy+crop_size, cx-crop_size:cx+crop_size]

                if roi.size != 0:
                    avg_color = cv2.mean(roi)[:3]
                    b, g, r_val = map(int, avg_color)
                    color_name = getColorName(r_val, g, b)

                    # Draw bounding box + label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} | {color_name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Voice Output
                    speak(f"I see a {color_name} {label}")
                    cv2.imshow("YOLO + Color Detection + Voice", frame)
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
