# traffic_violation_sim_demo.py
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import os
import time
import sqlite3
import csv
import matplotlib.pyplot as plt

# ----------- Config ----------
video_path = "tr.mp4"
model = YOLO("yolov8n.pt")
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

vehicle_classes = ["bicycle", "car", "motorcycle", "bus", "truck"]
traffic_light_class = "traffic light"
violated_ids = set()
save_folder = "violations"
os.makedirs(save_folder, exist_ok=True)

# Database (violations + simulated notifications)
db_path = "traffic_violations.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS violations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT,
    timestamp TEXT,
    violation_type TEXT,
    penalty INTEGER,
    image_path TEXT,
    phone TEXT
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS sent_notifications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT,
    timestamp TEXT,
    method TEXT,
    message TEXT,
    sent_to TEXT
)
""")
conn.commit()

# CSV log for simulated notifications (human-readable)
notif_csv = "simulated_notifications.csv"
if not os.path.exists(notif_csv):
    with open(notif_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vehicle_id", "timestamp", "method", "message", "sent_to"])

# Simple mock owner DB (demo). In real system you'd use plate recognition and a real registry.
vehicle_owners = {
    # tracker_id: owner record (demo only)
    1: {"plate": "TN09AB1234", "phone": "+919876543210"},
    2: {"plate": "AP02XY6789", "phone": "+919812345678"},
    # Add more as needed for demo
}

# Counters for graph
counts = {"VIOLATION": 0, "MOVING": 0, "NORMAL": 0}

# ---------- Helper functions ----------
def simulate_send_notification(vehicle_plate, phone, message, method="SMS"):
    """Simulate sending a message: log to DB, CSV and print to console."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    # Insert into DB
    c.execute("INSERT INTO sent_notifications(vehicle_id, timestamp, method, message, sent_to) VALUES (?, ?, ?, ?, ?)",
              (vehicle_plate, ts, method, message, phone))
    conn.commit()
    # Append to CSV
    with open(notif_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([vehicle_plate, ts, method, message, phone])
    # Print to console (demo)
    print(f"[SIMULATED {method}] to {phone} | vehicle: {vehicle_plate} | time: {ts}")
    print("Message body:")
    print(message)
    print("-" * 60)

def traffic_light_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    # handle invalid box
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)
    if x2 <= x1 or y2 <= y1:
        return "UNKNOWN"
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, np.array([0,100,100]), np.array([10,255,255])) + \
               cv2.inRange(hsv, np.array([160,100,100]), np.array([180,255,255]))
    mask_orange = cv2.inRange(hsv, np.array([10,100,100]), np.array([25,255,255]))
    mask_green = cv2.inRange(hsv, np.array([40,50,50]), np.array([90,255,255]))
    counts = {"RED": cv2.countNonZero(mask_red),
              "ORANGE": cv2.countNonZero(mask_orange),
              "GREEN": cv2.countNonZero(mask_green)}
    return max(counts, key=counts.get)

# ---------- Video loop ----------
cap = cv2.VideoCapture(video_path)
frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1
    frame = cv2.resize(frame, (1280, 720))
    results = model.predict(frame, conf=0.5, verbose=False)

    detections = []
    traffic_lights = []

    # Collect detections
    for r in results:
        boxes = r.boxes.xyxy
        confs = r.boxes.conf
        classes = r.boxes.cls
        for box, conf, cls in zip(boxes, confs, classes):
            cls_name = model.model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            if cls_name in vehicle_classes:
                detections.append([x1, y1, x2, y2, float(conf)])
            elif cls_name == traffic_light_class:
                traffic_lights.append([x1, y1, x2, y2])

    dets = np.array(detections) if len(detections) > 0 else np.empty((0,5))
    tracks = tracker.update(dets)

    # Determine current light and stop line
    current_light = "GREEN"
    stop_line_y = 0
    if len(traffic_lights) > 0:
        traffic_lights = sorted(traffic_lights, key=lambda x: x[1])  # topmost
        current_light = traffic_light_color(frame, traffic_lights[0])
        stop_line_y = traffic_lights[0][3] + 50
        tx1, ty1, tx2, ty2 = traffic_lights[0]
        color_map = {"RED": (0,0,255), "ORANGE": (0,165,255), "GREEN": (0,255,0)}
        if current_light in color_map:
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color_map[current_light], 2)
            cv2.putText(frame, current_light, (tx1, ty1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[current_light], 2)

    # draw stop line
    cv2.line(frame, (0, stop_line_y), (frame.shape[1], stop_line_y), (255,255,255), 2)

    # process tracked vehicles
    for tr in tracks:
        if len(tr) < 5:
            continue
        x1, y1, x2, y2, track_id = tr.astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # decide label
        if track_id in violated_ids:
            label = "VIOLATION"
            color = (0,0,255)
        elif cy > stop_line_y and current_light == "RED":
            label = "VIOLATION"
            color = (0,0,255)
            violated_ids.add(track_id)
            counts["VIOLATION"] += 1

            # Save cropped image of violator
            vehicle_crop = frame[max(y1,0):max(y1,0)+max(1,y2-y1), max(x1,0):max(x1,0)+max(1,x2-x1)]
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            img_path = os.path.join(save_folder, f"vehicle_{track_id}_{int(time.time())}.jpg")
            if vehicle_crop.size > 0:
                cv2.imwrite(img_path, vehicle_crop)

            # Lookup owner (demo). If not present, use placeholder.
            owner = vehicle_owners.get(track_id, {"plate": f"UNKNOWN_{track_id}", "phone": "+0000000000"})
            plate = owner["plate"]
            phone = owner["phone"]
            penalty = 500  # demo amount

            # Insert violation record into DB
            c.execute("INSERT INTO violations(vehicle_id, timestamp, violation_type, penalty, image_path, phone) VALUES (?, ?, ?, ?, ?, ?)",
                      (plate, ts, "RED LIGHT", penalty, img_path, phone))
            conn.commit()

            # Simulate sending a notification (no real SMS). This logs to DB & CSV.
            message = (f"NOTICE: Your vehicle {plate} was detected violating a red light on {ts}. "
                       f"Penalty: ₹{penalty}. See evidence: {img_path}")
            simulate_send_notification(plate, phone, message, method="SIMULATED_SMS")

        elif cy > stop_line_y and current_light == "ORANGE":
            label = "MOVING"
            color = (0,165,255)
            counts["MOVING"] += 1
        else:
            label = "NORMAL"
            color = (0,255,0)
            counts["NORMAL"] += 1

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label}-{int(track_id)}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # show frame (press q to exit)
    cv2.imshow("Smart Traffic Violation System (Simulated Notifications)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# --------- After run: show/save summary and graph ----------
print("Run finished. Counts:", counts)

# Create bar chart
labels = list(counts.keys())
values = [counts[k] for k in labels]
plt.figure(figsize=(6,4))
bars = plt.bar(labels, values)
plt.title("Vehicle Status Counts")
plt.ylabel("Number of occurrences")
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{int(h)}", ha='center')
plt.tight_layout()
chart_path = "violation_stats.png"
plt.savefig(chart_path)
print(f"Saved stats chart to {chart_path}")
plt.show()

# Close DB
conn.close()

