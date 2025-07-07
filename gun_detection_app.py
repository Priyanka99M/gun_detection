import cv2
import imutils
import datetime
import os

# Load Haar cascade
gun_cascade = cv2.CascadeClassifier('cascade.xml')
if gun_cascade.empty():
    print("[ERROR] Could not load cascade.xml. Check path.")
    exit()

# Initialize webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("[ERROR] Cannot access camera. Check permissions in System Settings.")
    exit()

# Create alerts folder if it doesn't exist
if not os.path.exists("alerts"):
    os.makedirs("alerts")

print("[INFO] Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        print("[ERROR] Failed to grab frame.")
        break

    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    guns = gun_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=20,
        minSize=(100, 100)
    )

    if len(guns) > 0:
        for (x, y, w, h) in guns:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "GUN DETECTED!", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save snapshot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"alerts/alert_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[ALERT] Gun detected! Image saved as {filename}")

    # Add timestamp
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow("Gun Detection Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[INFO] Exiting.")
        break

camera.release()
cv2.destroyAllWindows()

