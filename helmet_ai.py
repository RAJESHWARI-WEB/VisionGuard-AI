from ultralytics import YOLO
import cv2
import winsound

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

violation_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    unsafe_workers = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if name == "person":
            unsafe_workers += 1
            violation_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "NO HELMET ALERT!", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Beep Sound
            winsound.Beep(300, 100)  # frequency, duration

    cv2.putText(frame, f"Unsafe Workers: {unsafe_workers}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Total Violations: {violation_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("AI Construction Safety Monitoring", frame)
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
