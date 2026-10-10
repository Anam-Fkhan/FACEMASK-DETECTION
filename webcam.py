
import cv2
from PIL import Image
from transformers import pipeline

print("Loading AI model. Please wait...")
model = pipeline(
    "image-classification",
    model="prithivMLmods/Face-Mask-Detection"
)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found. Check camera permissions.")
    raise SystemExit

print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray, 1.1, 5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(rgb_face)

        predictions = model(pil_face)
        best = max(predictions, key=lambda p: p["score"])
        label = best["label"].lower()
        confidence = best["score"]

        if "without" in label or "no_mask" in label or "no mask" in label:
            text = f"No Mask {confidence:.0%}"
            color = (0, 0, 255)
        elif "mask" in label:
            text = f"Mask {confidence:.0%}"
            color = (0, 200, 0)
        else:
            text = f"{best['label']} {confidence:.0%}"
            color = (0, 165, 255)

        cv2.rectangle(
            frame, (x, y), (x+w, y+h), color, 2
        )
        cv2.putText(
            frame, text, (x, max(y-10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
        )

    cv2.imshow("AI Face Mask Detection - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()