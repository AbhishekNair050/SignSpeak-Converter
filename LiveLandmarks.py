import cv2
import time
import numpy as np
from collections import deque
from preprocess import *

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

sequence = deque(maxlen=input_shape[1])
for _ in range(input_shape[1]):
    sequence.append(np.zeros((input_shape[2], 3)))

step_length = 60
TIME_PER_STEP = step_length / 30.0
step_time = time.time()
frame_time = 0
step = []
label = ""

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    fps = str(int(1 / (time.time() - frame_time)))
    frame_time = time.time()

    # frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    frame_landmarks = get_frame_landmarks(frame_rgb)

    for point in frame_landmarks:
        X = int(point[0] * width)
        y = int(point[1] * height)
        cv2.circle(frame, (X, y), 2, (0, 255, 0), -1)
    cv2.putText(
        frame,
        fps,
        (560, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Label: {label}",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 0, 150),
        2,
        cv2.LINE_AA,
    )

    step.append(frame_landmarks)

    if time.time() - step_time >= TIME_PER_STEP:
        step = np.array(step)
        step = np.apply_along_axis(
            lambda arr: np.interp(
                np.linspace(0, 1, step_length), np.linspace(0, 1, arr.shape[0]), arr
            ),
            axis=0,
            arr=step,
        )

        sequence.extend(step)
        predictions = predict(np.array(sequence))
        predictions = predictions.reshape(-1)
        score = np.max(predictions)
        index = predictions.argmax()
        certainty = calculate_certainty(
            score, range_positive[index], range_negative[index]
        )

        if certainty > 0.7:
            label = index_label_mapping[str(index)]
        else:
            label = "None"
        # label = index_label_mapping[str(index)]
        print(f"Label: {label}, Certainty: {certainty * 100:.2f}%")
        step_time = time.time()
        step = []

    cv2.imshow("Test", frame)
    cv2.setWindowProperty("Test", cv2.WND_PROP_TOPMOST, 1)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
