import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7
)
# mp_face = mp.solutions.face_detection
# face = mp_face.FaceDetection()
# Initialize the video capture
video_path = "output.mp4"
cap = cv2.VideoCapture(video_path)


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (640, 480))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 20, 150)
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    results = hands.process(frame)
    # results_face = face.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                edges_3channel, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
    # if results_face.detections:
    #     for detection in results_face.detections:
    #         mp.solutions.drawing_utils.draw_detection(edges_3channel, detection)

    out.write(edges_3channel)

cap.release()
out.release()
cv2.destroyAllWindows()
