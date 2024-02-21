import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from old_preprocessing import *
import time
import json

hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
filtered_hand = list(range(21))

filtered_hand = list(range(21))

filtered_pose = [11, 12, 13, 14, 15, 16]

filtered_face = [
    4,
    6,
    8,
    9,
    33,
    37,
    40,
    46,
    52,
    55,
    61,
    70,
    80,
    82,
    84,
    87,
    88,
    91,
    105,
    107,
    133,
    145,
    154,
    157,
    159,
    161,
    163,
    263,
    267,
    270,
    276,
    282,
    285,
    291,
    300,
    310,
    312,
    314,
    317,
    318,
    321,
    334,
    336,
    362,
    374,
    381,
    384,
    386,
    388,
    390,
    468,
    473,
]

HAND_NUM = len(filtered_hand)
POSE_NUM = len(filtered_pose)
FACE_NUM = len(filtered_face)


def get_frame_landmarks(frame):

    all_landmarks = np.zeros((HAND_NUM * 2 + POSE_NUM + FACE_NUM, 3))

    def get_hands(frame):
        results_hands = hands.process(frame)
        if results_hands.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                if results_hands.multi_handedness[i].classification[0].index == 0:
                    all_landmarks[:HAND_NUM, :] = np.array(
                        [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    )  # right
                else:
                    all_landmarks[HAND_NUM : HAND_NUM * 2, :] = np.array(
                        [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    )  # left

    def get_pose(frame):
        results_pose = pose.process(frame)
        if results_pose.pose_landmarks:
            all_landmarks[HAND_NUM * 2 : HAND_NUM * 2 + POSE_NUM, :] = np.array(
                [(lm.x, lm.y, lm.z) for lm in results_pose.pose_landmarks.landmark]
            )[filtered_pose]

    def get_face(frame):
        results_face = face_mesh.process(frame)
        if results_face.multi_face_landmarks:
            all_landmarks[HAND_NUM * 2 + POSE_NUM :, :] = np.array(
                [
                    (lm.x, lm.y, lm.z)
                    for lm in results_face.multi_face_landmarks[0].landmark
                ]
            )[filtered_face]

    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.submit(get_hands, frame)
        executor.submit(get_pose, frame)
        executor.submit(get_face, frame)

    return all_landmarks


def get_video_landmarks():
    cap = cv2.VideoCapture(0)
    all_frame_landmarks = []
    previous_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # frame.flags.writeable = False
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_landmarks = get_frame_landmarks(frame)
        frame_landmarks = np.expand_dims(frame_landmarks, axis=0)
        current_time = time.time()
        elapsed_time = current_time - previous_time
        all_frame_landmarks.append(frame_landmarks)
        previous_time = current_time
        fps = 1 / elapsed_time
        cv2.putText(
            frame,
            str(int(fps)) + " FPS",
            (7, 30),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(hands.reset)
        executor.submit(pose.reset)
        executor.submit(face_mesh.reset)

    return np.array(all_frame_landmarks)


landmarks = (
    [x for x in filtered_hand]
    + [x + HAND_NUM for x in filtered_hand]
    + [x + HAND_NUM * 2 for x in filtered_pose]
    + [x + HAND_NUM * 2 + POSE_NUM for x in filtered_face]
)
index_mapping_path = "Test/index_label_mapping.json"
index_mapping = json.load(open(index_mapping_path, "r"))
all_frame_landmarks = get_video_landmarks()
# all_frame_landmarks = np.array(all_frame_landmarks)
all_frame_landmarks = padding(all_frame_landmarks, length=120, pad=-100)
all_frame_landmarks = sequences(all_frame_landmarks, length=60, step=20, pad=-100)
all_frame_landmarks = interpolate(all_frame_landmarks, length=100)
all_frame_landmarks = padding11(all_frame_landmarks, length=120, pad=0)
model = get_model()
ypred = model.predict(all_frame_landmarks)
ypred = ypred.reshape(-1)
ypred = ypred.argmax()
print(ypred)
label = index_mapping[str(ypred)]
print(label)
