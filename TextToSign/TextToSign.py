import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Video input and output paths
video_path = "output.mp4"  # Replace with your video path
output_path = "output_realistic.mp4"  # Replace with desired output path

# Read video capture
cap = cv2.VideoCapture(video_path)

# Define video writer for output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Video codec
width, height = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)
video_writer = cv2.VideoWriter(
    output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height)
)

# Filtered face landmarks indices
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

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Cannot read frame from video stream.")
        break

    results_hands = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results_pose = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results_face_mesh = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    op_img = np.zeros([height, width, 3], dtype=np.uint8)
    op_img.fill(255)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                op_img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=1
                ),
            )

            landmarks_np = np.array(
                [(lm.x * width, lm.y * height) for lm in hand_landmarks.landmark]
            )

            for i in range(len(landmarks_np) - 1):
                cv2.line(
                    op_img,
                    (int(landmarks_np[i][0]), int(landmarks_np[i][1])),
                    (int(landmarks_np[i + 1][0]), int(landmarks_np[i + 1][1])),
                    (0, 0, 255),
                    2,
                )

    if results_pose.pose_landmarks:
        mp_draw = mp.solutions.drawing_utils
        mp_draw.draw_landmarks(
            op_img,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec((0, 255, 0), 3, 1),
            mp_draw.DrawingSpec((255, 0, 255), 3, 1),
        )

    if results_face_mesh.multi_face_landmarks:
        for face_landmarks in results_face_mesh.multi_face_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                op_img,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,  # Disable landmark drawing
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                ),
            )

    video_writer.write(op_img)

    # cv2.imshow("Realistic Avatar", op_img)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
