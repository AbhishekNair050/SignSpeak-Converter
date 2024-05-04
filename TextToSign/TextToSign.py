import cv2
import mediapipe as mp
import numpy as np
import subprocess
import imageio


def signvector(inputpath, output="outputnew.gif"):
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
    video_path = inputpath  # Replace with your video path
    output_path = output  # Replace with desired output path
    # audio_extraction_cmd = (
    #     f"ffmpeg -y -i {inputpath} -vn -acodec copy original_audio.aac"
    # )
    # subprocess.run(audio_extraction_cmd, shell=True)
    # Read video capture
    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Cannot read frame from video stream.")
            break

        results_hands = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results_pose = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        results_face_mesh = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        op_img = np.zeros_like(img)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    op_img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 0, 0), thickness=2, circle_radius=1
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=0.5
                    ),
                )

                landmarks_np = np.array(
                    [
                        (lm.x * img.shape[1], lm.y * img.shape[0])
                        for lm in hand_landmarks.landmark
                    ]
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

        # Use the provided list of filtered face landmarks
        filtered_face = [
            0,
            4,
            7,
            8,
            10,
            13,
            14,
            17,
            21,
            33,
            37,
            39,
            40,
            46,
            52,
            53,
            54,
            55,
            58,
            61,
            63,
            65,
            66,
            67,
            70,
            78,
            80,
            81,
            82,
            84,
            87,
            88,
            91,
            93,
            95,
            103,
            105,
            107,
            109,
            127,
            132,
            133,
            136,
            144,
            145,
            146,
            148,
            149,
            150,
            152,
            153,
            154,
            155,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            172,
            173,
            176,
            178,
            181,
            185,
            191,
            234,
            246,
            249,
            251,
            263,
            267,
            269,
            270,
            276,
            282,
            283,
            284,
            285,
            288,
            291,
            293,
            295,
            296,
            297,
            300,
            308,
            310,
            311,
            312,
            314,
            317,
            318,
            321,
            323,
            324,
            332,
            334,
            336,
            338,
            356,
            361,
            362,
            365,
            373,
            374,
            375,
            377,
            378,
            379,
            380,
            381,
            382,
            384,
            385,
            386,
            387,
            388,
            389,
            390,
            397,
            398,
            400,
            402,
            405,
            409,
            415,
            454,
            466,
            468,
            473,
        ]

        # Loop over the detected face landmarks
        if results_face_mesh.multi_face_landmarks:
            for face_landmarks in results_face_mesh.multi_face_landmarks:
                # Draw filtered face landmarks with small circles
                for landmark_idx in filtered_face:
                    try:
                        landmark_pt = face_landmarks.landmark[landmark_idx]
                        landmark_px = int(landmark_pt.x * img.shape[1])
                        landmark_py = int(landmark_pt.y * img.shape[0])
                        cv2.circle(
                            op_img,
                            (landmark_px, landmark_py),
                            radius=1,
                            color=(0, 255, 0),
                            thickness=-1,
                        )
                    except:
                        pass

        frames.append(op_img)

    cap.release()
    cv2.destroyAllWindows()

    imageio.mimsave(output_path, frames, fps=30)

    return output_path
