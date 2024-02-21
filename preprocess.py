import cv2
import json
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from concurrent.futures import ThreadPoolExecutor

model_path = "Models\model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

filtered_hand = list(range(21))
filtered_pose = [11, 12, 13, 14, 15, 16]

num_hand = len(filtered_hand)
num_pose = len(filtered_pose)

hands = mp.solutions.hands.Hands()
pose = mp.solutions.pose.Pose()

input_shape = list(map(int, interpreter.get_input_details()[0]["shape"]))
output_shape = list(map(int, interpreter.get_output_details()[0]["shape"]))
input_shape, output_shape


def get_frame_landmarks(frame):

    all_landmarks = np.zeros((num_hand * 2 + num_pose, 3))

    def get_hands(frame):
        results_hands = hands.process(frame)
        if results_hands.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                if results_hands.multi_handedness[i].classification[0].index == 0:
                    all_landmarks[:num_hand, :] = np.array(
                        [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    )  # right
                else:
                    all_landmarks[num_hand : num_hand * 2, :] = np.array(
                        [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    )  # left

    def get_pose(frame):
        results_pose = pose.process(frame)
        if results_pose.pose_landmarks:
            all_landmarks[num_hand * 2 : num_hand * 2 + num_pose, :] = np.array(
                [(lm.x, lm.y, lm.z) for lm in results_pose.pose_landmarks.landmark]
            )[filtered_pose]

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(get_hands, frame)
        executor.submit(get_pose, frame)

    return all_landmarks


gloss_mapping_path = "Test\\590_gloss_mapping.json"
index_gloss_mapping_path = "Test\\590_index_gloss_mapping.json"
index_label_mapping_path = "Test\\590_index_label_mapping.json"

gloss_mapping = json.load(open(gloss_mapping_path, "r"))
index_gloss_mapping = json.load(open(index_gloss_mapping_path, "r"))
index_label_mapping = json.load(open(index_label_mapping_path, "r"))

range_positive = np.load("Test\\range_positive.npy")
range_negative = np.load("Test\\range_negative.npy")


def predict(input_data):
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    return output


def calculate_certainty(pred, pos_range, neg_range):
    z_score_true = abs(pred - pos_range[1]) / (pos_range[2] - pos_range[0])
    z_score_false = abs(pred - neg_range[1]) / (neg_range[2] - neg_range[0])
    return z_score_false / (z_score_true + z_score_false)
