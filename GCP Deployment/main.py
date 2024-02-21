from flask import Flask, render_template, request, jsonify
import numpy as np
from preprocess import *
import traceback
import cv2

app = Flask(__name__, static_folder="static", template_folder="templates")

sequence = deque(maxlen=input_shape[1])
for i in range(input_shape[1]):
    sequence.append(np.zeros((input_shape[2], 3)))

step_length = 60
TIME_PER_STEP = step_length / 30.0
previous_time = time.time()
frame_time = time.time()
global label
global certainty
label = ""
certainty = 0.0


@app.route("/", methods=["GET"])
def index():
    return render_template("page1.html")


@app.route("/ourvision", methods=["GET"])
def ourvision():
    return render_template("ourvision.html")


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")


@app.route("/contactus", methods=["GET"])
def contactus():
    return render_template("contactus.html")


@app.route("/trynow", methods=["GET"])
def trynow():
    global label, certainty
    return render_template(
        "trynow.html", label=label, certainty=round(certainty * 100, 2)
    )


@app.route("/update_labels_certainty", methods=["POST"])
def update_labels_certainty():
    global label, certainty
    return jsonify({"label": label, "certainty": round(certainty * 100, 2)})


@app.route("/process_frame", methods=["POST"])
def process_frame():
    global label
    global certainty
    global previous_time
    global frame_time
    global sequence
    all_landmarks = []
    try:
        epsilon = 1e-6
        frame_file = request.files["frame"]
        frame_data = bytearray(frame_file.read())
        frame_np = np.asarray(frame_data, dtype=np.uint8)
        current_time = time.time()

        if current_time - frame_time > epsilon:
            fps = str(int(1 / (current_time - frame_time)))
        else:
            fps = "0"

        frame_time = current_time

        frame_np = np.array(frame_np)

        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        height, width, _ = frame.shape

        frame_np = np.array(frame)
        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        frame_landmarks = get_frame_landmarks(frame_rgb)
        all_landmarks.append(frame_landmarks)
        if time.time() - previous_time >= TIME_PER_STEP:
            all_landmarks = np.array(all_landmarks)
            all_landmarks = np.apply_along_axis(
                lambda arr: np.interp(
                    np.linspace(0, 1, step_length), np.linspace(0, 1, arr.shape[0]), arr
                ),
                axis=0,
                arr=all_landmarks,
            )
            sequence.extend(all_landmarks)
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
            print(f"Label: {label}, Certainty: {certainty * 100:.2f}%")
            previous_time = time.time()
            certainty = round(certainty * 100, 2)

        return jsonify({"label": label, "certainty": certainty})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
