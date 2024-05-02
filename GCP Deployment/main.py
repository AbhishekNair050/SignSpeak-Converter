from flask import Flask, render_template, request, jsonify, send_file, make_response
import numpy as np
from preprocess import *
import traceback
import cv2
from TextToSign.ASL_scraper import *
import subprocess, ffmpeg

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
global sentence
sentence = ""
label = ""
certainty = 0.0


@app.route("/", methods=["GET"])
def index():
    return render_template("page1.html")


@app.route("/vision", methods=["GET"])
def ourvision():
    return render_template("ourvision.html")


@app.route("/impact", methods=["GET"])
def about():
    return render_template("impact.html")


@app.route("/contactus", methods=["GET"])
def contactus():
    return render_template("contactus.html")


@app.route("/trynow", methods=["GET"])
def trynow():
    global label, certainty
    return render_template(
        "trynow.html", label=label, certainty=round(certainty * 100, 2)
    )


@app.route("/use", methods=["GET"])
def use():
    return render_template("use.html")


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
    global sentence
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


def combine_videos(video_paths, output_path, output_size=(640, 480), fps=30):
    video_list = []

    for video_path in video_paths:
        video = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = video.read()

            if not ret:  
                print(f"Finished processing video: {video_path}")
                break

            frame = cv2.resize(frame, output_size)
            frames.append(frame)

        video.release()

        temp_video_path = os.path.join(
            os.path.dirname(output_path), f"temp_{os.path.basename(video_path)}"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, output_size)
        for frame in frames:
            out.write(frame)
        out.release()

        video_list.append(temp_video_path)

    audio_path = "stock_audio.mp3"
    video_list_file = os.path.join(os.path.dirname(output_path), "video_list.txt")
    with open(video_list_file, "w") as f:
        for video_path in video_list:
            f.write(f"file '{video_path}'\n")

    ffmpeg_cmd = f"ffmpeg -y -f concat -safe 0 -i {video_list_file} -i {audio_path} -c:v libx264 -c:a aac -shortest {output_path}"
    subprocess.run(ffmpeg_cmd, shell=True, check=True)

    for video_path in video_list:
        os.remove(video_path)
    os.remove(video_list_file)

    return output_path


@app.route("/texttosign", methods=["POST"])
def texttosign():
    videos = []
    sentence = request.json.get("sentence", "")
    database_dir = "database"
    word = sentence.split(" ")

    for i in word:
        if not os.path.exists(f"{database_dir}/{i}.mp4"):
            res = download_video(i)
            if res:
                os.rename(f"{i}.mp4", f"{database_dir}/{i}.mp4")
                videos.append(f"{database_dir}/{i}.mp4")
            else:
                print("Error downloading video.")
        else:
            videos.append(f"{database_dir}/{i}.mp4")

    combined_video_frames = combine_videos(videos, "output.mp4")
    # cap = cv2.VideoCapture("output.mp4")
    # # Encode the combined video using cv2.VideoWriter
    # output_size = (640, 480)  # Adjust the output size as per your requirements
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fps = 30  # Adjust the FPS as per your requirements
    # out = cv2.VideoWriter("temp.mp4", fourcc, fps, output_size)

    # for frame in combined_video_frames:
    #     frame = cv2.resize(frame, output_size)  # Resize the frame if needed
    #     out.write(frame)

    # out.release()

    with open("output.mp4", "rb") as f:
        video_bytes = f.read()

    # os.remove("temp.mp4")

    response = make_response(video_bytes)
    response.headers.set("Content-Type", "video/mp4")
    response.headers.set("Content-Disposition", "attachment", filename="output.mp4")
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
