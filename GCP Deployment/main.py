from flask import Flask, render_template, request, jsonify, make_response
import numpy as np
from preprocess import *
import traceback
import cv2
from TextToSign.ASL_scraper import *
import google.generativeai as genai
from signtext import *
from googletrans import Translator
from moviepy.editor import VideoFileClip, concatenate_videoclips
import re
from google.cloud import *

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

app = Flask(__name__, static_folder="static", template_folder="templates")
translator = Translator()
sequence = deque(maxlen=input_shape[1])
for i in range(input_shape[1]):
    sequence.append(np.zeros((input_shape[2], 3)))

step_length = 90
TIME_PER_STEP = step_length / 30.0
previous_time = time.time()
frame_time = time.time()
global label
global certainty
global sentence
sentence = ""
label = ""
certainty = 0.0

api_key = "AIzaSyB1VIUzH3CfLjVGVflRdWG3rIx1t3wOlnE"


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
    global label, certainty, sentence
    return jsonify(
        {"label": label, "certainty": round(certainty * 100, 2), "sentence": sentence}
    )


def form_sentence(sequence):
    hardcoded_prompt = "form proper sentences using these words, these are predicted by a sign language ASL recognition model, give only one output: "
    sequence = [str(i) for i in sequence]
    for i in sequence:
        if i == "None" or i == "undefined":
            sequence.remove(i)
    if len(sequence) == 0:
        return "No proper sentence can be formed"
    inputt = hardcoded_prompt + " ".join(sequence)
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel("gemini-pro")
    result = llm.generate_content(inputt)
    sentence = result.text
    return sentence


@app.route("/process_frame", methods=["POST"])
def process_frame():
    global label, certainty, previous_time, frame_time, sequence, sentence
    all_landmarks = []
    seq = []
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
            seq.append(label)
            try:
                sentence = form_sentence(seq)
            except Exception as e:
                sentence = "No proper sentence can be formed"
            print(f"Sentence: {sentence}")
        return jsonify({"label": label, "certainty": certainty, "sentence": sentence})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


from google.cloud import storage
import tempfile


def combine_videos(
    video_paths,
    output_size=(640, 480),
    fps=30,
    bucket_name="staging.signtesti.appspot.com",
    output_blob_name="output.mp4",
):
    video_clips = []

    for video_path in video_paths:
        clip = VideoFileClip(video_path)
        if clip.size != output_size:
            clip = clip.resize(output_size)
        video_clips.append(clip)

    final_clip = concatenate_videoclips(video_clips)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file_path = temp_file.name
        final_clip.write_videofile(
            temp_file_path, fps=fps, codec="libx264", audio_codec="aac"
        )
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(output_blob_name)
    blob.upload_from_filename(temp_file_path)
    os.remove(temp_file_path)
    return f"gs://{bucket_name}/{output_blob_name}"


@app.route("/texttosign", methods=["POST"])
def texttosign():
    videos = []
    sentence = request.json.get("sentence", "")
    if re.match("^[a-zA-Z]*$", sentence):
        sentence = sentence.lower()
    else:
        sentence = translator.translate(sentence, dest="en").text.lower()
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

    combined_video_frames = combine_videos(videos)
    if combined_video_frames:
        gif_bytes = signvector()

    # os.remove("output.mp4")
    # os.remove("outputnew.gif")
    response = make_response(gif_bytes)
    response.headers.set("Content-Type", "image/gif")
    response.headers.set("Content-Disposition", "attachment", filename="output.gif")
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
