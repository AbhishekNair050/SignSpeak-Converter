# SignSpeak Converter - Breaking the Silence, Connecting Lives.

SignSpeak Converter is a web application that translates sign language gestures into text in real-time using machine learning models. The project aims to bridge the communication gap between the deaf and hearing communities, making conversations more accessible and inclusive.

the application can be accessed at - https://project-signspeak.appspot.com/
**disclaimer -** 
- It takes time to boot for the first time due to automatic scaling.
- due to limited redource, text to sign only works with one word as of now on this deployment, with more words, the response time is too high casing the process to die.
## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Data Preprocessing](#data-preprocessing)
- [Deployment](#deployment)
- [Future Plan](#Future-Plan)

## Project Structure

The project is structured as follows:

```
SignSpeak/
├── GCP Deployment/
│   ├── static/
│   ├── templates/
│   ├── assets/
│   ├── main.py
│   ├── app.yaml
│   └── requirements.txt
├── frontend/
├── models/
│   └──model.tflite
├── Test/
├── unoptimized model/
├── SignSpeak.ipynb
├── preprocess/
└── livelandmarks/
```

- **GCP Deployment**: Contains files for deploying the application on Google Cloud Platform (GCP).
  - `static`: Stores CSS files and images.
  - `templates`: Stores HTML files and assets (models and other necessary files).
  - `main.py`: Flask application for handling the web server and real-time sign language translation.
  - `signtext.py`: Text to Sign avatar creation
  - `app.yaml`: Configuration file for App Engine deployment.
  - `requirements.txt`: Python dependencies.
- **frontend**: Contains HTML and CSS files for the web interface.
- **models**: Stores the TFLite models for web deployment.
- **Test**: Stores preprocessed data files.
- **unoptimized model**: Contains the raw H5 model and related code (not used for web deployment due to size constraints).
- **SignSpeak.ipynb**: Main Jupyter Notebook for data preprocessing, model training, and evaluation.
- **preprocess**: Contains utility functions for preprocessing data and extracting landmarks.
- **livelandmarks**: Contains utility functions for live landmark extraction (used in `main.py`).
  
## Features
### Sign-to-Text Conversion Pipeline

1. **Input Sentence**: The user provides a textual sentence as input to the application.
2. **Split the Word and Download Videos**: The sentence is tokenized into individual words, and pre-recorded sign language video clips for each word are downloaded from a database or repository.
3. **Combine Videos**: The individual word videos are concatenated to form a single continuous video representing the entire input sentence in sign language.
4. **Extract Landmarks**: Computer vision techniques are employed to detect and extract facial landmarks and hand keypoints from the combined video.
5. **Create Avatar as a GIF**: Using the extracted landmarks, an animated avatar is rendered as a GIF, performing the sign language gestures corresponding to the input sentence.
6. **Display the GIF**: The generated GIF animation is presented to the user as the visual representation of the input text in sign language.
![1](https://github.com/AbhishekNair050/SignSpeak-Converter/assets/114457983/99174843-0545-4c8b-8041-638a0e75a7b1)

### Real-Time Inference

1. **Website Inference**: The real-time inference process is initiated from the application's web interface.
2. **Capture from Webcam**: The user's sign language gestures are captured in real-time through the device's webcam.
3. **Send it to Backend Frame by Frame**: The webcam feed is segmented into individual frames, which are transmitted to the backend server for processing.
4. **Decode the Bytes**: The received frames are decoded from their byte representation into a format suitable for further processing.
5. **Process the Frames**: The decoded frames undergo pre-processing steps, such as normalization and augmentation, to enhance the quality of the input data.
6. **LLM Gemini Pro**: A large language model (LLM), specifically the Gemini Pro model, is employed for sign language gesture recognition and translation.
7. **Form Coherent Sentences**: The LLM processes the input frames and forms coherent sentences by recognizing and interpreting the sign language gestures.
8. **Add the Words to the Sequence**: As the LLM processes each frame, it appends the recognized words to a sequence, building the complete translated sentence incrementally.
9. **Update Label, Certainty, Sentence**: The LLM updates the predicted label (the recognized word or gesture), the certainty score associated with the prediction, and the current state of the translated sentence.
10. **Predict the Label**: The LLM continuously predicts the most likely label (word or gesture) based on the input frames.
11. **Display it in the Website**: The translated text output from the LLM is displayed in real-time on the application's web interface, providing the user with the textual interpretation of their sign language gestures.
![2](https://github.com/AbhishekNair050/SignSpeak-Converter/assets/114457983/4bb41c4a-c4de-4683-84ab-58a96af1c163)

### Text to Sign

The Text-to-Sign feature of SignSpeak Converter allows users to input a text sentence and receive a GIF animation of an avatar signing the corresponding gestures. The process involves the following steps:
1. **Input Sentence**: The user provides a textual sentence as input to the application.
2. **Split 
the Word and Download Videos**: The sentence is tokenized into individual words, and pre-recorded sign language video clips for each word are downloaded from a database or repository.
3. **Combine Videos**: The individual word videos are concatenated to form a single continuous video representing the entire input sentence in sign language.
4. **Extract Landmarks**: Computer vision techniques are employed to detect and extract facial landmarks and hand keypoints from the combined video.
5. **Create Avatar as a GIF**: Using the extracted landmarks, an animated avatar is rendered as a GIF, performing the sign language gestures corresponding to the input sentence.
6. **Display the GIF**: The generated GIF animation is presented to the user as the visual representation of the input text in sign language.
![Untitled design](https://github.com/AbhishekNair050/SignSpeak-Converter/assets/114457983/12c98b5d-e03a-43cd-b6a4-faa7f632cf60)

## Installation


1. Clone the repository:

```
git clone https://github.com/AbhishekNair050/SignSpeak-Converter.git
```
2. Go to "GCP Deployment"
```
cd "../GCP Deployment"
```
3. Install the required Python packages:

```
pip install -r requirements.txt
```

## Usage
1. Due to storage read/write permission issue on GCP deployment, we implemented a direct read write from Google Cloud Storage. While running on localhost, comment the `combine_video` function in `main.py` and uncomment the function below it. Also comment the Cloud Storage implementation in `signtext.py` for it to work on your local machine
   
2. Run the Flask application from /GCP Deployment:

```
python main.py
```

3. Open your web browser and navigate to `http://localhost:5000`.

4. Allow access to your webcam when prompted.

5. Go to http://localhost:5000/trynow and you will see two sections:
   - **Sign to Text**: Start signing, the application will translate your gestures into text in real-time.
   - **Text to Sign**: Enter a text sentence, and the application will generate a GIF animation of an avatar signing the corresponding gestures.

`main.py` is the Flask application responsible for setting up the web server and handling the real-time sign language translation process. It receives video frames from the client, processes them to extract landmarks, and performs sign language translation using our pre-trained model. The translated label and the associated certainty are then returned to the client as a JSON response. These labels are added to a sequence which is then feeded to Gemini Pro to form actual sentences. Additionally, it handles the text-to-sign feature by tokenizing the input text, retrieving pre-recorded sign language videos for each word, and combining them into a single GIF animation.

The application is also deployed on Google Cloud Platform (GCP) and can be accessed at https://project-signspeak.appspot.com/

## Models
The model architecture is based on a transformer structure with convolutional and self-attention blocks. Here are the key components:

- **Stem Convolution:** A dense layer is used to project the input landmarks to a lower dimension.

- **Convolutional Blocks:** A series of convolutional blocks are applied, each consisting of a depthwise separable convolution, batch normalization, and an Efficient Channel Attention (ECA) layer.

- **Transformer Blocks:** Transformer blocks are introduced, consisting of a multi-head self-attention layer and a feed-forward network. These blocks capture long-range dependencies in the input sequence.

- **Top Convolution:** A dense layer is applied to project the features to a higher dimension.

- **Global Average Pooling:** A global average pooling layer is used to aggregate the temporal information.

- **Late Dropout:** A late dropout layer is applied to prevent overfitting.

- **Classification Head:** A final dense layer with softmax activation is used for multi-class classification, where the output corresponds to the glosses.

The model is compiled with a sparse categorical cross-entropy loss and optimized using the Rectified Adam optimizer with a custom learning rate schedule (OneCycleLR). The Lookahead optimizer wrapper is also used for better convergence.

**Model Optimization and Deployment:**

After training, the model is saved in the H5 format and converted to a TFLite model for efficient deployment on web and mobile applications. The TFLite model is evaluated on the test set, and the gloss accuracy and label accuracy are reported.
The project uses two TFLite models for sign language translation:

1. **Hand and Pose Model**: This model uses hand and pose landmarks to translate sign language gestures.
2. **Hand, Face, and Pose Model**: This model incorporates hand, face, and pose landmarks for improved translation accuracy.
We ended up using the model trained on just the hand and pose landmarks as it performed better and gave better results.

The models are located in the `models` directory.

Additionally, the code includes a section for calculating the confidence range and certainty of the model's predictions. This involves estimating the positive and negative ranges for each class based on the mean and standard deviation of the predictions. A certainty score can then be calculated using these ranges, which could be useful for interpreting the model's outputs during inference.

## Data Preprocessing

The `SignSpeak.ipynb` notebook contains the code for data preprocessing, including:

- Data splitting: The dataset is split into training, validation and test sets for model development and evaluation.

- Landmark extraction: MediaPipe is used to extract hand, pose and facial landmarks from the video frames. 

- Data encoding: The extracted landmarks are encoded as NumPy arrays for efficient usage during training.

- Data augmentation: Techniques like rotation, shifting and zooming are applied to boost model generalization. 

- Label encoding: The textual labels are encoded as integer classes for model training.

- Label filtering: FastText removes semantically irrelevant labels to focus the model on key terms.

- Model training and evaluation: The model is trained on the processed dataset and evaluated on the test set for metrics like accuracy.

### Landmark Extraction

The MediaPipe library is used to extract hand, pose, and face landmarks from the video frames. Two versions of landmark extraction are implemented:

1. **Including Face Landmarks:**
   - The `filtered_hand` list contains indices for 21 hand landmarks.
   - The `filtered_pose` list contains indices for 6 pose landmarks.
   - The `filtered_face` list contains indices for 147 face landmarks.
   - The `get_frame_landmarks` function uses the MediaPipe solutions for hands, pose, and face mesh to extract the corresponding landmarks from the input frame.
   - The extracted landmarks are concatenated and returned as a single NumPy array.

2. **Excluding Face Landmarks (Better Results):**
   - The `filtered_hand` list contains indices for 21 hand landmarks.
   - The `filtered_pose` list contains indices for 13 pose landmarks.
   - The `get_frame_landmarks` function uses the MediaPipe solutions for hands and pose to extract the corresponding landmarks from the input frame.
   - The extracted hand and pose landmarks are concatenated and returned as a single NumPy array.

The `get_video_landmarks` function is used to extract landmarks for an entire video. It reads frames from the video file, calls `get_frame_landmarks` for each frame, and stores the landmarks in a list. Parallel processing with `ThreadPoolExecutor` is used for efficient landmark extraction.

### Data Encoding

The extracted video landmarks are saved as NumPy arrays for efficient loading during training and inference. The `draw_landmarks` function is provided to visualize the extracted landmarks on the video frames.

### Data Augmentation

Various augmentation techniques are applied to the training data to improve model generalization:

- **Rotation:** The `rotate`, `rotate_z`, `rotate_y`, and `rotate_x` functions apply rotations along different axes using rotation matrices.
- **Zoom:** The `zoom` function applies random zooming by scaling the non-zero landmarks around the center.
- **Shift:** The `shift` function applies random shifts by translating the non-zero landmarks.
- **Mask:** The `mask` function randomly masks a portion of the hand, pose, and face landmarks by setting them to zero.
- **Speedup:** The `speedup` function skips every other frame, effectively increasing the video speed.

The `apply_augmentations` function randomly applies a combination of these augmentation techniques to the input data.

The `augment` function creates augmented copies of the training data by calling `apply_augmentations` multiple times for each sample.

### Permutation

The training, validation, and test data are randomly permuted using `np.random.permutation` to improve generalization and prevent overfitting.

### Padding

The sequences are padded to a fixed length (120 frames) using the `padding` function. Sequences shorter than the fixed length are padded with zeros, while longer sequences are truncated.

### Sequencing

The padded sequences are further divided into smaller sub-sequences of length 60, with a step size of 20 frames, using the `sequences` function. This helps the model capture temporal dependencies within shorter sequences.

### Interpolation

The sub-sequences are interpolated to a fixed length (100 frames) using linear interpolation with the `interpolate` function, then they are furhter sequenced to 120 frames length. This ensures consistent input shapes for the model.

### Label Encoding

The labels (glosses) are encoded as integers using a mapping dictionary. The `gloss_mapping`, `index_gloss_mapping`, and `index_label_mapping` dictionaries are created and saved as JSON files for future reference.

### Label Filtering

The FastText library is used to filter the labels based on their semantic similarity to a set of predefined words related to the target domain (e.g., taxi-related words). This step helps to reduce the number of classes and improve model performance by focusing on the most relevant labels.

## Text To Sign
The Text-to-Sign feature involves tokenizing the input text sentence into individual words, retrieving pre-recorded sign language video clips for each word from a database, concatenating these videos into a single continuous video representation, extracting relevant landmarks (hand keypoints, body pose, facial features) using computer vision techniques, rendering an animated avatar mimicking the sign language gestures and facial expressions based on the extracted landmarks, compiling the rendered animations into a GIF file, and ultimately displaying the GIF animation to the user as the visual interpretation of their input text in sign language.

## Deployment

The application is deployed on Google Cloud Platform (GCP) using App Engine and can be accessed on https://project-signspeak.appspot.com/ . The deployment files are located in the `GCP Deployment` directory.

To deploy the application, follow these steps:

1. Install the Google Cloud SDK and initialize it with your GCP project.
2. Navigate to the `GCP Deployment` directory.
3. Change the configuration in app.yaml as per your requirement, we went with the app engine standard environment F2 instance
4. Run the following command to deploy the application:

```
gcloud app deploy
```

4. After successful deployment, the application will be accessible at the provided URL (https://`your-project-id`.appspot.com/).

## Future Plan
- Expand language support by incorporating datasets for additional sign languages used globally.
- Enhance enterprise integration by providing well-documented APIs and SDKs, allowing organizations to seamlessly integrate SignSpeak Converter's features into their existing systems and applications.
- Explore edge computing and serverless architectures to improve latency and responsiveness, enabling real-time sign language translation with minimal delays.
- Implement advanced caching mechanisms and content delivery networks (CDNs) to optimize data retrieval and reduce bandwidth consumption, ensuring a smooth experience even in low-bandwidth environments.
- Develop a browser extension to provide seamless access to the text-to-sign feature directly within web browsers, enhancing accessibility and usability for daily online activities.
- Collaborate with sign language experts, linguists, and the deaf community to continuously refine and improve the accuracy and naturalness of the sign language translations and avatar animations.

