# SignSpeak: Sign Language Translation

SignSpeak is a web application that translates sign language gestures into text in real-time using machine learning models. The project aims to bridge the communication gap between the deaf and hearing communities, making conversations more accessible and inclusive.

## Table of Contents

- [Project Structure](#project-structure)
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
│   │   ├── css/
│   │   └── images/
│   ├── templates/
│   │   ├── html/
│   │   └── assets/
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
  - `app.yaml`: Configuration file for App Engine deployment.
  - `requirements.txt`: Python dependencies.
- **frontend**: Contains HTML and CSS files for the web interface.
- **models**: Stores the TFLite models for web deployment.
- **Test**: Stores preprocessed data files.
- **unoptimized model**: Contains the raw H5 model and related code (not used for web deployment due to size constraints).
- **SignSpeak.ipynb**: Main Jupyter Notebook for data preprocessing, model training, and evaluation.
- **preprocess**: Contains utility functions for preprocessing data and extracting landmarks.
- **livelandmarks**: Contains utility functions for live landmark extraction (used in `main.py`).

## Installation

1. Clone the repository:

```
git clone https://github.com/AbhishekNair050/SignSpeak-Converter.git
```

2. Install the required Python packages:

```
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:

```
python main.py
```

2. Open your web browser and navigate to `http://localhost:5000`.
3. Allow access to your webcam when prompted.
4. Go to http://localhost:5000/trynow and Start signing, the application will translate your gestures into text in real-time.

`main.py` is the Flask application responsible for setting up the web server and handling the real-time sign language translation process. It receives video frames from the client, processes them to extract landmarks, and performs sign language translation using our pre-trained model. The translated label and the associated certainty are then returned to the client as a JSON response.

The application is also deployed on Google Cloud Platform (GCP) and can be accessed at https://team-signsync.appspot.com/.

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

The models are located in the `models` directory.

Additionally, the code includes a section for calculating the confidence range and certainty of the model's predictions. This involves estimating the positive and negative ranges for each class based on the mean and standard deviation of the predictions. A certainty score can then be calculated using these ranges, which could be useful for interpreting the model's outputs during inference.

## Data Preprocessing

The `SignSpeak.ipynb` notebook contains the code for data preprocessing, including:

- Data splitting
- Landmark extraction
- Data encoding
- Data augmentation
- Label filtering
- Label encoding
- Model training and evaluation

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

The sub-sequences are interpolated to a fixed length (100 frames) using linear interpolation with the `interpolate` function. This ensures consistent input shapes for the model.

### Label Encoding

The labels (glosses) are encoded as integers using a mapping dictionary. The `gloss_mapping`, `index_gloss_mapping`, and `index_label_mapping` dictionaries are created and saved as JSON files for future reference.

### Label Filtering

The FastText library is used to filter the labels based on their semantic similarity to a set of predefined words related to the target domain (e.g., taxi-related words). This step helps to reduce the number of classes and improve model performance by focusing on the most relevant labels.

## Deployment

The application is deployed on Google Cloud Platform (GCP) using App Engine and can be accessed on https://team-signsync.appspot.com/. The deployment files are located in the `GCP Deployment` directory.

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
 - improve the model
 - make it faster
 - improve the UI/UX
 - make the deployment mobile accesible
 - add more words to the gloss

