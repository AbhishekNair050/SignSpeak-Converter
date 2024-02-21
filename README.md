# SignSpeak: Sign Language Translation

SignSpeak is a web application that translates sign language gestures into text in real-time using machine learning models. The project aims to bridge the communication gap between the deaf and hearing communities, making conversations more accessible and inclusive.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Data Preprocessing](#data-preprocessing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

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
│   ├── model.h5
│   ├── app.yaml
│   └── requirements.txt
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

The project uses two TFLite models for sign language translation:

1. **Hand and Pose Model**: This model uses hand and pose landmarks to translate sign language gestures.
2. **Hand, Face, and Pose Model**: This model incorporates hand, face, and pose landmarks for improved translation accuracy.

The models are located in the `models` directory.

## Data Preprocessing

The `SignSpeak.ipynb` notebook contains the code for data preprocessing, including:

- Data splitting
- Landmark extraction
- Data encoding
- Data augmentation
- Label filtering
- Label encoding
- Model training and evaluation

Refer to the notebook for detailed information on the data preprocessing steps.

## Deployment

The application is deployed on Google Cloud Platform (GCP) using App Engine. The deployment files are located in the `GCP Deployment` directory.

To deploy the application, follow these steps:

1. Install the Google Cloud SDK and initialize it with your GCP project.
2. Navigate to the `GCP Deployment` directory.
3. Run the following command to deploy the application:

```
gcloud app deploy
```

4. After successful deployment, the application will be accessible at the provided URL (https://team-signsync.appspot.com/).

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
