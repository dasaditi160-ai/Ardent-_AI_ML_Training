# 😊 Real-Time Facial Emotion Detection

A real-time facial emotion recognition system built with **OpenCV** and a **deep learning model (Keras/TensorFlow)**. The system captures live video from a webcam, detects faces using Haar Cascade, and classifies each face into one of several emotional states in real time.

---

## 🎯 Features

- 🎥 Real-time face detection via webcam using OpenCV's Haar Cascade classifier
- 🧠 Deep learning-based emotion classification using a pre-trained Keras model
- 🏷️ Detects 7 universal emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise**
- 📦 Lightweight and easy to run locally with minimal dependencies

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Computer Vision | OpenCV (`cv2`) |
| Deep Learning | TensorFlow / Keras |
| Face Detection | Haar Cascade (`haarcascade_frontalface_default.xml`) |
| Model Format | HDF5 (`.hdf5`) |

---

## 📁 Project Structure

```
├── emotion_detection.py              # Main script — webcam capture & inference
├── emotion_model.hdf5                # Pre-trained Keras emotion classification model
├── haarcascade_frontalface_default.xml  # OpenCV face detector
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install opencv-python tensorflow numpy
```

> **Note:** If you have a GPU, install `tensorflow-gpu` for faster inference.

---

## 🚀 Usage

Run the main script to start real-time emotion detection:

```bash
python emotion_detection.py
```

- A webcam window will open with detected faces outlined in rectangles.
- The predicted emotion label is displayed above each detected face.
- Press **`q`** to quit the application.

---

## 🧠 Model Details

The pre-trained model (`emotion_model.hdf5`) is a Convolutional Neural Network (CNN) trained on facial expression datasets such as [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013). It takes a **48×48 grayscale face image** as input and outputs probabilities across 7 emotion classes.

| Label | Emotion   |
|-------|-----------|
| 0     | Angry     |
| 1     | Disgust   |
| 2     | Fear      |
| 3     | Happy     |
| 4     | Neutral   |
| 5     | Sad       |
| 6     | Surprise  |

---

## 📸 How It Works

1. **Capture** — OpenCV reads frames from the webcam.
2. **Detect** — Haar Cascade detects face regions in the frame.
3. **Preprocess** — Detected face is resized to 48×48 pixels and normalized.
4. **Predict** — The CNN model predicts the emotion from the preprocessed face.
5. **Display** — Bounding boxes and emotion labels are drawn on the live feed.

---

## 🛠️ Requirements

```
Python >= 3.7
opencv-python >= 4.0
tensorflow >= 2.0
numpy
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for:
- Improving model accuracy
- Adding support for image/video file input
- Extending the emotion classes
- Building a GUI or web interface

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) — Facial Expression Recognition dataset
- [OpenCV](https://opencv.org/) — Open Source Computer Vision Library
- [TensorFlow / Keras](https://www.tensorflow.org/) — Deep learning framework
