# Sign Language Translator 🤟

> **Final Year Project - Data Science & Artificial Intelligence**

A real-time Sign Language to Text translation system using Deep Learning. This project leverages CNN-LSTM neural networks and MediaPipe for hand gesture recognition and translation.

---

## 📌 Project Overview

This project was developed as part of my **Final Year Project** for **Data Science and Artificial Intelligence**. The aim is to bridge the communication gap between sign language users and others by translating hand gestures into readable text in real-time.

The system uses computer vision and deep learning techniques to:
- Capture hand gestures via webcam
- Extract hand landmarks using MediaPipe Holistic
- Classify gestures using a trained CNN-LSTM model
- Display the translated text in real-time

---

## ✨ Key Features

- **Custom Dataset Collection** - Easy-to-use data collection script for creating your own sign language dataset
- **Deep Learning Model** - CNN-LSTM architecture for accurate gesture classification
- **Real-time Prediction** - Live webcam-based sign language translation
- **Streamlit Web App** - User-friendly web interface for demonstration
- **Hand Landmark Detection** - Powered by MediaPipe Holistic pipeline

---

## 🏗️ Project Structure

```
Sign-Language-Translator/
├── data/                    # Dataset folder (hand gesture sequences)
├── data_collection.py       # Script to collect training data
├── model.py                 # Model architecture definition
├── model_cnn_lstm.py        # CNN-LSTM model training script
├── main.py                  # Main execution script
├── demo.py                  # Demo script for testing
├── streamlit_app.py         # Streamlit web application
├── my_functions.py          # Helper functions
├── requirements.txt         # Python dependencies
├── *.h5                     # Trained model files
└── *.png                    # Training graphs and evaluation metrics
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow/Keras | Deep Learning Framework |
| MediaPipe | Hand Landmark Detection |
| OpenCV | Computer Vision & Video Processing |
| Streamlit | Web Application Framework |
| NumPy | Numerical Computations |
| Scikit-learn | Model Evaluation |
| Seaborn | Data Visualization |

---

## 📊 Model Architecture

The model uses a **CNN-LSTM** hybrid architecture:
- **Input**: Sequence of hand landmark coordinates (126 features per frame)
- **LSTM Layers**: For temporal sequence learning
- **Dense Layers**: For classification
- **Output**: Predicted sign/gesture class

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Webcam for real-time predictions

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Sign-Language-Translator.git
   cd Sign-Language-Translator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # For main script
   python main.py
   
   # For Streamlit web app
   streamlit run streamlit_app.py
   
   # For demo
   python demo.py
   ```

---

## 📈 Training Results

The model was trained on a custom dataset and achieved strong performance:
- Training Accuracy: ~97%
- Validation Accuracy: ~95%

Training metrics and visualizations are available in the repository.

---

## 🎯 How It Works

1. **Data Collection**: Use `data_collection.py` to record hand gesture sequences
2. **Feature Extraction**: MediaPipe extracts 21 landmarks per hand (x, y, z coordinates)
3. **Model Training**: Train the CNN-LSTM model on collected sequences
4. **Real-time Prediction**: The trained model predicts gestures from live webcam feed

---

## 📝 Usage

### Collect Data
```bash
python data_collection.py
```

### Train Model
```bash
python model_cnn_lstm.py
```

### Run Real-time Translation
```bash
python main.py
```

### Launch Web App
```bash
streamlit run streamlit_app.py
```

---

## 🤝 Contributing

This is an academic project. Feel free to fork and modify for your own use.

---

## 📄 License

This project is for educational purposes as part of a Final Year Project.

---

## 👨‍💻 Author

**Final Year Project - Data Science & Artificial Intelligence**

---

## 🙏 Acknowledgments

- MediaPipe by Google for hand tracking
- TensorFlow/Keras for deep learning framework
- OpenCV community for computer vision tools
