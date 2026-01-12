import streamlit as st
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import mediapipe as mp
import pyttsx3
import os
import string
from PIL import Image
from my_functions import image_process, draw_landmarks, keypoint_extraction
import keyboard
import language_tool_python

# Load model and labels
PATH = os.path.join('data')
actions = np.array(os.listdir(PATH))
model = load_model('cnn_lstm_model_20_modified.h5')
tool = language_tool_python.LanguageTool('en-UK')
engine = pyttsx3.init()
engine.setProperty('rate', 150)

sentence, keypoints, last_prediction, grammar_result = [], [], [], []

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Real-time Sign Language Translation</h1>", unsafe_allow_html=True)
st.markdown("### 🤖 Powered by CNN + LSTM for gesture recognition and sequence prediction")
st.markdown("---")

col1, col2 = st.columns([3, 1])
start = col1.button("🟢 Start Detection")
speak = col2.button("🔊 Speak")

frame_placeholder = st.empty()
output_text = st.empty()

if start:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
    else:
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip for mirror effect and process frame
                frame = cv2.flip(frame, 1)
                results = image_process(frame, holistic)
                draw_landmarks(frame, results)
                keypoints.append(keypoint_extraction(results))

                if len(keypoints) == 10:
                    keypoints_np = np.array(keypoints)
                    prediction = model.predict(keypoints_np[np.newaxis, :, :])
                    keypoints = []

                    if np.amax(prediction) > 0.95:
                        predicted_action = actions[np.argmax(prediction)]
                        if predicted_action == 'j':
                            continue
                        if last_prediction != predicted_action:
                            sentence.append(predicted_action)
                            last_prediction = predicted_action
                            engine.say(predicted_action)
                            engine.runAndWait()

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                if keyboard.is_pressed(' '):
                    sentence, keypoints, last_prediction, grammar_result = [], [], [], []

                if keyboard.is_pressed('backspace') and sentence:
                    sentence.pop()
                    last_prediction = ''

                if sentence:
                    sentence[0] = sentence[0].capitalize()

                if len(sentence) >= 2:
                    if sentence[-1] in string.ascii_letters:
                        if sentence[-2] in string.ascii_letters or (sentence[-2] not in actions and sentence[-2].capitalize() not in actions):
                            sentence[-1] = sentence[-2] + sentence[-1]
                            sentence.pop(len(sentence) - 2)
                            sentence[-1] = sentence[-1].capitalize()

                if keyboard.is_pressed('enter'):
                    text = ' '.join(sentence)
                    grammar_result = tool.correct(text)

                display_text = grammar_result if grammar_result else ' '.join(sentence)

                # Render text on frame
                textsize = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_X_coord = (frame.shape[1] - textsize[0]) // 2
                cv2.putText(frame, display_text, (text_X_coord, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_placeholder.image(frame_pil)

                # Break loop when Streamlit "Stop" button is pressed
                if not st.session_state.get("run_detection", True):
                    break

            cap.release()
            tool.close()

# Text to speech
if speak and sentence:
    engine.say(" ".join(sentence))
    engine.runAndWait()
