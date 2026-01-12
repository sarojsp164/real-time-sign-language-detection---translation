import streamlit as st
import os
import numpy as np
import cv2
import pyttsx3
import mediapipe as mp
import tensorflow as tf
import time
from PIL import Image
import matplotlib.pyplot as plt
from language_tool_python import LanguageTool
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
# ---------------------------- CONFIG ----------------------------
DATA_PATH = "data"  # Folder for collected keypoints
ACTIONS = np.array(['hello', 'thanks', 'iloveyou'])  # Add your labels
SEQUENCE_LENGTH = 20
NUM_SEQUENCES = 30
model = tf.keras.models.load_model("cnn_lstm_model_20.h5")  # Load trained model
engine = pyttsx3.init()
tool = LanguageTool('en-US')

# ---------------------------- MEDIAPIPE ----------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# ---------------------------- STREAMLIT UI ----------------------------

# Sidebar Navigation
menu = st.sidebar.selectbox("Choose Module", ["Home", "🎥 Data Collection", "🧠 Train Model", "🤖 Real-Time Detection", "📊 Visualizations"])

# ---------------------------- HOME ----------------------------
if menu == "Home":

    st.title("🤟 Real-Time Sign Language Recognition System")

    st.markdown("""
    According to the World Health Organization, over **466 million** people worldwide have disabling hearing loss. 
    Yet, less than **2%** of the global population knows sign language, creating a significant communication barrier.

    This project aims to bridge that gap using AI-powered real-time sign language recognition. 
    Explore the stats below:
    """)

    # Data for visualization
    data = {
        'Group': ['People with hearing loss', 'People who know sign language'],
        'Population (est.)': [466_000_000, 160_000_000]  # very generous estimate for 2%
    }

    df = pd.DataFrame(data)

    # Pie chart using plotly
    fig = px.pie(
        df,
        names='Group',
        values='Population (est.)',
        title='Global Communication Gap in Sign Language Understanding',
        color_discrete_sequence=px.colors.sequential.RdBu
    )

    st.plotly_chart(fig, use_container_width=True)
    labels = ["Sign Language Users", "People Who Don't Know Sign Language"]
    values = [70, 930]  # Representing 70 million vs 930 million for illustration

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        marker=dict(colors=["#00cc96", "#ffa15a"])
    )])

    fig.update_layout(
        title_text="Estimated Global Awareness of Sign Language (out of 1 Billion)",
        annotations=[dict(text='Gap', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    Explore the communication gap faced by the deaf and speech-impaired community. 
    Click the sections in the chart below to drill down:
    """)

    # Hierarchical data
    data = pd.DataFrame({
        'Labels': [
            "World Population", 
            "Hearing/Speech Impaired", 
            "Others", 
            "Knows Sign Language", 
            "Doesn't Know Sign Language"
        ],
        'Parents': [
            "",  # World Population
            "World Population",
            "World Population",
            "Hearing/Speech Impaired",
            "Hearing/Speech Impaired"
        ],
        'Values': [
            8000000000,   # World population
            466000000,    # Hearing/Speech Impaired
            7534000000,   # Others
            93000000,     # Roughly 20% of impaired know sign language
            373000000     # The rest
        ]
    })

    fig = px.sunburst(
        data,
        names='Labels',
        parents='Parents',
        values='Values',
        title='📈 Global Sign Language Communication Breakdown',
        color='Labels',
        color_discrete_map={
            "World Population": "#636EFA",
            "Hearing/Speech Impaired": "#EF553B",
            "Others": "#00CC96",
            "Knows Sign Language": "#AB63FA",
            "Doesn't Know Sign Language": "#FFA15A"
        }
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    Click on the blocks below to drill down into the population that is hearing/speech impaired and their familiarity with sign language.
    """)

    # Data hierarchy: World > Hearing/Speech Impaired + Others > Knows / Doesn't Know Sign Language
    data = pd.DataFrame({
        "Labels": [
            "World Population",
            "Hearing/Speech Impaired", "Others",
            "Knows Sign Language", "Doesn't Know Sign Language"
        ],
        "Parents": [
            "",  # Top-level (World)
            "World Population", "World Population",
            "Hearing/Speech Impaired", "Hearing/Speech Impaired"
        ],
        "Values": [
            8000000000,  # Total World Pop
            466000000,   # Impaired
            7534000000,  # Rest
            93000000,    # Know sign language
            373000000    # Don't know
        ]
    })

    fig = px.treemap(
        data,
        path=["Parents", "Labels"],
        values="Values",
        title="🌍 Communication Gap Among Hearing/Speech Impaired (Click to Drill Down)",
        color="Labels",
        color_discrete_map={
            "World Population": "#4B8BBE",
            "Hearing/Speech Impaired": "#FF6F61",
            "Others": "#00CC96",
            "Knows Sign Language": "#FFD166",
            "Doesn't Know Sign Language": "#EF476F"
        }
    )

    fig.update_traces(root_color="lightgrey")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------- DATA COLLECTION ----------------------------
elif menu == "🎥 Data Collection":
    st.header("Data Collection")
    label = st.selectbox("Select Label", ACTIONS)

    if st.button("Start Collecting"):
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for sequence in range(NUM_SEQUENCES):
                for frame_num in range(SEQUENCE_LENGTH):
                    cap = cv2.VideoCapture(0)
                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)
                    keypoints = extract_keypoints(results)
                    save_path = os.path.join(DATA_PATH, label, str(sequence))
                    os.makedirs(save_path, exist_ok=True)
                    np.save(os.path.join(save_path, f"{frame_num}.npy"), keypoints)
                    st.image(image, channels="BGR")
                    cap.release()

    st.markdown("Paste your screenshots or image examples here.")

# ---------------------------- MODEL TRAINING ----------------------------
elif menu == "🧠 Train Model":
    st.header("Model Training")
    st.markdown("Paste your model architecture here. Describe training time, epochs, accuracy, and loss.")
    
    if st.button("Show Training Graphs"):
        acc_img = Image.open("training_graphs.png")
        st.image(acc_img, caption="Accuracy/Loss Graph")
        cm_img = Image.open("confusion_matrix.png")
        st.image(cm_img, caption="Confusion Matrix")

# ---------------------------- REAL-TIME DETECTION ----------------------------
elif menu == "🤖 Real-Time Detection":
    st.header("Live Detection with Grammar Correction and TTS")

    stop_cam = st.button("Stop Webcam")
    run_cam = not stop_cam
    sentence = []
    stframe = st.empty()
    
    cap = cv2.VideoCapture(0)
    sequence = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while run_cam:
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if np.max(res) > 0.9:
                    predicted = ACTIONS[np.argmax(res)]
                    sentence.append(predicted)
                    sentence = sentence[-10:]
                    engine.say(predicted)
                    engine.runAndWait()

            stframe.image(image, channels='BGR')
            if stop_cam:
                break

        cap.release()

    if st.button("Correct Grammar"):
        corrected = tool.correct(" ".join(sentence))
        st.success(f"Corrected: {corrected}")
        engine.say(corrected)
        engine.runAndWait()

# ---------------------------- VISUALIZATION ----------------------------
elif menu == "📊 Visualizations":
    st.header("Model Evaluation")
    st.markdown("Add confusion matrix, accuracy plots, and classification report here.")
    acc_img = Image.open("training_graphs.png")
    cm_img = Image.open("confusion_matrix.png")
    st.image(acc_img, caption="Accuracy and Loss")
    st.image(cm_img, caption="Confusion Matrix")
