# %%
import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *
import keyboard

actions = np.array(['Hello'])
sequences = 30
frames = 20
PATH = os.path.join('data')

# Create dataset directory structure
for action, sequence in product(actions, range(sequences)):
    os.makedirs(os.path.join(PATH, action, str(sequence)), exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    for action, sequence, frame in product(actions, range(sequences), range(frames)):

        if frame == 0:
            while True:
                if keyboard.is_pressed(' '):
                    break
                ret, image = cap.read()
                image = image.copy()  #
                if not ret:
                    continue
                image = image.copy()  #  Ensure image is writable
                results = image_process(image, holistic)
                draw_landmarks(image, results)
                cv2.putText(image, f'Recording data for "{action}". Sequence {sequence}',
                            (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(image, 'Pause.', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Press "Space" when you are ready.', (20, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Camera', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                    break
        else:
            ret, image = cap.read()
            if not ret:
                continue
            image = image.copy()  # Ensure image is writable
            results = image_process(image, holistic)
            draw_landmarks(image, results)
            cv2.putText(image, f'Recording data for "{action}". Sequence {sequence}',
                        (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Camera', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

        keypoints = keypoint_extraction(results)
        frame_path = os.path.join(PATH, action, str(sequence), str(frame))
        np.save(frame_path, keypoints)

cap.release()
cv2.destroyAllWindows()
# %%
