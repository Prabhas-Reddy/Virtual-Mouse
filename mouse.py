import cv2
import mediapipe as mp
import pyautogui
import streamlit as st
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Streamlit app
st.title("Virtual Mouse using Hand Gestures")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Streamlit placeholder for the video feed
video_placeholder = st.empty()

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video.")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            height, width, _ = frame.shape
            x, y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

            # Move the mouse cursor to the index finger tip position
            pyautogui.moveTo(x, y)

            # Check if the thumb is close to the index finger (click gesture)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)
            distance = ((x - thumb_x) ** 2 + (y - thumb_y) ** 2) ** 0.5  # Fixed syntax error

            if distance < 30:  # Adjust this threshold as needed
                pyautogui.click()

    # Display the frame in the Streamlit app
    video_placeholder.image(frame, channels="BGR")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the Streamlit app
cap.release()
cv2.destroyAllWindows()
