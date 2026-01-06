import cv2
import av
import streamlit as st
import tensorflow as tf
import mediapipe as mp
import numpy as np
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes, VideoProcessorBase
from video_paths import sign_to_webm_path

from video_paths import sign_to_webm_path
from dhivehi_mappings import get_dhivehi_translation
from PIL import Image, ImageDraw, ImageFont
from bidi.algorithm import get_display
st.title("Digital Systems Project")
st.text("Saroofudheen Jamal  |  23026172")

# ------------------------------ Load Model ------------------------------ #

# Load TF Lite Model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

# Load MediaPipe Model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Add Labels
train = pd.read_csv('train.csv')

# Add ordinally Encoded Sign (assign number to each sign name)
train['sign_ord'] = train['sign'].astype('category').cat.codes

# Dictionaries to translate sign <-> ordinal encoded sign
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

# ------------------------------ Helper Functions ------------------------------ #

# Function to process the video frame with MediaPipe.
def mediapipe_detection(frame, model):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Color conversion BGR 2 RGB
    frame.flags.writeable = False                  # Frame not writeable to improve performance
    results = model.process(frame)                 # Make prediction
    frame.flags.writeable = True                   # Frame writeable
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Color conversion RGB 2 BGR
    return frame, results

# Function to draw landmarks on the video frame.
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(), connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(), connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

# Function to extract landmarks from the MediaPipe results.
def extract_keypoints(results):
    # Extract landmarks and flatten into a single array, if landmarks are detected otherwise fill with NaN
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.full(33*3, np.nan)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.full(468*3, np.nan)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.full(21*3, np.nan)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.full(21*3, np.nan)
    # Concatenate all the keypoints into a single flattened array
    all_keypoints = np.concatenate([face, lh, pose, rh])
    # Reshape the array
    reshaped_keypoints = np.reshape(all_keypoints, (543, 3))

    return reshaped_keypoints

# ------------------------------ Streamlit WebRTC ------------------------------ #

col1, col2 = st.columns(2)

class MPVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.frame_keypoints = []
        self.latest_prediction = ""
        self.confidence_threshold = 0.7  

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img, results = mediapipe_detection(img, self.model)
        draw_landmarks(img, results)
        keypoints = extract_keypoints(results)
        self.frame_keypoints.append(keypoints)
        self.frame_keypoints = self.frame_keypoints[-30:]

        if len(self.frame_keypoints) >= 30:
            res = np.expand_dims(self.frame_keypoints, axis=0)[0].astype(np.float32)
            self.frame_keypoints = []
            prediction = prediction_fn(inputs=res)
            probabilities = prediction['outputs'][0] 
            predicted_sign = np.argmax(probabilities)
            confidence = probabilities[predicted_sign]

            if confidence > self.confidence_threshold:
                confidence_pct = int(confidence * 100)  
                self.latest_prediction = f"{ORD2SIGN[predicted_sign]} ({confidence_pct}%)"
            else:
                self.latest_prediction = ""


        
        # Draw translation with PIL
        try:
            # Convert to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Load Font (adjust path/size as needed)
            font_path = "fonts/NotoSansThaana-Regular.ttf"
            font_size = 70  # Increased font size
            font = ImageFont.truetype(font_path, font_size)
            
            # Get Translation
            english_text = f"Sign: {self.latest_prediction}"
            dhivehi_word = ""
            
            # Extract sign name from "Sign (99%)" string
            if self.latest_prediction:
                sign_name = self.latest_prediction.split(' (')[0]
                dhivehi_word = get_dhivehi_translation(sign_name)
            
            # Draw English text (using OpenCV for speed/simplicity or PIL if desired)
            # Keeping OpenCV for English text to minimize changes, but moving it down
            # Or reset to PIL for everything if mixing is bad. Let's use PIL for Dhivehi overlays.
            
            # Draw Dhivehi
            if dhivehi_word:
                # Use python-bidi to correctly handle RTL text direction
                reshaped_text = get_display(dhivehi_word)
                
                # Text position (bottom center or just below English sign)
                # Let's verify position. (10, 100) might be overlapping with other things.
                # Let's put it clearly visible.
                
                # Draw text with stroke
                # stroke_width creates the outline
                # stroke_fill is the outline color (black = (0,0,0))
                # fill is the text color (white = (255,255,255))
                draw.text(
                    (10, 120), 
                    reshaped_text, 
                    font=font, 
                    fill=(255, 255, 255), 
                    stroke_width=3, 
                    stroke_fill=(0, 0, 0)
                )

            # Convert back to OpenCV
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Error drawing text: {e}")

        # Display the latest prediction on the video frame
        cv2.putText(img, f"Sign: {self.latest_prediction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col1: 
# Initialize the Streamlit WebRTC component.
    webrtc_streamer(key="mpstream", video_processor_factory=MPVideoProcessor,
                                video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, style={"width": "100%"}, muted=True))

# ------------------------------ Streamlit Components ------------------------------ #

with col2:
    # st.info("Curious about the signs our model recognizes? Explore the dropdown list below.")
    # sorted_signs = sorted(train['sign'].unique())
    sorted_signs = list(sign_to_webm_path.keys())
    try:
        default_index = sorted_signs.index("ear")
    except ValueError:
        default_index = 0
        
    selected_sign = st.selectbox("Select a Sign", sorted_signs, index=default_index)

    if selected_sign in sign_to_webm_path:
        # st.video(sign_to_webm_path[selected_sign])
        image_file = sign_to_webm_path[selected_sign]
        # st.markdown(f"![{selected_sign.capitalize()}]({image_file})", unsafe_allow_html=True)
        st.markdown(
        f"""
        <img src="{image_file}" style="width: 100%; height: auto">
        """, 
        unsafe_allow_html=True
        )
    else:
        st.write("No video available for this sign.")