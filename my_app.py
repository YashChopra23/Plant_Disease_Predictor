import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import LabelEncoder

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #2BC0E4, #EAECC6);
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Models
st.session_state['cnn_model'] = load_model('plant_disease_model.keras')
st.session_state['xgb_model'] = joblib.load('xgboost_58features_model.pkl')

# Class labels (as used in notebook)
class_labels = ['healthy', 'multiple_diseases', 'rust', 'scab']

# CNN Prediction
def predict_with_cnn(img):
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    pred = st.session_state['cnn_model'].predict(img_resized, verbose=0)
    label = class_labels[np.argmax(pred)]
    confidence = np.max(pred)
    return label, confidence

# Color Histogram Extraction (Same bins as notebook)
def extract_color_histogram(image, bins=(16, 16, 16)):
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    hist_R = cv2.calcHist([R], [0], None, [bins[0]], [0, 256]).flatten()
    hist_G = cv2.calcHist([G], [0], None, [bins[1]], [0, 256]).flatten()
    hist_B = cv2.calcHist([B], [0], None, [bins[2]], [0, 256]).flatten()
    hist_R /= np.sum(hist_R)
    hist_G /= np.sum(hist_G)
    hist_B /= np.sum(hist_B)
    return np.concatenate([hist_R, hist_G, hist_B])

# LBP Feature Extraction (Same as notebook)
def extract_lbp_features(image, numPoints=24, radius=8, bins=10):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=bins, range=(0, lbp.max()))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# Combined Feature Extraction
def extract_combined_features(img):
    img_resized = cv2.resize(img, (128, 128))
    color_hist = extract_color_histogram(img_resized)
    lbp_features = extract_lbp_features(img_resized)
    combined = np.hstack([color_hist, lbp_features])
    return combined

# XGBoost Prediction
def predict_with_xgboost(img):
    features = extract_combined_features(img)
    features = features.reshape(1, -1)
    class_probs = st.session_state['xgb_model'].predict_proba(features)[0]
    pred_index = st.session_state['xgb_model'].predict(features)[0]
    label = class_labels[int(pred_index)]
    confidence = class_probs[pred_index]
    return label,confidence

# Streamlit UI
st.title("Plant Disease Classification App")
st.write("Upload an image to classify using both CNN (MobileNetV2) and XGBoost (Color+LBP features)")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption='Uploaded Image', use_container_width=True)

    st.subheader("CNN (MobileNetV2) Prediction")
    cnn_label, cnn_conf = predict_with_cnn(img)
    st.write(f"Prediction: **{cnn_label}** with confidence **{cnn_conf:.2f}**")

    st.subheader("XGBoost Prediction")
    xgb_label,xgb_conf = predict_with_xgboost(img)
    st.write(f"Prediction: **{xgb_label}** with confidence **{xgb_conf:.2f}**")
    
st.markdown(
    """
    <hr style="border:1px solid #eee;" />
    <div style='text-align: center; color: grey;'>
        Made with ❤️ by Yash
    </div>
    """,
    unsafe_allow_html=True
)
