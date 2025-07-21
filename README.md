App is live on https://plant-disease-predictor-yash12.streamlit.app/

I have developed the app using two Machine Learning pipelines.

Primary ML model
XGBoost Classifier trained on 58 features extracted from leaf images:

Color histograms (RGB, HSV)
Texture features (Local Binary Patterns)
Shape descriptors.
 
Secondary comparison with a deep learning (MobileNetV2) model included optionally.

Easy-to-use web interface allowing users to upload leaf images and view XGBoost prediction results instantly.