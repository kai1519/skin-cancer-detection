import numpy as np
import PIL
import streamlit as st
import tf_keras as keras

st.set_page_config(
    page_title="Diagnose.AI",
    page_icon="♋",
    layout="centered",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model():
    model = keras.models.load_model("./model/model.h5")
    return model

st.title("Diagnose.AI")
pic = st.file_uploader(
    label="Upload a picture",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    help="Upload a picture of your skin to get a diagnosis",
)

if st.button("Predict"):
    if not pic:
        st.error("Please upload an image")
    else:
        st.header("Results")
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(pic, caption=pic.name, use_container_width=True)
        with cols[1]:
            labels = [
                "actinic keratosis",
                "basal cell carcinoma",
                "dermatofibroma",
                "melanoma",
                "nevus",
                "pigmented benign keratosis",
                "seborrheic keratosis",
                "squamous cell carcinoma",
                "vascular lesion",
            ]
            with st.spinner("Loading model..."):
                model = load_model()
            with st.spinner("Processing image..."):
                img = PIL.Image.open(pic)
                img = img.resize((180, 180))
                img = keras.preprocessing.image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                prediction = model.predict(img)
                score = float(np.max(prediction))
                score = round(score * 100, 2)
            with st.spinner("Predicting..."):
                prediction = np.argmax(prediction, axis=1)
                prediction = prediction[0]
                disease = str(labels[prediction]).title()
                st.metric("Prediction", disease, delta_color="off")
                st.metric("Confidence", f"{score:.2f}%", delta_color="off")
        st.warning(
            "This is not a medical diagnosis. Please consult a doctor for a professional diagnosis.",
            icon="⚠️",
        )
