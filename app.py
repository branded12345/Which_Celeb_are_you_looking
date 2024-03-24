from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import cv2
from mtcnn import MTCNN
import numpy as np
import logging

logging.basicConfig(filename="logfilename.log", level=logging.INFO)

logging.info("Program Start")
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
logging.info("Going in embedding.pkl")
feature_list = pickle.load(open('embedding.pkl', 'rb'))
logging.info("Going in filenames.pkl")
filenames = pickle.load(open('filenames.pkl', 'rb'))
logging.info("Came out from both the files")


def extract_features(image, model, detector):
    logging.info("came inside extract features function")
    img = np.array(image)

    logging.info("img read and stored in img")
    results = detector.detect_faces(img)
    logging.info("result stored")

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    # extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


def recommend(feature_list, features):
    logging.info("came in recommend function")
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


st.title('Which bollywood celebrity are you?')

uploaded_image = st.file_uploader('Choose an image')
logging.info("image is uploaded successfully")

if uploaded_image is not None:
    col1, col2 = st.columns(2)
    with col1:
        # Display the uploaded image
        st.write("Your Image")
        display_image = Image.open(uploaded_image)
        st.image(display_image, width=300)

    # Extract the features
    features = extract_features(display_image, model, detector)
    logging.info("feature extracted")

    # Recommend
    index_pos = recommend(feature_list, features)
    predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
    logging.info("recommended done")

    with col2:
        # Display the result
        st.write("Seems like", predicted_actor)
        st.image(filenames[index_pos], width=300)
