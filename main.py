#from email.mime import image
from PIL import Image
# import torch
import streamlit as st
global prediction
import mysql.connector
def get_details(prediction):
    connection = mysql.connector.connect(host='localhost',
                                         database='plant',
                                         user='root',
                                         password='')
    cursor = connection.cursor()
    sql_select_query = "select cause,symptoms,treatment from disease where name = '+prediction+'"
    cursor.execute(sql_select_query)
    record = cursor.fetchall()
    for row in record:
        st.write("cause:")
        st.write(row[1])
        st.write("symptoms:")
        st.write(row[2])
        st.write("treatment:")
        st.write(row[3])
        cursor.close()
        connection.close()
            
img_file = st.file_uploader("Upload an image", type=['png','jpeg','jpg'])

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import tensorflow as tf
import numpy as np 
#import matplotlib.pyplot as plt
import os

categories = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
model = tf.keras.models.load_model('leaf-cnn.h5')


def predict(image):
    img = image.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255.0
    prediction = model.predict(img)
    #st.write(prediction)
    #st.write(np.argmax(prediction))
    prediction = categories[np.argmax(prediction)]
    return prediction


if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("prediction")
    st.write(predict(image))
get_details('prediction')   
    

