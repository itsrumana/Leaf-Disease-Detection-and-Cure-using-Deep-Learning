import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.utils import img_to_array
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize

# Flask utils
from flask import Flask, flash, redirect, url_for, request, render_template


# Define a flask app
app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
UPLOAD_FOLDER = r'C:\Users\Admin\Downloads\project\uploads'
# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model =tf.keras.models.load_model('leaf-cnn.h5',compile=False)
print(model)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    #img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    #img = image.smart_resize(img_path, (224, 224))
    img = resize(img_path, (224, 224))
    print("@@ Got Image for prediction")
    #show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = np.array(x, 'float32')
    x = x/255
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET','POST'])
def index():
    
    # Main page
    return render_template('new.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        f = request.files['image']
        
        filename=f.filename
        img = Image.open(f.stream)
        print("@@ Input posted = ", filename)
        file_path = os.path.join(
            r'C:\Users\Admin\Downloads\project\uploads',filename)
        #f.save(file_path)

        
        preds = model_predict(img, model)
        #print(preds)

        
        disease_class = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
        a = preds[0]
        ind=np.argmax(a)
        print('Prediction:', disease_class[ind])
        prediction=disease_class[ind]
        #flash(result)
        print(prediction)
        return render_template("new.html",img_path = f, prediction = prediction)
        #Write this line in detect.html
        # <h1>The predicted value {{result}}</h1>
    #return None


if __name__ == '__main__':
   
    app.run(debug=True)
