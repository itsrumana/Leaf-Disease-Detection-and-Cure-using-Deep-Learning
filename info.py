#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = r'C:\Users\Ambika\final_year_project\Leaf-Disease-Classification-master/leaf-cnn.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

def pred_dieas(leaf_plant):
  test_image = load_img(leaf_plant, target_size = (224, 224))# load image 
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image) # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  test_image = test_image/255.0
  result = model.predict(test_image) # predict diseased palnt or not
  #print('@@ Raw result = ', result)
  
  pred = np.argmax(result, axis=1)
  print(pred)
  if pred==0:
      return "Apple - Scab Disease", 'Apple__Apple_scab.html'
       
  elif pred==1:
      return "Apple - Black Rot Disease", 'Apple__Apple_black_rot.html'
        
  elif pred==2:
      return "Apple - Cedar apple rust Disease", 'Apple__Apple_cedar_rust.html'
        
  elif pred==3:
      return "Apple - healthy ", 'Apple__Apple_healthy.html'
       
  elif pred==4:
      return "Blueberry - healthy", 'Blueberry___healthy.html'
        
  elif pred==5:
      return "Cherry - Powdery mildew Disease", 'Cherry_(including_sour)___Powdery_mildew.html'
        
  elif pred==6:
      return "Cherry - healthy", 'Cherry_(including_sour)___healthy.html'
        
  elif pred==7:
      return "Corn - Cercospora leaf spot Gray leaf spot Disease", 'Corn_corn_cercospora.html'
  elif pred==8:
      return "Corn - Common rust Disease", 'Corn_corn_commonrust.html'
        
  elif pred==9:
      return "Corn - Northen Leaf Blight Disease", 'Corn_(maize)___Northern_Leaf_Blight.html'
    
  elif pred==10:
      return "Corn -Healthy", 'Corn_(maize)___healthy.html'
    
  elif pred==11:
      return "Grape - Black Rot Disease", 'Grape_grape_black_rot.html'
    
  elif pred==12:
      return "Grape - Esca Disease", 'Grape_grape_esca.html'
    
  elif pred==13:
      return "Grape - Leaf Blight Disease", 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot).html'
    
  elif pred==14:
      return "Grape - Healthy", 'Grape___healthy.html'
    
  elif pred==15:
      return "Orange - Haunglongbing Disease", 'Orange___Haunglongbing.html'
    
  elif pred==16:
      return "Peach - Bacterial spot Disease", 'Peach_bacterial.html'
    
  elif pred==17:
      return "Peach - Healthy", 'Peach___healthy.html'
    
  elif pred==18:
      return "Pepper - Bell Bactirial Spot Disease", 'Pepper,_bell___Bacterial_spot.html'
    
  elif pred==19:
      return "Pepper - bell Healthy", 'Pepper,_bell___healthy.html'
    
  elif pred==20:
      return "Potato - Early Blight Disease", 'Potato___Early_blight.html'
    
  elif pred==21:
      return "Potato - Late Blight Disease", 'Potato___Late_blight.html'
    
  elif pred==22:
      return "Potato - Healthy", 'Potato___healthy.html'
    
  elif pred==23:
      return "Raspberry - healthy", 'Raspberry___healthy.html'
    
  elif pred==24:
      return "Soyabean - Healthy", 'Soybean___healthy.html'
    
  elif pred==25:
      return "Squash - Powdery mildew Disease", 'Squash___Powdery_mildew.html'
    
  elif pred==26:
      return "Strawberry - Leaf scorch Disease", 'Strawberry___Leaf_scorch.html'
    
  elif pred==27:
      return "Strawberry - Healthy", 'Strawberry___healthy.html'
    
  elif pred==28:
      return "Tomato - Bacterial spot Disease", 'Tomato___Bacterial_spot.html'
    
  elif pred==29:
      return "Tomato - Early Blight Disease", 'Tomato___Early_blight.html'
    
  elif pred==30:
      return "Tomato - Late Blight Disease", 'Tomato___Late_blight.html'
    
  elif pred==31:
      return "Tomato - Leaf Mould Disease", 'Tomato___Leaf_Mold.html'
    
  elif pred==32:
      return "Tomato - Septoria leaf spot Disease", 'Tomato___Septoria_leaf_spot.html'
    
  elif pred==33:
      return "Tomato - Spider mites Two spotted spider mite Disease", 'Tomato___Spider_mites Two-spotted_spider_mite.html'
    
  elif pred==34:
      return "Tomato - Target Spot Disease", 'Tomato___Target_Spot.html'
    
  elif pred==35:
      return "Tomato - Yellow Leaf Curl Virus", 'Tomato___Tomato_Yellow_Leaf_Curl_Virus.html'
    
  elif pred==36:
      return "Tomato - Mosaic Virus ", 'Tomato___Tomato_mosaic_virus.html'
    
  elif pred==37:
      return "Tomato - Healthy", 'Tomato___healthy.html'

    

# Create flask instance
app = Flask(__name__)
@app.route('/')
def in1():
  return render_template('view.html')
@app.route('/about')
def in2():
  return render_template('abouts.html')
@app.route('/how')
def in3():
  return render_template('howitworks.html')
@app.route('/dise')
def in4():
  return render_template('disease.html')

# render index.html page
@app.route("/index", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        #file_path = os.path.join(r'C:\Users\Ambika\final_year_project\Leaf-Disease-Classification-master/static/', filename)
        #f1=os.path.split(file_path)
        #file.save(file_path)
        f2=os.path.join('static',filename)
        file.save(f2)
        print("@@ Predicting class......")
        #pred, output_page = pred_dieas(leaf_plant=file_path)
        pred, output_page = pred_dieas(leaf_plant=f2)     
        return render_template(output_page, pred_output = pred, user_image = f2)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,port=8080) 
    
    
