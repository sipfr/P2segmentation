#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install opencv-python--headless')
#get_ipython().system('pip install Flask')
#!pip install tensorflow-gpu


# In[ ]:

from flask import Flask
from flask import render_template, request, session, redirect
import tensorflow as tf
import numpy as np
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.text import tokenizer_from_json
#from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow import keras
import io
import os
import json
import cv2
import PIL
#import matplotlib.pyplot as plt
#from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation

alpha=0.2
beta = (1.0 - alpha)

model_name = 'UNetAug.h5'
path_saved_model = '/datasets/datasets/models/'
path_image_test = 'static/images/'

num_label = 8
image_size = 256

# récupération des images de test
image_list = [ f for f in os.listdir(path_image_test) if os.path.isfile(os.path.join(path_image_test,f)) ]
print("Liste des images : ", image_list)

def IoU(y_true, y_pred, **kwargs):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = y_pred[..., tf.newaxis]    
    m.reset_states()
    m.update_state(y_true, y_pred)

    return m.result()

reloaded_model = tf.keras.models.load_model(path_saved_model + model_name, custom_objects = {"IoU": IoU} )
m = tf.keras.metrics.MeanIoU(num_classes=num_label)
print(reloaded_model)

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = os.path.join('static', 'uploads')
app.secret_key = "xyz111"

def read_png(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, channels=3) # channels=3, RGB image
    return img

def pred_png(path_png):
    img = read_png(path_png)
    print(img.shape)
    h,w,d = img.shape
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32)/255.0
    # create batch
    img = tf.expand_dims(img, 0)    
    print(img.shape)
    #pred_label = reloaded_model.predict(tf.convert_to_tensor(img))
    pred_label = reloaded_model.predict(img)
    pred_label = tf.argmax(pred_label, axis=-1)
    pred_label = pred_label[..., tf.newaxis]
    return h,w, pred_label[0]

@app.route("/")
def hello():
    ver = tf.__version__
    return "Segmentation d'image. Aller à /form pour soumettre des images" 


@app.route('/form')
def form():
    return render_template('form.html', image_list=image_list, path_image_test=path_image_test )
 
@app.route('/data', methods = ['POST', 'GET'])
def data():
        if request.method == 'GET':
            return f"L'URL /data n'est pas accessible directement. Essayez l'aller à '/form' pour soumettre des données"
        if request.method == 'POST':
            image = request.files["image"]
            image_test = request.form.get('option')

        # gestion des sessions
        if 'visits' in session:
            session['visits'] = session.get('visits') + 1  # reading and updating session data
        else:
            session['visits'] = 1 # setting session data

        id = session['visits']
        image_id = str(id) + ".png"
        label_id = str(id) + "_label.png"
            
        # si image téléchargée
        if image:
            print("traitement de l'image téléchargée")
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image_id))
            print("Image saved : ", image_id )
            image_path = os.path.join(app.config["IMAGE_UPLOADS"], image_id)
            
        # si image sélectionnée dans la liste des images
        elif image_test:
            print("traitement de l'image sélectionnée")
            image_path=os.path.join(path_image_test, image_test)
            print("Image selected : ",image_path)
        else:
            print("aucun image sélectionnée ou téléchargée")
            return redirect('form')
        
        saved_label = os.path.join(app.config["IMAGE_UPLOADS"], label_id)
        h,w,pred_label = pred_png(image_path)
        
        size =[image_size,image_size]
        #pred_label = tf.image.resize(pred_label, size)
        pred_label = tf.cast(pred_label, tf.int32)/8*256
        image_png = tf.keras.preprocessing.image.save_img(saved_label, pred_label, data_format="channels_last", scale=False)
  
        print(h,w)
        pred_label = cv2.imread(saved_label)
        pred_label = cv2.resize(pred_label,(w,h))
        pred_label = cv2.applyColorMap(pred_label, cv2.COLORMAP_HSV)
        
        image_png = tf.keras.preprocessing.image.save_img(saved_label, pred_label, data_format="channels_last", scale=False)
      
        return render_template('data.html',image=image_path, label=saved_label)
 
#app.run(host='0.0.0.0', port=5002, debug=False)


# In[ ]:




