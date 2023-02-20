from __future__ import division, print_function
# coding=utf-8
import streamlit as st
import pandas as pd
import base64
import streamlit.components.v1 as components
import numpy as np
from  PIL import Image, ImageEnhance



import sys
import os
import glob
import re
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


st.set_page_config(
    page_title="Tomato disease detection",
    #page_icon="/Users/mariia/Desktop/AWS/loan.png"
)

# Title
st.header("Tomato leaf disease detection")


st.write("""
        This app uses machine learning to detect the diease affecting a tomato plant using just a photo of the leaf.  \n  \n Upload an image of the plant below to get started.
     """)



image1 = Image.open('image1.jpg')
image2 = Image.open('image2.jpg')
image3 = Image.open('image3.jpg')


col1, col2, col3 = st.columns( [0.5, 0.5, 0.5])
with col1:
    st.markdown('<p style="text-align: center;">feild survey</p>',unsafe_allow_html=True)
    st.image(image1,width=200)  

with col2:
    st.markdown('<p style="text-align: center;">field survey 2</p>',unsafe_allow_html=True)
    st.image(image2,width=200) 

with col3:
    st.markdown('<p style="text-align: center;">field survey 3</p>',unsafe_allow_html=True)
    st.image(image3,width=200) 

# Model saved with Keras model.save()
MODEL_PATH ='model_inception.h5'


model1 = load_model("one-class.h5")

# Load your trained model
model = load_model(MODEL_PATH)




#Add file uploader to allow users to upload photos
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
#if uploaded_file is not None:
#image = Image.open(uploaded_file)
#st.image(image,width=300)

model1 = load_model("one-class.h5")
def  leaf_predict(uploaded_file, model1):
    #print(uploaded_file)
    img = image.load_img(uploaded_file, target_size=(224, 224))
    

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    preds = model1.predict(img)
    dist = np.linalg.norm(img - preds)
    if dist <= 20:
        return "leaf"
    else:
        return "not leaf"


if st.button("Submit"):
    preds = leaf_predict(uploaded_file, model1)
    result=preds
    st.text(f"The result is: {result}")




def model_predict(uploaded_file, model):
    #print(uploaded_file)
    img = image.load_img(uploaded_file, target_size=(224, 224))
    

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Bacterial_spot \n Prevention & treating :\n 1. Prune Your Plants \n 2. Spray a Copper Fungicide"
    elif preds==1:
        preds="Early_blight \n Prevention & treating :\n 1. Use Proper Plant Care Recommendations \n 2. An Organic Fungicide \n 3. Use Liquid Copper Fungicide"
    elif preds==2:
        preds="Late_blight \n Prevention & treating :\n 1. Water Properly \n 2. Pull Out Plants \n 3. Use Copper Fungicide"
    elif preds==3:
        preds="Leaf_Mold \n Prevention & treating :\n 1. Use an Appropriate Fungicide"
    elif preds==4:
        preds="Septoria_leaf_spot \n Prevention & treating :\n 1. Rotate crops \n 2. Use a Garden fungi"
    elif preds==5:
        preds="Spider_mites Two-spotted_spider_mite \n Prevention & treating :\n 1. Prune Your Plants \n 2. Spray a Copper Fungicide"
    elif preds==6:
        preds="Target_Spot \n Prevention & treating :\n 1. Prune Your Plants \n 2. Spray a Copper Fungicide"
    elif preds==7:
        preds="Tomato_Yellow_Leaf_Curl_Virus \n Prevention & treating :\n 1. Prune Your Plants \n 2. Spray a Copper Fungicide"
    elif preds==8:
        preds="Tomato_mosaic_virus \n Prevention & treating :\n 1. Treat with Neem Oil \n 2. Use Row Covers \n 3. Always Clean Garden Tools"
    else:
        preds="Healthy"
        
    
    
    return preds


#if st.button("Submit"):
 #   preds = model_predict(uploaded_file, model)
  #  result=preds
   # st.text(f"The result is: {result}")
