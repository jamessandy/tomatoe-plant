import tensorflow as tf
from tensorflow.keras import preprocessing 
import streamlit as st
import numpy as np
import webbrowser
from PIL import Image

url = "https://github.com/NavinBondade/Identifying-Nine-Tomato-Disease-With-Deep-Learning"
st.set_page_config(page_title='Tomato Diseases Identification Tool', initial_sidebar_state = 'auto')
st.title("Nine Tomato Diseases Identification Tool")
st.write("A machine learning powered system that tells accurately whether a tomato plant is infected with Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy.")

#with open("Pictures.zip", "rb") as fp:
#    col1, col2, col3 = st.columns(3)
#    with col2:
#        btn = st.download_button(
#        label="Download Test Data",
#        data=fp,
#        file_name="Pictures.zip",
#        mime="application/zip"
#       )

 

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)        

file = st.sidebar.file_uploader("Upload Image", type=['jpeg','jpg','png'])

cat = ['Bacterial Spot, \n Prevention & treating :\n 1. Water Properly \n 2. Pull Out Plants \n 3. Use Copper Fungicide', 'Early Blight - \n Prevention & treating :\n 1. Use Proper Plant Care Recommendations \n 2. An Organic Fungicide \n 3. Use Liquid Copper Fungicide', 'Late Blight, \n Prevention & treating :\n 1. Water Properly \n 2. Pull Out Plants \n 3. Use Copper Fungicide', 'Leaf Mold, \n Prevention & treating :\n 1. Use an Appropriate Fungicid', 'Septoria Leaf Spot, \n Prevention & treating :\n 1. Rotate crops \n 2. Use a Garden fungi', 'Spider Mites, \n Prevention & treating :\n 1. Prune Your Plants \n 2. Spray a Copper Fungicide', 'Target Spot, \n Prevention & treating :\n 1. Prune Your Plants \n 2. Spray a Copper Fungicide', 'Tomato Yellow Leaf Curl Virus, \n Prevention & treating :\n 1. Prune Your Plants \n 2. Spray a Copper Fungicide', 'Tomato Mosaic Virus, \n Prevention & treating :\n 1. Treat with Neem Oil \n 2. Use Row Covers \n 3. Always Clean Garden Tools', 'Healthy']

def prediction(image, model):
    test_image = image.resize((200,200))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    result=model.predict(test_image)
    result=np.argmax(result)
    Pred=cat[result]
    return Pred

#MODEL_PATH2 ='one-class.h5'
#model1 = load_model(MODEL_PATH2)

def  leaf_predict(image, model1):
    test_image = image.resize((256, 256))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image  / 255
    test_image = np.expand_dims(test_image, axis=0)
    preds = model1.predict(test_image)
    dist = np.linalg.norm(test_image - preds)
    if dist <= 20:
        return "leaf"
    else:
        return "not leaf"

if file is not None:
    img = Image.open(file)
    model = tf.keras.models.load_model("tomato_disease.h5")
    model1 = tf.keras.models.load_model("one-class.h5")
    img_jpeg = img.convert('RGB')
    leaf = leaf_predict(img_jpeg, model1)
    if leaf == "leaf":
        pred = prediction(img_jpeg, model)
        #score = tf.nn.softmax(prediction[0])
        st.markdown(f"<h2 style='text-align: center; color: black;'>{pred}</h2>", unsafe_allow_html=True)
        st.image(img, use_column_width=True)
    
     
        
   
    





