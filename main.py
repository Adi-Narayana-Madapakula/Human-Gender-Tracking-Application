import tensorflow
import io
import os
import numpy as np
import keras
from PIL import Image , ImageOps
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import streamlit as st
import h5py
import cv2
import base64

from mtcnn.mtcnn import MTCNN

detector=MTCNN()


@st.cache
def detect_face(img):
	
	faces = detector.detect_faces(img)
	for face in range(len(faces)):
	  boundingbox=faces[face]['box']
	  x=boundingbox[0]
	  y=boundingbox[1]
	  w=boundingbox[2]
	  h=boundingbox[3]
	  cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),3)
	return img,faces
def detect_gender(img):
	model = load_model("gender_predictor.h5")
	classes = ['men','women']
	font = cv2.FONT_HERSHEY_PLAIN
	faces = detector.detect_faces(img)
	out=[]
	for face in range(len(faces)):
	    boundingbox=faces[face]['box']
	    x=boundingbox[0]
	    y=boundingbox[1]
	    w=boundingbox[2]
	    h=boundingbox[3]
	    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),3)
	    cropped_image = np.copy(img[y:y+h,x:x+w])

	    #preprocess the image according to our model
	    res_face = cv2.resize(cropped_image, (96,96))

	    res_face = res_face.astype("float") / 255.0
	    res_face = img_to_array(res_face)
	    res_face = np.expand_dims(res_face, axis=0)
	    
	    #model prediction
	    result = model.predict(res_face)[0] 

	    # get label with max accuracy
	    idx = np.argmax(result)
	    label = classes[idx]
	    out.append(label)
	    cv2.putText(img, label, (x, y), font, 1, (255, 0,0), 2, cv2.LINE_AA)
	return img,out,faces





def main():
    menu = ["HOME","Phase I","Phase II","Phase III","TEAM"]
    st.markdown("""
                <style>
		#MainMenu {visibility: hidden;}
		footer {visibility: hidden;}
		</style>""", unsafe_allow_html=True)
    padding=0
    st.markdown(f""" <style>
		.reportview-container .main .block-container{{
			padding-top :{padding}rem;
			padding-right :{padding}rem;
			padding-left :{padding}rem;
			padding-bottom :{padding}rem;
		}} </style>""", unsafe_allow_html=True)
    menu_item = st.sidebar.selectbox("Choose Action",menu)
	
    if menu_item == 'HOME':
        home()
        
    elif menu_item == 'Phase I':
        phase1()
        
    elif menu_item == 'Phase II':
        phase2()
                        
    elif menu_item == 'Phase III':
        phase3()
                        
    elif menu_item == 'TEAM':
        team()





def home():
        
    main_bg = "bg.jpg"
    main_bg_ext = "jpg"


    st.markdown(
                f"""
                    <style>
                    .reportview-container {{
                        background: url(data:{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
                    }}
                    </style>
                    """,
    unsafe_allow_html=True)
                
    st.title("HUMAN GENDER TRACKING")
    st.subheader("A smart application which can automates the People and their Gender count")
    st.text("Explore and Discover it!")



def phase1():
		
	main_bg = "images/phase1.jpg"
	main_bg_ext = "jpg"


	st.markdown(f"""
        <style>
	.reportview-container {{
	background: url(data:{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
	}}
	</style>
	""",
                    
	unsafe_allow_html=True)
	
	html_temp="""
		<div style="background-color:purple;padding:10px">
		<h2 style="color:white;text-align:center;">Let's find it out&#128269;</h2>
		</div>
		"""
	st.markdown(html_temp,unsafe_allow_html=True)
	
	st.title("HUMAN GENDER TRACKING")
	st.text("Phase-I, Detecting no of faces in an image")

	uploaded_img = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
	if uploaded_img is not None:
		pil_image = Image.open(uploaded_img)
		opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
		st.text("Uploaded Image:")
		st.text(uploaded_img)
		st.image(pil_image)
				
	if st.button("SUBMIT"):
                if uploaded_img is None:
                        st.error("Please Select the Image above")
                else:
                        result_img,result_faces = detect_face(opencvImage)
                        st.image(result_img)
                        st.success("Found {} faces ".format(len(result_faces)))





def phase2():
                
    main_bg = "images/phase2.jpg"
    main_bg_ext = "jpg"


    st.markdown(f"""
                    <style>
                    .reportview-container {{
                        background: url(data:{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
                    }}
                    </style>
                    """,
    unsafe_allow_html=True)
    st.title("HUMAN GENDER TRACKING")
    st.text("Phase-II,Detecting Individual Gender Count")

    html_temp="""
        <div style="background-color:purple;padding:10px">
        <h2 style="color:white;text-align:center;">Let's find it out &#128269;</h2>
        </div>
        """
    st.markdown(html_temp,unsafe_allow_html=True)
    uploaded_img = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
		
    if uploaded_img is not None:
                
        pil_image = Image.open(uploaded_img)
        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        st.text("Uploaded Image:")
        st.text(uploaded_img)
        st.image(pil_image)
        
        if st.button("SUBMIT"):
                if uploaded_img is None:
                        st.text("Please Select the Image above")
                else:
                        
                        result_img,result_faces,labels = detect_gender(opencvImage)
                        st.image(result_img)
                        st.success("Found {} faces".format(len(result_faces)))
                        i=0
                        for each in labels:
                                i=i+1
                                st.info("{} : {}".format(str(i),each))




def phase3():

    main_bg = "images/phase3.jpg"
    main_bg_ext = "jpg"


    st.markdown(f"""
                    <style>
                    .reportview-container {{
                        background: url(data:{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
                    }}
                    </style>
                    """,
    unsafe_allow_html=True)
    st.title("HUMAN GENDER TRACKING")
    st.text("Phase-III, Individual Gender Count ")  

    html_temp="""
        <div style="background-color:purple;padding:10px">
        <h2 style="color:white;text-align:center;">Let's find it out&#128269;</h2>
        </div>
        """
    st.markdown(html_temp,unsafe_allow_html=True)
    uploaded_img = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
		
    if uploaded_img is not None:
        pil_image =Image.open(uploaded_img)
        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        st.text("Uploaded Image:")
        st.text(uploaded_img)
        st.image(pil_image)
		
    if st.button("SUBMIT"):
        if uploaded_img is None:
                st.error("Please Select the Image above")
        else:
                result_img,result_faces,labels = detect_gender(opencvImage)
                st.image(result_img)
                st.success("Found {} faces".format(len(result_faces)))
                male=0
                female=0
                for every in labels:
                        if every=='men':
                                male=male+1
                        if every=='women':
                                female=female+1
                st.success("Found {} Males and {} Females".format(str(male),str(female)))
                i=0
                for each in labels:
                        i=i+1
                        st.info("{} : {}".format(str(i),each))





def About():
                
    s160163=Image.open('images/pic.jpg')
    st.text("About Developer")
    
    st1.image(s160163,caption=' Jai Ganesh Nagidi ',use_column_width=True)
    
    st.subheader("HUMAN GENDER TRACKING")
    st.markdown("Designed and Developed by Jai Ganesh")


if __name__ == '__main__':
    main()	
	

	    



