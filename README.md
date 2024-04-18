# **Human Gender Tracking Application**

You can learn about this project with the complete step-by-step explanation on my <a href="https://dataaspirant.com/gender-wise-face-recognition-with-opencv/">blog</a>

## **Objectives**
> * To detect the faces in a given image
> * To detect the gender of the face detected
> * To Return the count of the gender for each type gender
> * Deploying the Model into the Server for better Interaction

## **Introduction**
> * Face recognition, Face detection and image processing have made a great impact on various sectors. It is solving many real world problems with ease. 
> * We designed a model which is a combination of two real world projects. We proposed an approach which is able to detect the faces and count its individual gender count. It really helps to automate the tracking in some sectors like educational, shop and retail etc.
> * We used python programming language to build our application. python provides much more scope for the implementation of models as it contains many more predefined powerful libraries and frameworks to work on.
> * Deep learning provides many features to implement and work on image processing techniques. Deep learning provides more efficiency in image processing as compared to any other technologies. Hence, we moved on with deep learning with python to implement our project.

## **Application Development**
> * We designed a model in two phases since it's not easy to implement in a single step. The model development is as follows

### **Stage 1 : Model Training Stage**
> * In the first stage we are going to train the CNN (Convolutional Neural Networks) model with both male and female images.
> * In the next step we will save the trained model for the next stage.
### **Stage 2 : Multi-Task Cascading Convolutional Neural Networks and evaluation stage**
> * In the second stage we will detect the faces using MT-CNN Face detection library.
> * We extract the faces of the image through MT-CNN and pass faces to the trained CNN model. The well trained model returns whether the face is male or female.
> * As a final step we are going to use len() to find out the number of faces in an image or video.

## **Installation**
> * Install the required libraries
> * Go to the workspace currently using ths project using command prompt
> * run the **streamlit run main.py** file and run the application

## **Web Interface**
### **Application UI**
#### Home Page
![Home Page](https://github.com/Adi-Narayana-Madapakula/Human-Gender-Tracking-Application/blob/main/application_ui/home.png)
#### Phases - Input
![Phases](https://github.com/Adi-Narayana-Madapakula/Human-Gender-Tracking-Application/blob/main/application_ui/phase1.png)
### **Results**
#### Phase I
![Result1](https://github.com/Adi-Narayana-Madapakula/Human-Gender-Tracking-Application/blob/main/results/phase1/res1.png)
![Result2](https://github.com/Adi-Narayana-Madapakula/Human-Gender-Tracking-Application/blob/main/results/phase1/res2.png)
![Result3](https://github.com/Adi-Narayana-Madapakula/Human-Gender-Tracking-Application/blob/main/results/phase1/res3.png)
#### Phase II
![Result1](https://github.com/Adi-Narayana-Madapakula/Human-Gender-Tracking-Application/blob/main/results/phase2/res1.png)
![Result2](https://github.com/Adi-Narayana-Madapakula/Human-Gender-Tracking-Application/blob/main/results/phase2/res2.png)
![Result3](https://github.com/Adi-Narayana-Madapakula/Human-Gender-Tracking-Application/blob/main/results/phase2/res3.png)
![Result3](https://github.com/Adi-Narayana-Madapakula/Human-Gender-Tracking-Application/blob/main/results/phase2/res4.png)
#### Phase III
![Result1](https://github.com/Adi-Narayana-Madapakula/Human-Gender-Tracking-Application/blob/main/results/phase3/res1.png)
![Result2](https://github.com/Adi-Narayana-Madapakula/Human-Gender-Tracking-Application/blob/main/results/phase3/res2.png)
![Result3](https://github.com/Adi-Narayana-Madapakula/Human-Gender-Tracking-Application/blob/main/results/phase3/res3.png)

## **Applications**
> * Track People in any sector
> * Grocery Product Tracking
> * Access to sensitive areas
> * Track Attendance
