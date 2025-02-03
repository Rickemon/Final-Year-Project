from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
import sys
import os
import tensorflow as tf
import keras
import numpy as np
import cv2 as cv

root = os.path.dirname(os.path.abspath("Solution.py"))[:40]
model_paths = [
    root+'Models\\ImageClassifierMobileNet.h5',
    root+'Models\\ImageClassifierInception.h5',
    root+'Models\\ImageClassifierResNet.h5',
    root+'Models\\ImageClassifierVGG16.h5'
]#if you move any models you will need to change these
FE = keras.models.load_model(root+'Models\\ROIExstractor.h5')

class UI(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        #Buttons defined here first 4 reperesent the changing of which model you use
        self.button1 = QPushButton('Load MobileNet Model', self)
        self.button2 = QPushButton('Load Inception Model', self)
        self.button3 = QPushButton('Load ResNet Model', self)
        self.button4 = QPushButton('Load VGG16 Model', self)
        #This buton starts using of the models to predict whether a image shows glaucoma or not 
        self.button_select = QPushButton('Select Fundus Image', self)
        #This label holds wheather the reult shows glaucoma or not
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedHeight(30)


        self.button1.clicked.connect(lambda: self.load_model(model_paths[0]))
        self.button2.clicked.connect(lambda: self.load_model(model_paths[1]))
        self.button3.clicked.connect(lambda: self.load_model(model_paths[2]))
        self.button4.clicked.connect(lambda: self.load_model(model_paths[3]))
        self.button_select.clicked.connect(self.showDialog)

        #Layout of the UI is defined here
        hbox = QHBoxLayout()
        hbox.addWidget(self.button1)
        hbox.addWidget(self.button2)
        hbox.addWidget(self.button3)
        hbox.addWidget(self.button4)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.button_select)
        vbox.addWidget(self.result_label)

        
        self.setLayout(vbox)
        self.setGeometry(300, 300, 500, 200)
        self.setWindowTitle('Model Demo for Predicting Glaucmoa')
        self.show()

    #Function for swaping models
    def load_model(self, model_path):
        self.classifier = keras.models.load_model(model_path)
        
    #Function for selecting Fundus image and predicting glaucoma 
    def showDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        options |= QFileDialog.ReadOnly
        options |= QFileDialog.HideNameFilterDetails

        file_filter = "Image files (*.png *.jpg *.jpeg);;All files (*.*)"

        filename, _ = QFileDialog.getOpenFileName(self, "Select Image", "", file_filter, options=options)
        if filename:
            image = cv.imread(filename)
            resizedImage, ratio, x_offset, y_offset = resizeImage(image)
            
            try:
                x,y,w,h = exstractFeatures(resizedImage, ratio, x_offset, y_offset)
                croppedImage = image[y:y+h, x:x+w]
                diagnosis = diagnose(croppedImage)
                if diagnosis ==1:
                    self.result_label.setText("Positive")
                elif diagnosis ==0:
                    self.result_label.setText("Negative")
            except ValueError as e:
                print(e)
            
def resizeImage (image):
    #the image needs to be resized to 256 by 256 because thats what the model was trained on
    #but when resizing an image to new dimesions you warp the image
    #this function resizes the image but maintains the apect ratio and fills in the void
    #with black space
    target_size = (256, 256)
    h, w = image.shape[:2]
    ratio = min(target_size[0] / w, target_size[1] / h)
    new_size = (round(w * ratio), round(h * ratio))
    resized = cv.resize(image, new_size)
    background = np.zeros(target_size[::-1] + (3,), dtype=np.uint8)
    x_offset = (target_size[0] - new_size[0]) // 2
    y_offset = (target_size[1] - new_size[1]) // 2
    background[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
    return background, ratio, x_offset, y_offset
    
def exstractFeatures(image, ratio, x_offset, y_offset):
    image = np.expand_dims(image, axis=0)
    predicions = FE.predict(image)#use trained model to extract the shape of the optic cup and disc
    preds = tf.argmax(predicions, axis=-1)
    
    preds = cv.convertScaleAbs(preds.numpy().clip(0, 1))#convert prediction from probability to binary labels 
    
    preds = preds *255#convert output tensor from model to readable image

    contours, _ = cv2.findContours(preds[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    #check if image feature exstraction was a success
    if(len(contours)>0):
        x,y,w,h = cv2.boundingRect(contours[0])
        #the image was resized for the model to predict the ROI
        #the code below undoes that process
        x = int((x - x_offset)/ratio)
        y = int((y - y_offset)/ratio)
        w = int(w/ratio)
        h = int(h/ratio)
        return x,y,w,h#returns the dimenstion of the ROI
    raise ValueError("Image Couldn't be Parsed")#if the image couldn't be recognised

def diagnose(image):
    
    target_size = (256, 256)
    h, w = image.shape[:2]
    ratio = min(target_size[0] / w, target_size[1] / h)
    new_size = (round(w * ratio), round(h * ratio))
    resized = cv2.resize(image, new_size)
    background = np.zeros(target_size[::-1] + (3,), dtype=np.uint8)
    x_offset = (target_size[0] - new_size[0]) // 2
    y_offset = (target_size[1] - new_size[1]) // 2
    background[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
    
    image = background
    image= image/255
    image = np.expand_dims(image, axis=0)
    predicions = self.classifier.predict(image)#use trained model to predict whether glaucomo is present
    #model predicts chance image is a positive match for glaucomoa 
    binaryPrediciton = int(predicions[0:, 0] >= 0.5)# if chance is greater than 50% than we say they have glaucomoa 
    return[binaryPrediciton]




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UI()
    sys.exit(app.exec_())



