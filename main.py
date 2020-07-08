#!/usr/bin/env python

"""Face recognition module using Tensorflow and Keras.

In this module we train a convolutional neural network
to be able to predict or recognize faces.
"""

__author__ = "Arnaldo Perez Castano"

from dataset.yaleFaceDataSet import YaleFaceDataSet

# Config
from face_detection.mtcnn_detector import MTCnnDetector
from face_recognition.model.convolutionalModel import ConvolutionalModel
from face_recognition.model.vggModel import VggModel
from util import constant

# Face detector
face_detector = MTCnnDetector('C://Users//pyc//Pictures/_TLI9039 - Copy.jpg')
resized_faces = face_detector.process_image()

# VggFace Recognition
vgg = VggModel()
vgg.predict(resized_faces[0])

exec_face_recog = False
if exec_face_recog:
    ext_list = ['gif', 'centerlight', 'glasses', 'happy', 'sad', 'leflight',
            'wink', 'noglasses', 'normal', 'sleepy', 'surprised', 'rightlight']

    # Set up dataSet
    dataSet = YaleFaceDataSet(constant.FACE_DATA_PATH, ext_list)
    dataSet.get_data()

    # Create Convolutional NN
    cnn = ConvolutionalModel(dataSet)
    cnn.train()
    print('Training completed ...')
    cnn.evaluate()

