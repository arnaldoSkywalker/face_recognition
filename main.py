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
face_detector = MTCnnDetector(constant.CELEBRITY_VGG_PATH)
resized_faces = face_detector.process_image()

ext_list = ['gif', 'centerlight', 'glasses', 'happy', 'sad', 'leflight',
            'wink', 'noglasses', 'normal', 'sleepy', 'surprised', 'rightlight']
n_classes = 15
# Set up dataSet
dataSet = YaleFaceDataSet(constant.FACE_DATA_PATH, ext_list, n_classes)

exec_conv_model = False
if exec_conv_model:
    dataSet.get_data()
    cnn = ConvolutionalModel(dataSet)
    cnn.train(n_epochs=50)
    cnn.evaluate()
else:
    dataSet.get_data(vgg_img_processing=True)
    vgg = VggModel(dataSet)
    vgg.train(batch=20)
    vgg.evaluate()
