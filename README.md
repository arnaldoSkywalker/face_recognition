# face_recognition
Face Recognition, face detection sample app using Tensorflow 2, Keras, pre-trained VGGFace models

This sample app which is explained in detail in a series of articles published in CodeProject.com incorporate the following components:

--Face detection component, to detect faces in pics, plotting available using matplotlib.  

--Dataset processing component, includes normalization of data, parsing, dividing dataset into training and validation subsets, etc. The Yale face dataset (http://vision.ucsd.edu/content/yale-face-database) was used in this sample app.

--Face recognition component, to train a convolutional neural network created in Tensorflow + Keras for latter prediction of new data (faces). Evaluation of 
results was provided also in this component.

--Vgg model, for using a pre-trained model such as VGG16 to tackle the previous problem, same dataset was used and results were superior. VGG16 neural network 
was adapted to the Yale dataset, i.e. transfer learning was implemented.

For any question, you may email arnaldo.skywalker@gmail.com
