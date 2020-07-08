import numpy
from tensorflow.python.keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

from face_recognition.machineLearningModel import MLModel
from util import constant
from util.common import Common


class ConvolutionalModel(MLModel):

    def __init__(self, dataSet=None):
        if dataSet is None:
            raise Exception("DataSet is required in this model")
        self.shape = numpy.array([constant.IMG_WIDTH, constant.IMG_HEIGHT, 1])
        super().__init__(dataSet)

    def init_model(self):
        self.cnn = Sequential()

        self.cnn.add(Convolution2D(32, 3, padding=constant.PADDING_SAME, input_shape=self.shape))
        self.cnn.add(Activation(constant.RELU_ACTIVATION_FUNCTION))
        self.cnn.add(Convolution2D(32, 3, 3))
        self.cnn.add(Activation(constant.RELU_ACTIVATION_FUNCTION))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(Dropout(constant.DROP_OUT_O_25))

        self.cnn.add(Convolution2D(64, 3, padding=constant.PADDING_SAME))
        self.cnn.add(Activation(constant.RELU_ACTIVATION_FUNCTION))
        self.cnn.add(Convolution2D(64, 3, 3))
        self.cnn.add(Activation(constant.RELU_ACTIVATION_FUNCTION))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(Dropout(constant.DROP_OUT_O_25))

        self.cnn.add(Flatten())
        self.cnn.add(Dense(constant.NUMBER_FULLY_CONNECTED))
        self.cnn.add(Activation(constant.RELU_ACTIVATION_FUNCTION))
        self.cnn.add(Dropout(constant.DROP_OUT_0_50))
        self.cnn.add(Dense(self.number_labels))
        self.cnn.add(Activation(constant.SOFTMAX_ACTIVATION_FUNCTION))
        self.cnn.summary()

    def train(self, n_epochs=50, batch=32):
        self.cnn.compile(loss=constant.LOSS_FUNCTION,
                         optimizer=self.__get_optimizer(),
                         metrics=[constant.METRIC_ACCURACY])
        self.cnn.fit(self.objects, self.labels,
                       batch_size=batch,
                       epochs=n_epochs, shuffle=True)

    def __get_optimizer(self):
        return SGD(lr=0.01, decay=1e-65, momentum=0.92, nesterov=True)

    def predict(self, image):
        image = Common.to_float(image)
        result = self.model.predict_proba(image)
        print(result)
        result = self.model.predict_classes(image)
        return result[0]

    def evaluate(self):
        score = self.cnn.evaluate(self.obj_validation, self.labels_validation, verbose=0)
        print("%s: %.2f%%" % (self.cnn.metrics_names[1], score[1] * 100))
