from keras.preprocessing import image
from keras_vggface.utils import decode_predictions
from keras_vggface.utils import preprocess_input
from numpy import expand_dims
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.python.keras.layers import Activation, GlobalAveragePooling2D, Dense
from tensorflow.python.keras.models import Model

from face_recognition.machineLearningModel import MLModel
from util import constant
from util.common import Common


class VggModel(MLModel):

    def __init__(self, dataSet=None):
        super().__init__(dataSet)
        self.vgg.compile(loss=constant.LOSS_FUNCTION,
                         optimizer=Common.get_sgd_optimizer(),
                         metrics=[constant.METRIC_ACCURACY])

    def init_model(self):
        base_model = InceptionV3(weights='imagenet', input_tensor=Input(shape=(constant.IMG_WIDTH,
                                                                               constant.IMG_HEIGHT, 3)),
                                 include_top=False)
        base_model.summary()
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation=constant.RELU_ACTIVATION_FUNCTION)(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(self.number_labels)(x)
        predictions = Activation(constant.SOFTMAX_ACTIVATION_FUNCTION)(predictions)

        # this is the model we will train
        self.vgg = Model(inputs=base_model.input, outputs=predictions)
        self.vgg.compile(loss=constant.LOSS_FUNCTION,
                      optimizer=Common.get_sgd_optimizer(),
                      metrics=[constant.METRIC_ACCURACY])

        self.vgg.summary()

    def get_model(self):
        return self.vgg

    def train(self, n_epochs = 2, batch = 32):
        self.vgg.fit(self.objects, self.labels,
                  batch_size=batch,
                  epochs=n_epochs, shuffle=True)

    def __process_img(self, img):
        img = image.load_img(img)
        img = image.img_to_array(img)
        return preprocess_input(expand_dims(img, axis=0), version=2)

    def predict(self, img_path):
        pixels = self.__process_img(img_path)
        preds = self.vgg.predict(pixels)
        return decode_predictions(preds)

    def evaluate(self):
        super(VggModel, self).evaluate()