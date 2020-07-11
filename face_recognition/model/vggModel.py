from keras.preprocessing import image
from keras_vggface.utils import decode_predictions
from keras_vggface.utils import preprocess_input
from numpy import expand_dims
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import MaxPooling2D, Dense, Flatten, Dropout, Convolution2D
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
        base_model = VGG16(weights=constant.IMAGENET, include_top=False,
                           input_tensor=Input(shape=(constant.IMG_WIDTH,
                           constant.IMG_HEIGHT, 3)))
        base_model.summary()

        layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
        x = layer_dict['block2_pool'].output

        # Stacking a new simple convolutional network on top of it
        x = Convolution2D(64, 3, padding=constant.PADDING_SAME)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(constant.NUMBER_FULLY_CONNECTED, activation=constant.RELU_ACTIVATION_FUNCTION)(x)
        x = Dropout(constant.DROP_OUT_O_25)(x)
        x = Dense(self.n_classes, activation=constant.SOFTMAX_ACTIVATION_FUNCTION)(x)

        self.vgg = Model(inputs=base_model.input, outputs=x)

        for layer in self.vgg.layers:
            if layer.name in layer_dict:
                layer.trainable = False

        self.vgg.summary()

    def get_model(self):
        return self.vgg

    def train(self, n_epochs = 50, batch = 32):
        self.vgg.fit(self.objects, self.labels,
                  batch_size=batch,
                  epochs=n_epochs)

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