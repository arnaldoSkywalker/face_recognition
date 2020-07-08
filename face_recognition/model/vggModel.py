from keras_vggface import VGGFace

from face_recognition.machineLearningModel import MLModel
from util import constant

class VggModel(MLModel):

    def __init__(self, dataSet=None):
        super().__init__(dataSet)

    def init_model(self):
        self.model = VGGFace(model=constant.RESNET_MODEL, include_top=False,
                             input_shape=(constant.IMG_WIDTH, constant.IMG_HEIGHT, 1),
                             pooling=constant.AVG_POOLING)

    def train(self):
        raise Exception("Unsupported operation in already trained model")

    def predict(self, object):
        return self.model.predict(object)