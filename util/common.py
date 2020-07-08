import numpy
from util import constant


class Common:

    @staticmethod
    def reshape_transform_data(data):
        data = numpy.array(data)
        result = data.reshape(data.shape[0], constant.IMG_WIDTH, constant.IMG_HEIGHT, 1)
        return Common.to_float(result)

    @staticmethod
    def to_float(value):
        return value.astype('float32')/255

