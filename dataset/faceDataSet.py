import abc
import os
import random
from abc import abstractmethod

import numpy
from keras_preprocessing.image import load_img, img_to_array
from skimage import io
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.utils import np_utils

from util import constant
from util.common import Common


class FaceDataSet(metaclass=abc.ABCMeta):

    def __init__(self, path,  extension_list):
        self.path = path
        self.ext_list = extension_list
        self.objects = []
        self.labels = []
        self.obj_validation = []
        self.labels_validation = []
        self.number_labels = 0

    def get_data(self, keras_img_processing=False):
        img_path_list = os.listdir(self.path)
        self.objects, self.labels = self.fetch_img_path(img_path_list, self.path, keras_img_processing)
        self.process_data(keras_img_processing)
        self.print_dataSet()

    def split_training_set(self):
        return train_test_split(self.objects, self.labels, test_size=0.3,
                                random_state=random.randint(0, 100))

    def process_data(self, keras_img_processing):
        self.objects, self.img_obj_validation, self.labels, self.img_labels_validation = \
            self.split_training_set()
        self.labels = np_utils.to_categorical(self.labels, self.number_labels)
        self.labels_validation = np_utils.to_categorical(self.img_labels_validation, self.number_labels)

        if keras_img_processing:
            objs = numpy.empty((len(self.labels), constant.IMG_WIDTH, constant.IMG_HEIGHT, 3))
            self.obj_validation = numpy.empty((len(self.labels_validation), constant.IMG_WIDTH, constant.IMG_HEIGHT, 3))
            for o in self.objects:
                numpy.append(objs, o)
            for o in self.img_obj_validation:
                numpy.append(self.obj_validation, o)
            self.objects = objs
        else:
            self.objects = Common.reshape_transform_data(self.objects)
            self.obj_validation = Common.reshape_transform_data(self.img_obj_validation)

    def fetch_img_path(self, img_path_list, path, keras_img_processing):
        images = []
        labels = []
        for img_path in img_path_list:
            if self.__check_ext(img_path):
                img_abs_path = os.path.abspath(os.path.join(path, img_path))
                if keras_img_processing:
                    image = load_img(img_abs_path, target_size=(constant.IMG_WIDTH,
                                                                constant.IMG_HEIGHT))
                    image = img_to_array(image)
                    image = Common.reshape_from_img(image)
                    image = preprocess_input(image)
                else:
                    image = io.imread(img_abs_path, as_gray=True)
                label = self.process_label(img_path)
                images.append(image)
                labels.append(label)
        return images, labels

    def __check_ext(self, file_path):
        for ext in self.ext_list:
            if file_path.endswith(ext):
                return True
        return False

    def print_dataSet(self):
        print(self.objects)
        print(self.labels)

    @abstractmethod
    def process_label(self, file_path):
        pass
