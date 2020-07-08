import abc
import os
import random
from skimage import io
from abc import abstractmethod
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils
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

    def get_data(self):
        img_path_list = os.listdir(self.path)
        self.objects, self.labels = self.fetch_img_path(img_path_list, self.path)
        self.process_data()
        self.print_dataSet()

    def process_data(self):
        self.objects, self.obj_validation, self.labels, self.labels_validation = \
            train_test_split(self.objects, self.labels, test_size=0.3, random_state=random.randint(0, 100))

        self.objects = Common.reshape_transform_data(self.objects)
        self.obj_validation = Common.reshape_transform_data(self.obj_validation)
        self.labels = np_utils.to_categorical(self.labels, self.number_labels)
        self.labels_validation = np_utils.to_categorical(self.labels_validation, self.number_labels)

    def fetch_img_path(self, img_path_list, path):
        images = []
        labels = []
        for img_path in img_path_list:
            if self.__check_ext(img_path):
                img_abs_path = os.path.abspath(os.path.join(path, img_path))
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
