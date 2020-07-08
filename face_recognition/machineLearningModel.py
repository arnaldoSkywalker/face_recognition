import abc
from abc import abstractmethod

class MLModel(metaclass=abc.ABCMeta):

    def __init__(self, dataSet=None):
        if dataSet is not None:
            self.objects = dataSet.objects
            self.labels = dataSet.labels
            self.obj_validation = dataSet.obj_validation
            self.labels_validation = dataSet.labels_validation
            self.number_labels = dataSet.number_labels
        self.init_model()

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, object):
        pass
