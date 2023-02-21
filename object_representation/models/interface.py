"""
Feature Extractor is interface for using feature extractors for point clouds.
If you want to use features from a new model you have to define a folder like the others and create the extractor
using this interface.

"""


class FeatureExtractor:

    def __init__(self):
        pass

    def load_model(self):
        pass

    @classmethod
    def __preprocess(cls, data):
        return data

    @classmethod
    def __call_model(cls, data):
        pass

    def extract(self, data):
        data = self.__preprocess(data)
        return self.__call_model(data)


