import sys, inspect
from .iae import Iae
from .pc_mae import Pc_Mae
from .pointnext import Pointnext

# Class returns unified list of model wrapper classes


def build_models(model_name_list):
    models = []
    for model_name in model_name_list:
        if model_name in inspect.getmembers(sys.modules[__name__]):  # check if model_name exists in model classes
            models.push(eval(model_name + '()'))
        else:
            raise BaseException('model_name cannot be instanciated')

    return models
