from .base import GenericCNN, LeNet5
from .torchvision_wrappers import ResNet18Custom, VGG11Custom, MobileNetV2Custom

def get_model_for_dataset(dataset_name, model_type="generic"):
    """
    Returns the appropriate model class based on dataset and model type.
    Returns the class itself, not an instance or a lambda, to ensure picklability.
    """
    # Dataset parameters
    params = {
        "mnist": {"in_channels": 1, "input_size": 28, "num_classes": 10},
        "fmnist": {"in_channels": 1, "input_size": 28, "num_classes": 10},
        "emnist": {"in_channels": 1, "input_size": 28, "num_classes": 10},
        "cifar10": {"in_channels": 3, "input_size": 32, "num_classes": 10},
    }
    
    dataset_params = params.get(dataset_name.lower(), params["mnist"])
    
    # Model selection mapping
    # We return a "wrapper" class that handles the specific params for that dataset
    # Instead of lambda, we define a simple dynamic class or just return the base
    
    if model_type == "resnet18":
        # We can't easily create a class with params inside this function and keep it picklable
        # So we'll use a modified approach in the Client.
        return ResNet18Custom
    elif model_type == "vgg11":
        return VGG11Custom
    elif model_type == "mobilenet":
        return MobileNetV2Custom
    elif model_type == "lenet":
        return LeNet5
    else:
        return GenericCNN

