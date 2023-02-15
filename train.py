import torch
from utils import constants, default_device
from data_loader import to_device, train_dl, val_dl
from utils.visualization import plot_losses, plot_accuracies, plot_lrs
from fit import fit_one_cycle
from utils.helper import evaluate
# from model.MobileNet import MobileNetV1
from model.ResNet import ResNet34
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
default_device = default_device.device

def training():
    model_MobileNetV1= to_device(ResNet34(3, constants.NUM_OF_FEATURES), default_device)
    opt_func = torch.optim.Adam
    history = fit_one_cycle(constants.NUM_OF_EPOCHS, constants.MAX_LEARNING_RATE, model_MobileNetV1, train_dl, val_dl,
                             grad_clip=constants.GRAD_CLIP,
                             weight_decay=constants.WEIGHT_DECAY,
                             opt_func=opt_func)
    plot_accuracies(history)
    plot_losses(history)
    plot_lrs(history)

    result = evaluate(model_MobileNetV1, val_dl)
    print(result)

training()