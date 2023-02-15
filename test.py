import torch
import seaborn as sn
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import constants
from sklearn.metrics import confusion_matrix
from data_loader import to_device, test_dataset, train_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from model.MobileNet import MobileNetV1
from model.ResNet import ResNet34

defaul_device = torch.device("cuda")

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), defaul_device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return train_dataset.classes[preds[0].item()]

def predict():
    model = ResNet34(
        image_channels=constants.image_channels,
        num_classes=constants.NUM_OF_FEATURES
    )
    model.cuda()
    model.eval()
    checkpoint = torch.load(constants.CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    y_true = []
    y_pred = []
    index = 0
    correct = 0
    wrong_total = 0
    for current_image in tqdm(test_dataset):
        index+=1
        img, label, path = current_image
        true_label = train_dataset.classes[label]
        pred_label = predict_image(img, model)
        y_true.append(true_label)
        y_pred.append(pred_label) 
        if pred_label==true_label:
            correct+=1
        else:
            wrong_total+=1
    matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1_score_mesuare = f1_score(y_true, y_pred,average="macro")
    print("precision = ", precision)
    print("recall = ", recall)
    print("F1_score = ", f1_score_mesuare)
    sn.set(font_scale=1.4)
    sn.heatmap(matrix, annot=True, annot_kws={"size": 16})
    plt.imshow(matrix, cmap="binary")
    plt.show()

if __name__=='__main__':
    predict()