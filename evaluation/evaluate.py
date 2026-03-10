import torch
from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, loader, device):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images,labels in loader:

            images = images.to(device)

            outputs = model(images)
            _,pred = torch.max(outputs,1)

            y_true.extend(labels.numpy())
            y_pred.extend(pred.cpu().numpy())

    acc = accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred,average="macro")

    print("Accuracy:",acc)
    print("F1-score:",f1)