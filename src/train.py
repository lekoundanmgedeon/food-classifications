import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageDataset
from model_baseline import simple_cnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = ImageDataset(
    "../data/train.csv",
    "../data/train/",
    transform
)

loader = DataLoader(dataset,batch_size=32,shuffle=True)

model = simple_cnn(num_classes=10).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


for epoch in range(10):

    correct = 0
    total = 0

    for images,labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total

    print(f"Epoch {epoch+1} | Loss {loss.item():.4f} | Accuracy {acc:.4f}")


torch.save(model.state_dict(),"../models/baseline_model.pth")