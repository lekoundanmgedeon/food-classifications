import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageDataset
from model_baseline import simple_cnn

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

model = simple_cnn(num_classes=10)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

for epoch in range(10):

    for images,labels in loader:

        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch",epoch,"loss",loss.item())