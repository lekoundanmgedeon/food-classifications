import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from model_baseline import simple_cnn
import os
import sys

# Ajouter le chemin pour importer depuis le dossier evaluation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluate import evaluate 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "train")

# transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset & Loaders ---
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- Model, Loss, Optimizer ---
model = simple_cnn(num_classes=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_acc = 0

for epoch in range(10):
    # PHASE D'ENTRAÎNEMENT
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # PHASE D'ÉVALUATION (Utilisation du module externe)
    print(f"\nEpoch {epoch+1}/10")
    val_acc, val_f1 = evaluate(model, val_loader, device)

    # SAUVEGARDE DU MEILLEUR MODÈLE
    if val_acc > best_acc:
        best_acc = val_acc
        save_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "baseline_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"--> Nouveau record ! Modèle sauvegardé : {save_path}")

print("\nEntraînement terminé.")
print(f"Meilleure Accuracy : {best_acc:.4f}")