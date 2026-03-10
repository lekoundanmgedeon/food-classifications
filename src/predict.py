import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model_baseline import simple_cnn, resnet18_transfer
from dataset import ImageDataset
import pandas as pd
import numpy as np


def load_model(model_path, num_classes=10, model_type='simple_cnn', device='cpu'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to saved model checkpoint
        num_classes: Number of output classes
        model_type: 'simple_cnn' or 'resnet18_transfer'
        device: 'cpu' or 'cuda'
    
    Returns:
        model: Loaded model on specified device
    """
    if model_type == 'simple_cnn':
        model = simple_cnn(num_classes)
    elif model_type == 'resnet18_transfer':
        model = resnet18_transfer(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def predict_on_dataset(model, data_loader, device='cpu'):

    model.eval()

    all_predictions = []
    all_probabilities = []
    all_ids = []

    with torch.no_grad():

        for images, ids in data_loader:

            images = images.to(device)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_ids.extend(ids)

    return np.array(all_ids), np.array(all_predictions), np.array(all_probabilities)


def generate_submission(ids, predictions, output_file='../submissions/submission.csv'):

    submission_df = pd.DataFrame({
        'id': ids,
        'prediction': predictions
    })

    submission_df.to_csv(output_file, index=False)

    print(f"Submission saved to {output_file}")

    return submission_df

def main():
    """Main prediction pipeline"""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 10
    model_type = 'simple_cnn'  # Change to 'resnet18_transfer' if needed
    model_path = '../models/model_checkpoint.pt'  # Path to trained model
    test_csv = '../data/test.csv'
    test_img_dir = '../data/test/'
    
    # Setup transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, num_classes, model_type, device)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = ImageDataset(
        test_csv,
        test_img_dir,
        transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Make predictions
    print("Making predictions...")
    ids, predictions, probabilities = predict_on_dataset(model, test_loader, device)
   
    # Generate submission
    print("Generating submission file...")
    generate_submission(ids, predictions)
    
    print("Done!")
    print(f"Sample predictions: {predictions[:10]}")
    
    return predictions, probabilities


if __name__ == '__main__':
    predictions, probabilities = main()