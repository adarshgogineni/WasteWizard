import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import os
import torch.nn as nn

# Define the class for your model, it should match the one used during training
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.network = models.resnet50(weights = None)  # Assuming we used ResNet50
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, xb):
        return self.network(xb)

def load_model(model_path, num_classes, device):
    # Initialize the model
    model = ResNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Set model to evaluate mode
    return model

def predict_image(image_path, model, device, transformations):
    image = Image.open(image_path)
    image_tensor = transformations(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_index = torch.max(probabilities, dim=1)
    return predicted_index.item()  # Return the index of the predicted class

def main():
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define your transformations
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
  
    
    # Set the directory containing your images
    image_dir = 'testimages'  # Change this to the path of your image folder
    image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.jpg')]  # Assuming they are jpg images
    
    # Load the model
    num_classes = 6  # Change this to the number of classes in your model
    model_path = 'garbage_classification_model.pth'
    model = load_model(model_path, num_classes, device)
    
    # Class labels (replace these with your actual labels)
    class_labels = ['metal', 'paper', 'glass', 'trash', 'plastic', 'cadrboard']  # Replace with actual class labels
    
    # Predict and print results
    for image_path in image_files:
        predicted_index = predict_image(image_path, model, device, transformations)
        print(f'Image: {image_path}, Predicted class: {class_labels[predicted_index]}')

if __name__ == '__main__':
    main()
