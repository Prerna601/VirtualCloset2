import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from train_model import FashionCNN  # Import the trained model class

# Load class labels (same as during training)
class_labels = ['Blazer', 'Celana_Panjang', 'Celana_Pendek', 'Gaun', 'Hoodie', 
                'Jaket', 'Jaket_Denim', 'Jaket_Olahraga', 'Jeans', 'Kaos', 
                'Kemeja', 'Mantel', 'Polo', 'Rok', 'Sweter']

# Define transformation (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize image to match model input
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Same normalization as training
])

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionCNN(num_classes=len(class_labels)).to(device)
model.load_state_dict(torch.load("fashion_classifier.pth", map_location=device))
model.eval()  # Set model to evaluation mode

def predict_image(image_path):
    """Loads an image, preprocesses it, and predicts its class."""
    img = Image.open(image_path).convert("RGB")  # Ensure image is RGB
    img = transform(img).unsqueeze(0)  # Add batch dimension
    
    img = img.to(device)  # Move to GPU if available

    with torch.no_grad():
        output = model(img)
        predicted_class = torch.argmax(output).item()
    
    return class_labels[predicted_class]

# Test the model with a new image
test_image_path = "dataset/Clothes_Dataset/test/Sweter/00588f87-cda1-46d7-93be-71f0fa889895.jpg"  # Update if needed
print(f"🖼️ Testing with image: {test_image_path}")

if os.path.exists(test_image_path):
    predicted_category = predict_image(test_image_path)
    print(f"🛍️ Predicted Category: {predicted_category}")  # 🔥 Now prints the prediction
else:
    print(f"❌ Error: Image '{test_image_path}' not found!")
