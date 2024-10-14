import torch
import torchvision.models as models  # Replace with your ViT model if needed
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# CIFAR-10 class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Define the model architecture (replace with your ViT if needed)
model = models.resnet18(num_classes=10)  # Use your custom model here

# Load the model weights
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the prediction function
def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]

# Create Gradio interface
interface = gr.Interface(fn=predict, 
                         inputs=gr.Image(type="pil"), 
                         outputs="label", 
                         title="CIFAR-10 Image Classification")

# Launch the app
interface.launch()

